# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Fri Sep 19 11:59:13 2025"
__email__    = "xuhan@ihep.ac.cn"


"""
Description: 
Updated with Randomized Subspace Accumulation (Matrix-Free Method) for 
high-performance coherent mode extraction from Monte-Carlo wavefronts.
"""

#-----------------------------------------------------------------------------#
# modules

from cat.source import _multi, _constant
from cat.source._srw_utils import _undulator
from cat.source._srw_utils import _srw_electron_beam
from cat.source._srw_utils import _propagate_wave_front

from cat.source import _support
from cat.source import _file_utils

from srwpy.srwlib import *

from scipy import fft
from scipy import interpolate
from skimage.restoration import unwrap_phase

import scipy.sparse.linalg as ssl
import h5py as h5
import numpy as np

import time
import psutil
import random
import sys
import os
import gc

#-----------------------------------------------------------------------------#
# parameters

_N_SVD_STEP = 5000
_N_WFR = 50000
_N_STEP = 10
_N_CMODE = 200 # Final modes to keep

#-----------------------------------------------------------------------------#
# functions

#-----------------------------------------------------------
# support function

def processing_bar(current_count, total_count, interval_time, start_time):
    
    print("\r", end = "")
    print(
        "processs: {}%: ".format(int(100*(current_count + 1) / total_count)), 
        "▋" * int(1 + 10*current_count / total_count), end = ""
        )
    print("  time cost: %.2f min" % (np.abs(interval_time - start_time)/60), end = "")
    
    sys.stdout.flush()
    time.sleep(0.005)

def fresnel_bluestein(
        field_input, wavelength, distance, pixel_in, pixel_out, count_out, 
        center_shift = (0, 0)
        ):
    
    """
    Corrected Bluestein propagator with descriptive variable names.
    """

    #---------------------------------------------------
    # unpack parameters
    
    vertical_count, horizontal_count = np.shape(field_input)
    x_pixel_in, y_pixel_in = pixel_in
    x_pixel_out, y_pixel_out = pixel_out
    x_count_out, y_count_out = count_out
    x_shift, y_shift = center_shift
    
    k_vector = 2 * np.pi / wavelength

    #---------------------------------------------------
    # 1. Quadratic Phase Factors (Physical Coordinates)
    
    x_coord_in = (np.arange(horizontal_count) - horizontal_count/2 + 0.5) * x_pixel_in
    y_coord_in = (np.arange(vertical_count) - vertical_count/2 + 0.5) * y_pixel_in
    
    x_coord_out = (
        np.arange(x_count_out) - x_count_out/2 + 0.5
        ) * x_pixel_out + x_shift
    y_coord_out = (
        np.arange(y_count_out) - y_count_out/2 + 0.5
        ) * y_pixel_out + y_shift
    
    phase_quadratic_source = np.exp(
        1j * k_vector / (2 * distance) * (
            x_coord_in[np.newaxis, :]**2 + 
            y_coord_in[:, np.newaxis]**2
            )
        )
    
    phase_quadratic_end = np.exp(
        1j * k_vector / (2 * distance) * (
            x_coord_out[np.newaxis, :]**2 + 
            y_coord_out[:, np.newaxis]**2
            )
        )

    #---------------------------------------------------
    # 2. Vectorized 1D Bluestein Core 
    
    def _bluestein_fft_1d_descriptive(
            vector_input, input_count, output_count, input_pixel, output_pixel, 
            shift_value, prop_distance, wave_len
            ):
        
        bluestein_coefficient = input_pixel * output_pixel / (wave_len * prop_distance)
        
        input_center_idx = (input_count - 1) / 2.0
        output_center_effective = (output_count - 1) / 2.0 - shift_value / output_pixel
        
        idx_input = np.arange(input_count)
        idx_output = np.arange(output_count)
        
        phase_pre_chirp = np.exp(
            -1j * 2 * np.pi * bluestein_coefficient * 
            (idx_input**2 / 2.0 - idx_input * output_center_effective)
            )
        
        phase_post_chirp = np.exp(
            -1j * 2 * np.pi * bluestein_coefficient * (
                idx_output**2 / 2.0 - idx_output * input_center_idx + 
                input_center_idx * output_center_effective
                )
            )
        
        fft_convolution_size = int(2**np.ceil(np.log2(input_count + output_count - 1)))
        
        kernel_spectrum = np.zeros(fft_convolution_size, dtype=complex)
        
        idx_kernel_pos = np.arange(output_count)
        kernel_spectrum[:output_count] = np.exp(
            1j * np.pi * bluestein_coefficient * idx_kernel_pos**2
            )
        
        idx_kernel_neg = np.arange(1, input_count)
        kernel_spectrum[-input_count+1:] = np.exp(
            1j * np.pi * bluestein_coefficient * idx_kernel_neg[::-1]**2
            )
        
        if vector_input.ndim == 1:
            input_signal = vector_input * phase_pre_chirp
        else:
            input_signal = vector_input * phase_pre_chirp[np.newaxis, :]
            
        fft_input = fft.fft(input_signal, fft_convolution_size, axis=-1)
        fft_kernel = fft.fft(kernel_spectrum, fft_convolution_size, axis=-1)
        
        convolved_spectrum = fft_input * fft_kernel
        raw_convolution_output = fft.ifft(convolved_spectrum, axis=-1)
        
        final_output = (
            raw_convolution_output[..., :output_count] * phase_post_chirp[np.newaxis, :]
            )
        
        return final_output

    #---------------------------------------------------
    # 3. Main Propagation Sequence
    
    bluestein_input = field_input * phase_quadratic_source
    
    field_horizontal = _bluestein_fft_1d_descriptive(
        bluestein_input, horizontal_count, x_count_out, 
        x_pixel_in, x_pixel_out, 
        x_shift, distance, wavelength
        )
    
    field_horizontal_transposed = field_horizontal.T 
    field_vertical_transposed = _bluestein_fft_1d_descriptive(
        field_horizontal_transposed, vertical_count, y_count_out, 
        y_pixel_in, y_pixel_out, 
        y_shift, distance, wavelength
        )
    
    bluestein_output = field_vertical_transposed.T
    
    #---------------------------------------------------
    # 4. Normalization & Final Phase
    
    field_output = bluestein_output * phase_quadratic_end * (
        (x_pixel_in * y_pixel_in) / (1j * wavelength * distance)
        ) 
    
    return field_output

#-----------------------------------------------------------
# wavefront calculation function

def _cal_srw_spectrum(
        undulator, electron_beam, energy_range = None, n_points = 100
        ):
    
    print("spectrum calculation start..... ", flush = True)
    
    import scipy.constants as codata
    
    def _cal_aperture(undulator, electron_beam):
    
        L_und = undulator['period_length'] * undulator['period_number']
        sig_r_prime = np.sqrt(wavelength / L_und)
        
        aperture = list()
        
        for sigma_div in [
                electron_beam['sigma_xd'], electron_beam['sigma_yd']
                ]:
            
            sigma_total_prime = np.sqrt(sig_r_prime**2 + sigma_div**2)
            sigma_size = sigma_total_prime * 30
            aperture.append(sigma_size * 3)
        
        return aperture
    
    def _red_shift(undulator, electron_beam):
        
        theta_eff = np.sqrt(
            electron_beam['sigma_xd']**2 + electron_beam['sigma_yd']**2
            ) / 1.5
        shift_factor = (gamma * theta_eff)**2
        energy_shift = undulator['hormonic_energy'] * shift_factor
        
        return energy_shift
        
    #---------------------------------------------------
    # construction of undualtor
    
    magnetic_field_harmonic = SRWLMagFldH() 
    magnetic_field_harmonic.n = undulator['n_hormonic']
    magnetic_field_harmonic.h_or_v = undulator['direction']
    _mult = float(93.37290413839085)
    
    gamma = electron_beam['energy'] / _constant._E_r
    wavelength = (
        (codata.c * codata.h / codata.e) / 
        undulator['hormonic_energy']
        )
    k_value = np.sqrt(
        2 * ((
            (2 * undulator['n_hormonic'] * wavelength * gamma**2) / 
            undulator['period_length']
            ) - 1))
    magnetic_field_harmonic.B = k_value / (_mult * undulator['period_length'])
    
    und = SRWLMagFldU([magnetic_field_harmonic])
    und.per = undulator['period_length'] 
    und.nPer = undulator['period_number'] 
    magnetic_field_containter = SRWLMagFldC(
        [und], array('d', [0]), array('d', [0]), array('d', [0])
        ) 
    
    #---------------------------------------------------
    # Electron Beam
    
    eBeam = SRWLPartBeam()
    
    eBeam.Iavg = electron_beam['current']
    eBeam.partStatMom1.x = 0
    eBeam.partStatMom1.y = 0
    eBeam.partStatMom1.z = 0
    
    eBeam.partStatMom1.gamma = gamma
    eBeam.partStatMom1.xp = 0
    eBeam.partStatMom1.yp = 0
    
    eBeam.arStatMom2[0 : 6] = array('d', [
        electron_beam['sigma_x0']**2, 0, electron_beam['sigma_xd']**2,
        electron_beam['sigma_y0']**2, 0, electron_beam['sigma_yd']**2,
        ])
    eBeam.arStatMom2[10] = electron_beam['energy_spread']**2

    #---------------------------------------------------
    # UR Stokes Parameters (mesh) for Spectral Flux
    
    aperture = _cal_aperture(undulator, electron_beam)
    energy_shift = _red_shift(undulator, electron_beam)
    nperiod = undulator['n_hormonic'] * undulator['period_number']
    
    if energy_range is None:
        energy_end = undulator['hormonic_energy'] * (1 + 1.5 / nperiod) - energy_shift
        energy_start = undulator['hormonic_energy'] * (1 - 4.0 / nperiod) - energy_shift
    else:
        energy_start, energy_end = energy_range
    
    stkF = SRWLStokes() 
    stkF.allocate(int(n_points), 1, 1) 
    stkF.mesh.zStart = 30
    stkF.mesh.eStart = float(energy_start)
    stkF.mesh.eFin = float(energy_end)
    stkF.mesh.xStart = -float(aperture[0]/2)
    stkF.mesh.xFin = float(aperture[0]/2)
    stkF.mesh.yStart = -float(aperture[1]/2)
    stkF.mesh.yFin = float(aperture[1]/2)
    
    srwl.CalcStokesUR(
        stkF, eBeam, und, 
        [1, int(undulator['n_hormonic'] + 2), 1.5, 1.5, 1]
        )

    return (
        np.linspace(stkF.mesh.eStart, stkF.mesh.eFin, stkF.mesh.ne),
        np.abs(stkF.arS[: int(n_points)])
        )

def _cal_energy_wfr(undulator, electron_beam, screen, energy_relative, photon_energy):
    
    #-----------------------------------------------------------
    # reference wavefront calcualtion
    
    und = _undulator(undulator)
    und.wave_length()
    und.cal_k(electron_beam_energy = electron_beam["energy"])
    
    und_magnetic_structure = und.magnetic_structure()
    
    e_beam = _srw_electron_beam(
        electron_beam, und.n_period, und.period_length, 
        und.n_hormonic
        )
    e_beam.monte_carlo()
    
    # calcualte the wavefront
    
    _part_beam, wavelength_resonance, resonance_energy = e_beam.after_monte_carlo(
        [0, 0, 0, 0, energy_relative], 
        und.period_length, und.k_vertical, und.k_horizontal
        )
    
    wavefront = _propagate_wave_front(screen, photon_energy)
    wavelength = 12398.4 / photon_energy * 1e-10
    wavefront._cal_wave_front(_part_beam, und_magnetic_structure) 
    
    # only s polarization is considered
    wavefront = np.reshape(
        wavefront.wfr.arEx, 
        (screen['nx'], screen['ny'], 2)
        )
    wavefront = wavefront[:, :, 0] - 1j * wavefront[: ,:, 1]

    #-----------------------------------------------------------
    # proapgate to source
    
    xpixel = float(abs(screen['xfin'] - screen['xstart']) / screen['nx'])
    ypixel = float(abs(screen['yfin'] - screen['ystart']) / screen['ny'])
    
    source_wfr = fresnel_bluestein(
        np.conjugate(wavefront), wavelength, -screen['screen'] + e_beam.initial_z,  
        [xpixel, ypixel], [float(xpixel/3.125), float(ypixel/3.125)],
        [int(screen['nx']/2), int(screen['ny']/2)]
        )
    
    return photon_energy, np.array(source_wfr, dtype = np.complex64)

def _cal_reference_wfr(
        undulator, electron_beam, screen, photon_energy, sampling = 100, 
        sigma_ratio = 3
        ):
    
    # ±3 responding to 99.73% area of gaussian
    energy_relative = np.linspace(-sigma_ratio, sigma_ratio, int(sampling))
    
    reference_wfrs = list()
    reference_energy = list()

    start_time = time.time()
    for idx in range(int(sampling)):
        
        interval_time = time.time()
        processing_bar(idx, int(sampling), interval_time, start_time)
        
        references = _cal_energy_wfr(
            undulator, electron_beam, screen, energy_relative[idx], photon_energy
            )
        reference_energy.append(references[0])
        reference_wfrs.append(references[1])
    
    return reference_wfrs, reference_energy
    
def _mc_brightness_convolution(
        undulator, electron_beam, screen, photon_energy, n_step = 10, 
        n_energy_sampling = 100, step_n_vector = 500, sigma_ratio = 3, 
        file_name = None
        ):
    
    screen = screen.copy()
    print("Reference wavefront calculation start..... ", flush = True)
    
    newSeed = random.randint(0, 1000000); random.seed(newSeed)
    n_electron = electron_beam['n_electron']
    n_cmode = screen['n_vector'] 
    
    # PDF for energy sampling
    probability = (1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * np.linspace(-sigma_ratio, sigma_ratio, int(n_energy_sampling))**2))
    probability /= np.sum(probability)
    probability = np.array(probability * n_electron, dtype = int)
    probability[int(n_energy_sampling/2)]  +=  int(n_electron - sum(probability))
    
    und = _undulator(undulator)
    und.wave_length()
    und.cal_k(electron_beam_energy = electron_beam["energy"])
    
    e_beam = _srw_electron_beam(electron_beam, und.n_period, und.period_length, undulator['n_hormonic'])
    screen['screen'] = screen['screen'] - e_beam.initial_z/2
    e_beam.monte_carlo()
    
    reference_wfrs, reference_energy = _cal_reference_wfr(
        undulator, electron_beam, screen, photon_energy, sampling = n_energy_sampling, sigma_ratio = sigma_ratio
        )
    
    # Coordinate system setup
    xpixel = float(abs(screen['xfin'] - screen['xstart']) / screen['nx'] / 3.125)
    ypixel = float(abs(screen['yfin'] - screen['ystart']) / screen['ny'] / 3.125)
    screen['nx'] = int(screen['nx'] / 2); screen['ny'] = int(screen['ny'] / 2)
    screen['xstart'] = -screen['nx'] * xpixel / 2; screen['ystart'] = -screen['ny'] * ypixel / 2
    screen['xfin'] = screen['nx'] * xpixel / 2; screen['yfin'] = screen['ny'] * ypixel / 2
    
    xtick = np.linspace(screen['xstart'], screen['xfin'], screen['nx'])
    ytick = np.linspace(screen['ystart'], screen['yfin'], screen['ny'])
    gridx, gridy = np.meshgrid(xtick, ytick)
    
    n_pixels_flat = int(screen['nx'] * screen['ny'])
    temp_wfrs = np.zeros(
        (int(n_electron), n_pixels_flat), dtype = np.complex64
        )

    print("\n wavefront calculation start.....", flush = True)
    print(time.asctime(time.localtime(time.time())), flush = True)
    
    probability_count = 0; energy_idx = 0
    
    for idx in range(int(n_electron)):
            
        mc_rand_array = [random.gauss(0, 1) for ir in range(5)]
        mc_rand_array[4] = np.linspace(-sigma_ratio, sigma_ratio, int(n_energy_sampling))[energy_idx]
        
        e_beam.monte_carlo()
        part_beam, wavelength, resonance_energy = e_beam.after_monte_carlo(
            mc_rand_array, und.period_length, und.k_vertical, und.k_horizontal
            )
        k_vector = 2 * np.pi / wavelength
        reference_wfr_idx = reference_wfrs[energy_idx]
        
        vibration_wfr = _support._vibration_shift(
            reference_wfr_idx, k_vector, [gridx, gridy], [xtick, ytick], screen,
            [part_beam.partStatMom1.xp, part_beam.partStatMom1.yp], 
            [part_beam.partStatMom1.x, part_beam.partStatMom1.y], 
            )
                
        temp_wfrs[idx, :] = np.reshape(vibration_wfr, (n_pixels_flat, )) 
        
        probability_count += 1
        if probability_count > int(probability[energy_idx]):
            energy_idx += 1
            probability_count = 0
        
    print("Final SVD calculation start..... ", flush = True)
    print(time.asctime(time.localtime(time.time())), flush = True)
    
    vector, value, evolution = ssl.svds(
        temp_wfrs.T, k = int(n_cmode + 10), 
        )
    vector = np.copy(vector[:, ::-1], order = 'C')
    value = np.copy(np.abs(value[::-1], order = 'C'))
    
    #------------------------------------------------------
    # save_results
    
    print("source file generation start..... ", flush = True)
    print(time.asctime(time.localtime(time.time())), flush = True)
    
    _file_utils._construct_source_file(
        file_name, electron_beam, undulator, screen, wavelength
        )
    cmode = vector[:, 0 : int(n_cmode)]
    
    with h5.File(file_name, "a") as f:
        
        coherence_group = f.create_group("coherence")
        coherence_group.create_dataset(
            "eig_value", data = value[0 : int(n_cmode)]
            )
        coherence_group.create_dataset("eig_vector", data = cmode) 
        
    print("Finished", flush = True)
    print(time.asctime(time.localtime(time.time())), flush = True)

        
#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    pass