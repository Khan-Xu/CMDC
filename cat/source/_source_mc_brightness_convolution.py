# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Fri Sep 19 11:59:13 2025"
__email__    = "xuhan@ihep.ac.cn"


"""
Description
"""

#-----------------------------------------------------------------------------#
# modules

import sys
import time
import psutil
import random

import h5py as h5
import numpy as np
from scipy import fft
from scipy import interpolate
import scipy.sparse.linalg as ssl
from skimage.restoration import unwrap_phase

from cat.source import _multi
from cat.source._srw_utils import _undulator
from cat.source._srw_utils import _srw_electron_beam
from cat.source._srw_utils import _propagate_wave_front

from cat.source import _support
from cat.source import _file_utils

#-----------------------------------------------------------------------------#
# parameters

_N_SVD_STEP = 5000
_N_WFR = 50000
_N_STEP = 10
_N_CMODE = 200

#-----------------------------------------------------------------------------#
# functions

def processing_bar(current_count, total_count, interval_time, start_time):
    
    print("\r", end = "")
    print(
        "propagate processs: {}%: ".format(int(100*(current_count + 1) / total_count)), 
        "▋" * int(1 + 10*current_count / total_count), end = ""
        )
    print("  time cost: %.2f min" % (np.abs(interval_time - start_time)/60), end = "")
    
    sys.stdout.flush()
    time.sleep(0.005)

def _fourier_shift(data, offset):

    """
    ---------------------------------------------------------------------------
    Shift a 2D array in the frequency domain using the Fourier transform.

    Args:
        data (numpy.ndarray): A 2D array of shape (M, N), representing the 
        input data to be shifted.
        offset (tuple of float): A tuple (dy, dx) where dy is the shift amount 
        in the row direction (y-axis) and dx is the shift amount in the column 
        direction (x-axis). Positive values shift downward or rightward, 
        respectively.
    
    Returns:
        numpy.ndarray: The shifted 2D array 
    ---------------------------------------------------------------------------
    """
    
    x_count, y_count = data.shape

    #------------------------------------------------------
    # construct y shift phase
    
    y_meshgrid = np.hstack([
        np.arange(np.floor(x_count/2), dtype = np.int),
        np.arange(np.floor(-x_count/2), 0, dtype = np.int)
        ])
    y_shift = np.exp(-1j * 2 * np.pi * offset[0] * y_meshgrid / y_count)
    y_shift = y_shift[None, :] 
    
    #------------------------------------------------------
    # construct x shift phase
    
    x_meshgrid = np.hstack([
        np.arange(np.floor(y_count/2), dtype = np.int),
        np.arange(np.floor(-y_count/2), 0, dtype = np.int)
        ])
    x_shift = np.exp(-1j * 2 * np.pi * offset[1] * x_meshgrid / x_count)
    x_shift = x_shift[:, None]  
    
    #------------------------------------------------------
    # shifted data
    
    shifted_data = np.fft.ifft2(
        np.fft.fft2(data) * (x_shift * y_shift)
        )
    
    return shifted_data

def _shift_impluse_geometry(xcount, ycount, impluse_q):
    
    from skimage.restoration import unwrap_phase
  
    x_range = np.arange(ycount)
    y_range = np.arange(xcount)

    points = np.meshgrid(x_range + 0.5, y_range - 0.5, indexing = 'xy')
    points = (
        np.reshape(points[0], (1, int(xcount * ycount))), 
        np.reshape(points[1], (1, int(xcount * ycount)))
        )
    
    func_abs = interpolate.RegularGridInterpolator(
        (x_range, y_range), np.abs(impluse_q), method = "linear", 
        bounds_error = False, fill_value = 0
        )
    func_ang = interpolate.RegularGridInterpolator(
        (x_range, y_range), unwrap_phase(np.angle(impluse_q)), 
        method = "linear", bounds_error = False, fill_value = 0
        )
    
    shifted_impulse_q = func_abs(points) * np.exp(1j * func_ang(points))
    shifted_impulse_q = np.rot90(np.reshape(
        shifted_impulse_q, (xcount, ycount)
        ))
    
    return shifted_impulse_q 
    
def _fresnel_dfft(
        xstart, xend, xcount, ystart, yend, ycount, 
        wavelength, distance, wavefront
        ):
        
    # the grid of the frequency
    
    qx_tick = np.linspace(0.25/xstart, 0.25/xend, xcount) * xcount
    qy_tick = np.linspace(0.25/ystart, 0.25/xend, ycount) * ycount
    qx_grid, qy_grid = np.meshgrid(qx_tick, qy_tick)

    # propagation function
    
    impulse_q = np.exp(
        (-1j * 2*np.pi / wavelength * distance) * 
        (1 - wavelength**2 * (qx_grid**2 + qy_grid**2) / 2)
        )
    impulse_q = _shift_impluse_geometry(xcount, ycount, impulse_q)
    propagated_wavefront = fft.ifft2(
        fft.fft2(wavefront) * fft.ifftshift(impulse_q)
        )
    
    return propagated_wavefront
    
def vibration_shift(data, k_vector, grid, tick, screen, vib_angle, offset):
    
    # phase shift
    
    x_tick, y_tick = tick
    rot_x_phase = np.exp(
        -1j * k_vector * (vib_angle[0] * (grid[0] + offset[0]))
        )
    rot_y_phase = np.exp(
        -1j * k_vector * (vib_angle[1] * (grid[1] + offset[1]))
        )
    
    # calculate range

    x_00, x_01, x_10, x_11 = _support._shift_plane(
        offset[0], x_tick, 
        screen['xstart'], screen['xfin'], screen['nx']
        )
    y_00, y_01, y_10, y_11 = _support._shift_plane(
        offset[1], y_tick, 
        screen['ystart'], screen['yfin'], screen['ny']
        )

    vibration_wfr = np.zeros((int(screen['nx']), int(screen['ny'])), dtype = np.complex64)
    vibration_wfr[y_10 : y_11, x_10 : x_11] = (
        (data * rot_x_phase * rot_y_phase)[y_00 : y_01, x_00 : x_01]
        )
    
    return vibration_wfr
            
def _cal_energy_wfr(undulator, electron_beam, screen, energy_relative):
    
    """
    ---------------------------------------------------------------------------
    calculate wavefronts from electron source to screens.
    
    args: undulator     - the parameters of undualtor.
          electron_beam - the parameters of electron_beam.
          screen        - the parameters of screen.
          mc_para       - the parameters of monte carlo.
    
    return: the single wavefront.
    ---------------------------------------------------------------------------
    """
    
    #-----------------------------------------------------------
    # reference wavefront calcualtion
    
    # The construction of undulator
    
    und = _undulator(undulator)
    und.wave_length()
    und.cal_k(electron_beam_energy = electron_beam["energy"])
    
    und_magnetic_structure = und.magnetic_structure()
    
    # The construction of electron beam
    
    e_beam = _srw_electron_beam(
        electron_beam, und.n_period, und.period_length, 
        und.n_hormonic
        )
    e_beam.monte_carlo()
    
    # calcualte the wavefront
    
    _part_beam, wavelength, resonance_energy = e_beam.after_monte_carlo(
        [0, 0, 0, 0, energy_relative], 
        und.period_length, und.k_vertical, und.k_horizontal
        )
    
    # print(resonance_energy, flush = True)
    
    wavefront = _propagate_wave_front(screen, resonance_energy)
    wavefront._cal_wave_front(_part_beam, und_magnetic_structure) 
    
    # only s polarization is considered
    wavefront = np.reshape(
        wavefront.wfr.arEx, (screen['nx'], screen['ny'], 2)
        )
    wavefront = wavefront[:, :, 0] - 1j * wavefront[: ,:, 1]
    energy = 12.398  / (wavelength * 1e10)

    #-----------------------------------------------------------
    # proapgate to source
    
    source_wfr = _fresnel_dfft(
        screen['xstart'], screen['xfin'], screen['nx'], 
        screen['ystart'], screen['yfin'], screen['ny'], 
        wavelength, -screen['screen'] + e_beam.initial_z/2, wavefront
        )[int(screen['nx']/4) : -int(screen['nx']/4),
          int(screen['ny']/4) : -int(screen['ny']/4),
          ]
    
    # source_wfr = wavefront
    
    return energy, np.array(source_wfr, dtype = np.complex64)
    
    
def _cal_reference_wfr(
        undulator, electron_beam, screen, sampling = 100, 
        sigma_ratio = 3
        ):
    
    """
    ---------------------------------------------------------------------------
    calculate wavefronts from electron source to screens.
    
    args: undulator     - the parameters of undualtor.
          electron_beam - the parameters of electron_beam.
          screen        - the parameters of screen.
          mc_para       - the parameters of monte carlo.
    
    return: the single wavefront.
    ---------------------------------------------------------------------------
    """

    # ±3 responding to 99.73% area of gaussian
    energy_relative = np.linspace(-sigma_ratio, sigma_ratio, int(sampling))
    
    reference_wfrs = list()
    reference_energy = list()

    start_time = time.time()
    for idx in range(int(sampling)):
        
        interval_time = time.time()
        processing_bar(idx, int(sampling), interval_time, start_time)
        
        references = _cal_energy_wfr(
            undulator, electron_beam, screen, energy_relative[idx]
            )
        reference_energy.append(references[0])
        reference_wfrs.append(references[1])
    
    return reference_wfrs, reference_energy
    
def mc_brightness_convolution(
        undulator, electron_beam, screen, n_step = 10, 
        n_energy_sampling = 100, step_n_vector = 500, sigma_ratio = 3, 
        file_name = None
        ):
     
    print("reference wavefront calculation start..... ", flush = True)
    print(time.asctime(time.localtime(time.time())), flush = True)
    
    newSeed = random.randint(0, 1000000)
    random.seed(newSeed)
    
    n_electron = electron_beam['n_electron']
    n_cmode = screen['n_vector'] 
    probability = (
        1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * np.linspace(
            -sigma_ratio, sigma_ratio, int(n_energy_sampling)
            )**2)
        )
    probability /= np.sum(probability)
    probability = np.array(probability * n_electron, dtype = int)
    probability[int(n_energy_sampling/2)]  +=  int(n_electron - sum(probability))
    
    xtick = np.linspace(screen['xstart'], screen['xfin'], screen['nx'])
    ytick = np.linspace(screen['ystart'], screen['yfin'], screen['ny'])
    gridx, gridy = np.meshgrid(xtick, ytick)
    
    # The construction of electron beam
    
    und = _undulator(undulator)
    und.wave_length()
    und.cal_k(electron_beam_energy = electron_beam["energy"])
    
    e_beam = _srw_electron_beam(
        electron_beam, und.n_period, und.period_length, 
        undulator['n_hormonic']
        )
    screen['screen'] = screen['screen'] - e_beam.initial_z/2
    e_beam.monte_carlo()
    
    reference_wfrs, reference_energy = _cal_reference_wfr(
        undulator, electron_beam, screen, sampling = n_energy_sampling,
        sigma_ratio = sigma_ratio
        )
    screen['nx'] /= 2
    screen['ny'] /= 2
    
    #------------------------------------------------------
    # for wavefrotn vibration
    
    temp_wfrs = np.zeros(
        (int(n_electron/n_step), int(screen['nx'] * screen['ny'])),
        dtype = np.complex64
        )
    temp_cmodes = np.zeros(
        (int(n_step*n_cmode), int(screen['nx'] * screen['ny'])),
        dtype = np.complex64
        )
    
    print('\n', flush = True)
    print("wavefront calculation start.....", flush = True)
    print(time.asctime(time.localtime(time.time())), flush = True)
    print('\n', flush = True)
    
    probability_count = 0
    energy_idx = 0
    
    #--------------------------------------------------------------------------
    for i_step in range(int(n_step)):
        
        print("STEP%d calculation start..... " % (i_step), flush = True)
        print(time.asctime(time.localtime(time.time())), flush = True)
        
        for idx in range(int(n_electron/n_step)):
            
            mc_rand_array = [random.gauss(0, 1) for ir in range(5)]
            mc_rand_array[4] = np.linspace(
                -sigma_ratio, sigma_ratio, int(n_energy_sampling)
                )[energy_idx]
            
            e_beam.monte_carlo()
            part_beam, wavelength, resonance_energy = e_beam.after_monte_carlo(
                mc_rand_array, und.period_length, und.k_vertical, 
                und.k_horizontal
                )
            k_vector = 2 * np.pi / wavelength
            reference_wfr_idx = reference_wfrs[energy_idx]
            
            vibration_wfr = vibration_shift(
                reference_wfr_idx, k_vector, [gridx, gridy], [xtick, ytick], screen,
                [part_beam.partStatMom1.xp, part_beam.partStatMom1.yp], 
                [part_beam.partStatMom1.x, part_beam.partStatMom1.y], 
                )
                    
            temp_wfrs[idx, :] = np.reshape(
                vibration_wfr, (int(screen['nx'] * screen['ny']), )
                ) 
            
            probability_count += 1
            if probability_count > int(probability[energy_idx]):
                energy_idx += 1
                print("energy: %03f keV" % reference_energy[energy_idx], flush = True)
                probability_count = 0
        
        print("STEP%d SVD calculation start..... " % (i_step), flush = True)
        print(time.asctime(time.localtime(time.time())), flush = True)
        
        vector, value, evolution = ssl.svds(
            temp_wfrs.T, k = int(step_n_vector + 10), 
            # tol = 1e-5, solver = 'propack'
            )
        temp_vector = (
            np.copy(vector[:, ::-1], order = 'C') * 
            np.copy(np.abs(value[::-1]), order = 'C')
            )
        
        temp_cmodes[int(n_cmode * i_step) : int(n_cmode * (i_step + 1)), :] = (
            temp_vector[:, 0 : int(n_cmode)].T
            )
        temp_wfrs = np.zeros(
            (int(n_electron/n_step), int(screen['nx'] * screen['ny'])),
            dtype = np.complex64
            )
        
        print("STEP%d SVD calculation finished" % (i_step), flush = True)
        print(time.asctime(time.localtime(time.time())), flush = True)
        print('\n', flush = True)
    
    #--------------------------------------------------------------------------
    
    #------------------------------------------------------
    # coherent mode calculation
    
    print("Final SVD calculation start..... ", flush = True)
    print(time.asctime(time.localtime(time.time())), flush = True)
    
    vector, value, evolution = ssl.svds(
        temp_cmodes.T, k = int(n_cmode + 10), 
        # tol = 1e-5, solver = 'propack'
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
    # cmode = np.real(cmode) - 1j * np.imag(cmode)
    
    with h5.File(file_name, "a") as f:
        
        coherence_group = f.create_group("coherence")
        coherence_group.create_dataset(
            "eig_value", data = value[0 : int(n_cmode)]
            )
        coherence_group.create_dataset("eig_vector", data = cmode) 
        
    print("Finished", flush = True)
    print(time.asctime(time.localtime(time.time())), flush = True)
        
#-----------------------------------------------------------------------------#
# classes

#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    pass