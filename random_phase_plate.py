# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Thu Oct  9 19:44:13 2025"
__email__    = "xuhan@ihep.ac.cn"


"""
Description
"""

#-----------------------------------------------------------------------------#
# modules

from scipy.linalg import svd
from cat.wave_optics.optics import source_optic, screen
from cat.wave_optics.propagate import fresnel

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(r'D:\File\Paper\CMDC\simulation\cat')
import os

#-----------------------------------------------------------------------------#
# parameters

file_header = r"D:\File\Paper\CMDC\simulation"
source_file_name = os.path.join(
    file_header, r"Gaussian_source_8keV_testCRL.h5"
    )

unit = '$\u03bc$m' 

#-----------------------------------------------------------------------------#
# functions

def generate_random_phase_plate(
        size = 256, phase_magnitude = 2 * np.pi, variation_frequency = 10.0
        ):
    
    from scipy.ndimage import gaussian_filter
    
    raw_noise = np.random.randn(size, size)
    sigma = max(1.0, size / (5.0 * variation_frequency)) 
    smoothed_noise = gaussian_filter(raw_noise, sigma = sigma)
    
    noise_min = smoothed_noise.min()
    noise_max = smoothed_noise.max()
    normalized_phase = (smoothed_noise - noise_min) / (noise_max - noise_min)
    phase_plate = normalized_phase * phase_magnitude
    
    return phase_plate

#--------------------------------------------------------------------
# the decoupling function

def decouple_csd(
        eigen_values, eigen_vectors, xcount, ycount, counts_mode = int(1000)
        ):
    
    mode_analysis = list()
    decoupled_modes = list()
    decoupled_modes_x = list()
    decoupled_modes_y = list()
    coupled_modes = list()
    
    n_modes_total = len(eigen_values)
    counts_mode = min(counts_mode, n_modes_total)
    
    eigen_values_norm = eigen_values / np.sum(eigen_values)
    total_energy = np.sum(np.array(eigen_values_norm[:counts_mode]))
    
    #-----------------------------------------------------------
    # loop over modes
    
    for idx in range(counts_mode):
        
        coherent_mode = eigen_vectors[idx]
        energy_weight = eigen_values_norm[idx]
       
        u_vector, singular, vh_vector = svd(coherent_mode, full_matrices = False)
        
        separable_part = singular[0] * np.outer(u_vector[:, 0], vh_vector[0, :])
        separable_x = np.sqrt(singular[0]) * u_vector[:, 0]
        separable_y = np.sqrt(singular[0]) * vh_vector[0, :]
        residual_part = coherent_mode - separable_part
       
        mode_energy = energy_weight  
        separable_energy = energy_weight * (singular[0]**2)  
        coupled_energy = mode_energy - separable_energy  
        
        if mode_energy > 0:
            separable_ratio = separable_energy / mode_energy 
            coupled_ratio = coupled_energy / mode_energy 
        else:
            separable_ratio = 0
            coupled_ratio = 0
       
        mode_analysis.append({
            'mode_index': idx,
            'mode_energy': mode_energy,
            'separable_energy': separable_energy,
            'coupled_energy': coupled_energy,
            'separable_ratio': separable_ratio,
            'coupled_ratio': coupled_ratio,
            'singular_values': singular  
            })
       
        decoupled_modes.append(energy_weight**0.5 * separable_part)
        decoupled_modes_x.append(energy_weight**0.25 * separable_x)
        decoupled_modes_y.append(energy_weight**0.25 * separable_y)
        coupled_modes.append(energy_weight**0.5 * residual_part)
       
    total_separable_energy = sum([k['separable_energy'] for k in mode_analysis])
    total_coupled_energy = sum([k['coupled_energy'] for k in mode_analysis])
    
    if total_energy > 0:
        overall_separable_ratio = total_separable_energy / total_energy 
        overall_coupled_ratio = total_coupled_energy / total_energy 
    else:
        overall_separable_ratio = 0
        overall_coupled_ratio = 0
    
    #-----------------------------------------------------------
    # re-compress coupled parts
    
    coupled_matrix = np.array([c.flatten() for c in coupled_modes])
    u_vector, singular, vh_vector = svd(coupled_matrix, full_matrices = False)
    
    print("-------------------------------------------")
    print(f"Separate Ratio: {overall_separable_ratio:.4f}", flush=True)
    
    threshold_n = list()
    
    for threshold in [0.90, 0.95, 0.99]:
        
        energy_threshold = (threshold - overall_separable_ratio) / overall_coupled_ratio 
        norm_singular = singular / np.sum(np.abs(singular))
        cumulative_energy = (np.cumsum(norm_singular**2) / np.sum(norm_singular**2))
        n_modes = np.argmax(cumulative_energy >= energy_threshold) + 1
        threshold_n.append(n_modes)
        print("compressed modes number: %d" % n_modes, flush = True)
    
    n_modes = 45
    coupled_modes_compressed = list()
    for idx in range(n_modes):
        
        mode_flat = vh_vector[idx, :]
        mode_2d = mode_flat.reshape(xcount, ycount)
        coupled_modes_compressed.append(mode_2d)
        
    result = {
        'mode_analysis': mode_analysis,
        'overall_analysis': {
            'total_energy': total_energy,
            'total_separable_energy': total_separable_energy,
            'total_coupled_energy': total_coupled_energy,
            'separable_ratio': overall_separable_ratio,
            'coupled_ratio': overall_coupled_ratio
        },
        'decoupled_modes': decoupled_modes,
        'decoupled_modes_x': decoupled_modes_x,
        'decoupled_modes_y': decoupled_modes_y,
        'coupled_modes_compressed': coupled_modes_compressed,
        'coupled_singular_values': singular,
        'coupled_cumulative_energy': cumulative_energy
        }
    
    decoupling_result = {
        'separable_ratio': overall_separable_ratio,
        'threshold_n': threshold_n,
        'n_modes': counts_mode,
        'ratios': np.ones(counts_mode),
        'modes_x': decoupled_modes_x,
        'modes_y': decoupled_modes_y,
        'spatial_dims': (xcount, ycount)
        }
    
    # singular values (weights) are multiplied into the modes here!!!!!
    coupling_result = {
        'n_modes': n_modes,
        'ratios': np.ones(n_modes),
        'modes': [
            singular[idx] * coupled_modes_compressed[idx] 
            for idx in range(n_modes)],
        'u_vector': u_vector,
        'spatial_dims': (xcount, ycount)
        }
        
    return decoupling_result, coupling_result, result

def reconstruct_intensity(
    modes_x, modes_y, residual_modes, u_vector, n_res = None, 
    check_shapes = True, return_parts = False
    ):
    
    n_modes = len(modes_x)
    k_res = min(n_res, len(residual_modes))        
    u_used = u_vector[:, :k_res]
    nx, ny = len(modes_x[0]), len(modes_y[0])

    coherent_modes = list()
    sep_fields = list()
    res_fields = list()
    intensity = np.zeros((nx, ny), dtype=np.float64)

    for i in range(n_modes):
        
        sep_field = np.outer(modes_x[i], modes_y[i])
        res_field = np.zeros((nx, ny), dtype = np.complex128)
        
        for j in range(k_res):
            res_field += u_used[i, j] * residual_modes[j]

        coherent_field = sep_field + res_field

        coherent_modes.append(coherent_field)
        sep_fields.append(sep_field)
        res_fields.append(res_field)
        intensity += np.abs(coherent_field) ** 2

    if return_parts:
        return coherent_modes, intensity, sep_fields, res_fields
    
    return coherent_modes, intensity

#-------------------------------------------------------------------
# optics system


#-----------------------------------------------------------------------------#
# classes

#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    print('----------------------------------------', flush = True)
    print("loading source", flush = True)
    
    
    count_mode = 50
    
    optic_source = source_optic(
        source_file_name = source_file_name, n_vector = int(count_mode), 
        position = 20
        )  
    source_screen = screen(
        optic = optic_source, n_vector = int(count_mode), 
        position = 60
        )
    fresnel(optic_source, source_screen)
    source_screen.cal_intensity()
    
    #----------------------------------------
    # addd a phase plate and CMDC
    
    phase_plate_screen = screen(
        optic = source_screen, 
        n_vector = int(count_mode), 
        position = 60
        )
    fresnel(source_screen, phase_plate_screen)
        
    #----------------------------------------
    # 2D propagation
    
    print("accepted screen calculation", flush = True)
    
    
    decoupling_result_list = list()
    intensity_2d = list()
    intensity_reconstructed = list()
    
    i = 11
    phase_plate = generate_random_phase_plate(
        size = phase_plate_screen.xcount, 
        phase_magnitude = (np.arange(20)[i] + 1) * 5 * np.pi / 20, 
        variation_frequency = 50
        )
    for idx in range(count_mode):
        phase_plate_screen.cmode[idx] *= np.exp(1j * phase_plate)
        
    observe_screen = screen(
        optic = phase_plate_screen, n_vector = int(count_mode), 
        position = 61
        )
    fresnel(phase_plate_screen, observe_screen)
    
    #---------------------------------------------------------------
    
    phase_plate_screen0 = screen(
        optic = source_screen, 
        n_vector = int(count_mode), 
        position = 60
        )
    fresnel(source_screen, phase_plate_screen0)
    observe_screen0 = screen(
        optic = phase_plate_screen0, n_vector = int(count_mode), 
        position = 61
        )
    fresnel(phase_plate_screen0, observe_screen0)
    observe_screen0.cal_intensity()
    intensity0 = observe_screen0.intensity
    
    #---------------------------------------------------------------
    
    observe_screen.decomposition()
    observe_screen.n = 48
    count_mode = 48
    
    decoupling_result, coupling_result, result = decouple_csd(
        np.abs(observe_screen.ratio[:count_mode])**2, 
        observe_screen.cmode, 
        observe_screen.xcount, observe_screen.ycount, 
        counts_mode = int(1000),
        )
    
    observe_screen.cal_intensity()
    intensity_reconstructed = list()
    r_square_list = list()
    mae = list()

    modes, intensity_2d = reconstruct_intensity(
        decoupling_result['modes_x'], decoupling_result['modes_y'], 
        coupling_result['modes'], coupling_result['u_vector'],
        n_res = 45, check_shapes = True, return_parts = False
        )
    
    for idx in range(40):
        
        modes, intensity = reconstruct_intensity(
            decoupling_result['modes_x'], decoupling_result['modes_y'], 
            coupling_result['modes'], coupling_result['u_vector'],
            n_res = idx, check_shapes = True, return_parts = False
            )
        error = np.abs(intensity_2d - intensity) 
        r_square = 1 - np.sum(np.abs(error)**2) / np.sum(np.abs(intensity_2d)**2)
        intensity_reconstructed.append(intensity)
        r_square_list.append(r_square)
        mae.append(np.max(np.abs(intensity_reconstructed[idx] - intensity_2d)))
        
    #----------------------------------------
    # vis
    
    vis = False
    
    if vis:
        
        figure, axes = plt.subplots(4, 4, figsize = (12, 10))
    
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        src_extent_list = [
            source_screen.xstart * 1e3, source_screen.xend * 1e3, 
            source_screen.ystart * 1e3, source_screen.yend * 1e3, 
            ]
        
        image = axes[0, 1].imshow(
            intensity_reconstructed[1] / np.max(intensity_2d), 
            extent = src_extent_list, origin = 'upper', cmap = 'coolwarm',
            )
        axes[0, 1].set_xlabel("x (mm)", fontsize = 12)
        axes[0, 1].set_ylabel("y (mm)", fontsize = 12)
        axes[0, 1].set_xlim([-0.5, 0.5])
        axes[0, 1].set_ylim([-0.5, 0.5])
        axes[0, 1].yaxis.set_label_coords(-0.20, 0.5)
        axes[0, 1].set_title("1D + 1 2D CM", fontsize = 12)
        divider = make_axes_locatable(axes[0, 1])
        cax = divider.append_axes("right", size = "5%", pad = 0.05) 
        cbar = figure.colorbar(image, cax = cax)
        axes[0, 1].set_box_aspect(1) 
        
        image = axes[1, 1].imshow(
            np.abs(intensity_reconstructed[1] - intensity_2d) / np.max(intensity_2d), 
            vmin = 0, vmax = 1, extent = src_extent_list, 
            origin = 'upper', cmap = 'coolwarm',
            )
        axes[1, 1].set_xlabel("x (mm)", fontsize = 12)
        axes[1, 1].set_ylabel("y (mm)", fontsize = 12)
        axes[1, 1].set_xlim([-0.5, 0.5])
        axes[1, 1].set_ylim([-0.5, 0.5])
        axes[1, 1].yaxis.set_label_coords(-0.20, 0.5)
        axes[1, 1].set_title("Difference", fontsize = 12)
        divider = make_axes_locatable(axes[1, 1])
        cax = divider.append_axes("right", size = "5%", pad = 0.05) 
        cbar = figure.colorbar(image, cax = cax)
        axes[1, 1].set_box_aspect(1) 
        
        image = axes[0, 2].imshow(
            intensity_reconstructed[4] / np.max(intensity_2d), 
            vmin = 0, vmax = 1, extent = src_extent_list, 
            origin = 'upper', cmap = 'coolwarm',
            )
        axes[0, 2].set_xlabel("x (mm)", fontsize = 12)
        axes[0, 2].set_ylabel("y (mm)", fontsize = 12)
        axes[0, 2].set_xlim([-0.5, 0.5])
        axes[0, 2].set_ylim([-0.5, 0.5])
        axes[0, 2].yaxis.set_label_coords(-0.20, 0.5)
        axes[0, 2].set_title("1D + 4 2D CMs", fontsize = 12)
        divider = make_axes_locatable(axes[0, 2])
        cax = divider.append_axes("right", size = "5%", pad = 0.05) 
        cbar = figure.colorbar(image, cax = cax)
        axes[0, 2].set_box_aspect(1) 
        
        image = axes[1, 2].imshow(
            np.abs(intensity_reconstructed[4] - intensity_2d) / np.max(intensity_2d), 
            vmin = 0, vmax = 1, extent = src_extent_list, 
            origin = 'upper', cmap = 'coolwarm',
            )
        axes[1, 2].set_xlabel("x (mm)", fontsize = 12)
        axes[1, 2].set_ylabel("y (mm)", fontsize = 12)
        axes[1, 2].set_xlim([-0.5, 0.5])
        axes[1, 2].set_ylim([-0.5, 0.5])
        axes[1, 2].yaxis.set_label_coords(-0.20, 0.5)
        axes[1, 2].set_title("Difference", fontsize = 12)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size = "5%", pad = 0.05) 
        cbar = figure.colorbar(image, cax = cax)
        axes[1, 2].set_box_aspect(1) 
        
        image = axes[0, 3].imshow(
            intensity_reconstructed[13] / np.max(intensity_2d), 
            vmin = 0, vmax = 1, extent = src_extent_list, 
            origin = 'upper', cmap = 'coolwarm',
            )
        axes[0, 3].set_xlabel("x (mm)", fontsize = 12)
        axes[0, 3].set_ylabel("y (mm)", fontsize = 12)
        axes[0, 3].set_xlim([-0.5, 0.5])
        axes[0, 3].set_ylim([-0.5, 0.5])
        axes[0, 3].yaxis.set_label_coords(-0.20, 0.5)
        axes[0, 3].set_title("1D + 12 2D CMs", fontsize = 12)
        divider = make_axes_locatable(axes[0, 3])
        cax = divider.append_axes("right", size = "5%", pad = 0.05) 
        cbar = figure.colorbar(image, cax = cax)
        axes[0, 3].set_box_aspect(1) 
        
        image = axes[1, 3].imshow(
            np.abs(intensity_reconstructed[13] - intensity_2d) / np.max(intensity_2d), 
            vmin = 0, vmax = 1, extent = src_extent_list, 
            origin = 'upper', cmap = 'coolwarm',
            )
        axes[1, 3].set_xlabel("x (mm)", fontsize = 12)
        axes[1, 3].set_ylabel("y (mm)", fontsize = 12)
        axes[1, 3].set_xlim([-0.5, 0.5])
        axes[1, 3].set_ylim([-0.5, 0.5])
        axes[1, 3].yaxis.set_label_coords(-0.20, 0.5)
        axes[1, 3].set_title("Difference", fontsize = 12)
        divider = make_axes_locatable(axes[1, 3])
        cax = divider.append_axes("right", size = "5%", pad = 0.05) 
        cbar = figure.colorbar(image, cax = cax)
        axes[1, 3].set_box_aspect(1) 
        
        image = axes[0, 0].imshow(
            intensity_2d / np.max(intensity_2d), 
            vmin = 0, vmax = 1, extent = src_extent_list, 
            origin = 'upper', cmap = 'coolwarm',
            )
        axes[0, 0].set_xlabel("x (mm)", fontsize = 12)
        axes[0, 0].set_ylabel("y (mm)", fontsize = 12)
        axes[0, 0].set_xlim([-0.5, 0.5])
        axes[0, 0].set_ylim([-0.5, 0.5])
        axes[0, 0].yaxis.set_label_coords(-0.25, 0.5)
        axes[0, 0].set_title("2D", fontsize = 12)
        divider = make_axes_locatable(axes[0, 0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05) 
        cbar = figure.colorbar(image, cax = cax)
        axes[0, 0].set_box_aspect(1) 
        
        x_data = result['coupled_cumulative_energy'][:40]
        y_mae = np.array(mae) / np.max(np.max(intensity_2d))
        ax1 = axes[1, 0] 
        
        color1 = 'tab:blue'
        ax1.scatter(x_data, r_square_list, s=40, alpha=0.5, color=color1)
        ax1.plot(x_data, r_square_list, linewidth=3, alpha=0.5, color=color1)
        
        ax1.set_xlabel("cumulative energy", fontsize=12)
        ax1.set_ylabel("R-square", fontsize=12, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()  
        color2 = 'tab:red'
        
        ax2.scatter(x_data, y_mae, s=40, alpha=0.5, color=color2)
        ax2.plot(x_data, y_mae, linewidth=3, alpha=0.5, color=color2)
        
        ax2.set_ylabel("MAE", fontsize=12, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2) 
        ax1.yaxis.set_label_coords(-0.25, 0.5)
        ax1.set_box_aspect(1) 
        ax2.set_box_aspect(1) 

        figure.tight_layout()
