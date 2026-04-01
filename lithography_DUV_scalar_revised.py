# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu"
__date__     = "Date : Mon Mar 23 22:14:44 2026"
__email__    = "xuhan@ihep.ac.cn"


"""
Description: an example of lithography simulation (DUV, scalar)
"""

#-----------------------------------------------------------------------------#
# modules

from computional_lithography_lib_revised import aerial_image_2d_fullstack, aerial_image_cmdc_fullstack
from computional_lithography_lib_revised import calculate_convolution_socs_1d, calculate_convolution_socs_2d
from computional_lithography_lib_revised import calculate_convolution_resudual
from computional_lithography_lib_revised import lithography_simulation
from scipy.linalg import svd

import numpy as np
import matplotlib.pyplot as plt

import argparse
import time

#-----------------------------------------------------------------------------#
# parameters

_DEFAULT_WAVELENGTH = 193.0
unit = '$\u03bc$m' 

parser = argparse.ArgumentParser(
    description = "Python Lithography Simulation (TCC-SOCS)"
    )
parser.add_argument('--Lx', type = int, default = 65536)
parser.add_argument('--Ly', type = int, default = 65536)
parser.add_argument('--maskSizeX', type = int, default = 1024)
parser.add_argument('--maskSizeY', type = int, default = 1024)
parser.add_argument('--spaceWidth', type = int, default = 290)
parser.add_argument('--srcSize', type = int, default = 81)

parser.add_argument('--NA', type = float, default = 0.30)
parser.add_argument('--defocus', type = float, default = 4000.0)
parser.add_argument('--nk', type = int, default = 61) 
parser.add_argument('--nk_coupled', type = int, default = 5)

args = parser.parse_args()
    
#-----------------------------------------------------------------------------#
# functions

#--------------------------------------------------
# CMDC function

def decouple_csd(
        eigen_values, eigen_vectors, xcount, ycount, counts_mode = int(1000)
        ):
    
    decoupled_modes = list()
    coupled_modes = list()
    decoupled_modes_x = list()
    decoupled_modes_y = list()
    mode_analysis = list()
    
    n_modes_total = len(eigen_values)
    counts_mode = min(counts_mode, n_modes_total)
    
    eigen_values_norm = eigen_values / np.sum(eigen_values)
    total_energy = np.sum(np.array(eigen_values_norm[:counts_mode]))
    
    #-----------------------------------------------------------
    # loop over modes
    
    for idx in range(counts_mode):
        
        coherent_mode = eigen_vectors[:, idx].reshape((xcount, ycount))  
        energy_weight = eigen_values_norm[idx]
       
        u_vector, singular, vh_vector = svd(coherent_mode, full_matrices = False)
        
        separable_part = singular[0] * np.outer(u_vector[:, 0], vh_vector[0, :])
        separable_x = np.sqrt(singular[0]) * u_vector[:, 0]
        separable_y = np.sqrt(singular[0]) * vh_vector[0, :]
        residual_part = coherent_mode - separable_part
       
        mode_energy = energy_weight  
        separable_energy = energy_weight * (singular[0]**2)  
        coupled_energy = mode_energy - separable_energy  
        
        separable_ratio = separable_energy / mode_energy 
        coupled_ratio = coupled_energy / mode_energy 
       
        mode_analysis.append({
            'mode_index': idx, 'mode_energy': mode_energy,
            'separable_energy': separable_energy, 'coupled_energy': coupled_energy,
            'separable_ratio': separable_ratio, 'coupled_ratio': coupled_ratio,
            'singular_values': singular  
            })
       
        decoupled_modes.append(energy_weight**0.5 * separable_part)
        decoupled_modes_x.append(energy_weight**0.25 * separable_x)
        decoupled_modes_y.append(energy_weight**0.25 * separable_y)
        coupled_modes.append(energy_weight**0.5 * residual_part)
       
    total_separable_energy = sum([k['separable_energy'] for k in mode_analysis])
    total_coupled_energy = sum([k['coupled_energy'] for k in mode_analysis])
    
    overall_separable_ratio = total_separable_energy / total_energy 
    overall_coupled_ratio = total_coupled_energy / total_energy 
    print(f"Separate Ratio: {overall_separable_ratio:.4f}", flush = True)
    
    #-----------------------------------------------------------
    # re-compress coupled parts
    
    coupled_matrix = np.array([c.flatten() for c in coupled_modes])
    u_vector, singular, vh_vector = svd(coupled_matrix, full_matrices = False)
        
    coupled_modes_compressed = list()
    
    for idx in range(counts_mode):

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
        'coupled_singular_values': singular
        }
    
    decoupling_result = {
        'n_modes': counts_mode,
        'ratios': np.ones(counts_mode),
        'modes_x': decoupled_modes_x,
        'modes_y': decoupled_modes_y,
        'spatial_dims': (xcount, ycount)
        }
    
    # singular values (weights) are multiplied into the modes here!!!!!
    coupling_result = {
        'u_vector': u_vector, 'n_modes': counts_mode, 'ratios': np.ones(counts_mode),
        'modes': [singular[idx] * coupled_modes_compressed[idx] for idx in range(counts_mode)],
        'spatial_dims': (xcount, ycount)
        }
        
    return decoupling_result, coupling_result, result

def decouple_mask(data_2d, xcount, ycount, threshold = 0.999):
    
    decoupled_x, decoupled_y = list(), list()
    u_vector, singular, vh_vector = svd(data_2d, full_matrices = False)
    
    norm_singular = singular / np.sum(np.abs(singular))
    cumulative_energy = (
        np.cumsum(norm_singular**2) / 
        np.sum(norm_singular**2)
        )
    n_modes = np.argmax(cumulative_energy >= threshold) + 1
    
    for idx in range(n_modes):
        decoupled_x.append(np.sqrt(singular[idx]) * u_vector[:, idx])
        decoupled_y.append(np.sqrt(singular[idx]) * vh_vector[idx, :])

    return decoupled_x, decoupled_y, singular

#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    #------------------------------------------------------
    # Setup Directories
    
    print("Lithography Simulation start.....", flush = True)
    print(time.asctime(time.localtime(time.time())), flush = True)
    
    simulation_instance = lithography_simulation()
    
    #------------------------------------------------------
    # load source and mask
    
    print("Loading source and mask...", flush = True)
    source_pupil = np.load("source_annular.npy")
    mask_layout = np.load("mask.npy")
    
    #------------------------------------------------------
    # SOCS and CMDC Decomposition
    
    print("-----------------------------------------------------", flush = True)
    print("SOCS Kernels decomposition...", flush = True)
    
    sigma_outer = simulation_instance._get_sigma_outer(source_pupil)
    kernels = simulation_instance.calculate_socs_implicit(
        source_pupil, args.NA, args.defocus, args.Lx, args.Ly, 
        sigma_outer, args.nk
        )
    eigen_vals, eigen_vecs, freq_cutoff_x, freq_cutoff_y = kernels
    
    decoupling_vecs, coupling_vecs, result = decouple_csd(
        eigen_vals, eigen_vecs, 
        int(eigen_vecs.shape[0]**0.5), int(eigen_vecs.shape[0]**0.5),
        counts_mode = args.nk
        )
    
    #------------------------------------------------------
    # kernel-mask convolution
    
    print("-----------------------------------------------------", flush = True)
    print("kernel-mask convolution...", flush = True)
    
    coherent_convolution_2d = calculate_convolution_socs_2d(
        mask_layout, eigen_vals[0 : int(args.nk)], 
        eigen_vecs[:, 0 : int(args.nk)], 
        freq_cutoff_x, freq_cutoff_y, output_kernels = True
        )
    
    decoupled_x, decoupled_y, singular = decouple_mask(
        mask_layout, args.maskSizeX, args.maskSizeY
        )
    coherent_convolution_cmdc = calculate_convolution_socs_1d(
        [decoupled_x, decoupled_y], 
        [decoupling_vecs['modes_x'], decoupling_vecs['modes_y']],
        args.nk, freq_cutoff_x, freq_cutoff_y, 
        output_kernels = False
        )

    if args.nk_coupled != 0:
        correction_convolution = calculate_convolution_resudual(
            mask_layout, coupling_vecs, args.nk_coupled,
            freq_cutoff_x, freq_cutoff_y
            )
    else:
        correction_convolution = None
    
    #------------------------------------------------------
    # areial image construction convolution
    
    print("-----------------------------------------------------", flush = True)
    print("areial image construction...", flush = True)

    areial_image_2d = aerial_image_2d_fullstack(coherent_convolution_2d)
    areial_image_cmdc = aerial_image_cmdc_fullstack(
        coherent_convolution_cmdc,
        residual_convolution = correction_convolution,
        u_vector = coupling_vecs['u_vector'] if correction_convolution is not None else None,
        n_res = args.nk_coupled
        )
    
    areial_image_cmdc /= np.max(areial_image_cmdc)
    areial_image_2d /= np.max(areial_image_2d)
    
    error = np.abs(areial_image_cmdc - areial_image_2d) 
    r_square = (
        1 - np.sum(np.abs(error)**2) / 
        np.sum(np.abs(areial_image_2d)**2)
        )
    mae = np.max(error)
    
    print("R-square:   %.4f" %(r_square))
    print("MAE:   %.4f" %(mae))
    print("simulation completed.", flush = True)
    
    #-----------------------------------------------------
    # visualization
    
    visulation = True
    
    kernel_list_intensity = list()
    kernel_n = int(2 * freq_cutoff_x + 1)  
    
    for idx in range(int(args.nk)):
        kernel_idx = eigen_vecs[:, idx].reshape(kernel_n, kernel_n)
        kernel_list_intensity.append(
            np.abs(kernel_idx)**2 / np.max(np.abs(kernel_idx)**2)
            )
    
    if visulation:
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        #-----------------------------------------------------------
        # litho parameters
        
        figure, axes = plt.subplots(4, 4, figsize = (12, 10))
        
        pixel = int(args.Lx / args.maskSizeX)
        mask_extent_list = [
            -args.maskSizeX/2 * pixel * 1e-3, args.maskSizeX/2 * pixel * 1e-3,
            -args.maskSizeY/2 * pixel * 1e-3, args.maskSizeY/2 * pixel * 1e-3
            ]
        kernel_range = np.linspace(-2.8, 2.8, int(kernel_n))
        
        areial_image_cmdc /= np.max(areial_image_cmdc)
        image = axes[0, 0].imshow(
            np.abs(areial_image_cmdc), extent = mask_extent_list, 
            vmin = 0, vmax = 1, origin = 'upper', cmap = 'coolwarm',
            )
        axes[0, 0].set_xlabel("x (%s)" % (unit), fontsize = 12)
        axes[0, 0].set_ylabel("y (%s)" % (unit), fontsize = 12)
        axes[0, 0].set_title("CMDC", fontsize = 12)
        axes[0, 0].yaxis.set_label_coords(-0.20, 0.5)
        divider = make_axes_locatable(axes[0, 0])
        cax = divider.append_axes("right", size = "5%", pad = 0.05) 
        cbar = figure.colorbar(image, cax = cax)
        
        areial_image_2d /= np.max(areial_image_2d)
        
        image = axes[0, 1].imshow(
            np.abs(areial_image_2d), extent = mask_extent_list, 
            vmin = 0, vmax = 1, origin = 'upper', cmap = 'coolwarm',
            )
        axes[0, 1].set_xlabel("x (%s)" % (unit), fontsize = 12)
        axes[0, 1].set_ylabel("y (%s)" % (unit), fontsize = 12)
        axes[0, 1].set_title("2D", fontsize = 12)
        axes[0, 1].set_box_aspect(1) 
        axes[0, 1].yaxis.set_label_coords(-0.20, 0.5)
        divider = make_axes_locatable(axes[0, 1])
        cax = divider.append_axes("right", size = "5%", pad = 0.05) 
        cbar = figure.colorbar(image, cax = cax)
        
        for idx in range(4):
            
            image = axes[1, idx].imshow(
                kernel_list_intensity[idx], extent = [-2.8, 2.8, -2.8, 2.8],
                vmin = 0, vmax = 1, origin = 'upper', cmap = 'coolwarm', 
                )
            axes[1, idx].set_xlabel("$f_x$ ($\mu$m$^{-1}$)", fontsize = 12)
            axes[1, idx].set_ylabel("$f_x$ ($\mu$m$^{-1}$)", fontsize = 12)
            axes[1, idx].yaxis.set_label_coords(-0.20, 0.5)
            divider = make_axes_locatable(axes[1, idx])
            cax = divider.append_axes("right", size = "5%", pad = 0.05) 
            cbar = figure.colorbar(image, cax = cax)
            
            axes[2, idx].plot(
                kernel_range, np.abs(decoupling_vecs['modes_x'][idx])**2 / 
                np.max(np.abs(decoupling_vecs['modes_x'])**2), 
                linewidth = 2, alpha = 0.7
                )
            axes[2, idx].set_xlabel("$f_x$ or $f_y$ ($\mu$m$^{-1}$)", fontsize = 12)
            axes[2, idx].set_ylabel("Intensity (a. u.)", fontsize = 12)
            
            axes[2, idx].plot(
                kernel_range, np.abs(decoupling_vecs['modes_y'][idx])**2 / 
                np.max(np.abs(decoupling_vecs['modes_y'])**2),
                linewidth = 2, alpha = 0.7
                )
            axes[2, idx].set_xlabel("$f_x$ or $f_y$ ($\mu$m$^{-1}$)", fontsize = 12)
            axes[2, idx].set_ylabel("Intensity (a. u.)", fontsize = 12)
            axes[2, idx].yaxis.set_label_coords(-0.25, 0.5)
            axes[2, idx].set_box_aspect(1)  
            
            axes[3, idx].imshow(
                np.abs(coupling_vecs['modes'][idx])**2 / 
                np.max(np.abs(coupling_vecs['modes'][idx])**2), 
                extent = [-2.8, 2.8, -2.8, 2.8],
                vmin = 0, vmax = 1, origin = 'upper', cmap = 'coolwarm', 
                )
            axes[3, idx].set_xlabel("$f_x$ ($\mu$m$^{-1}$)", fontsize = 12)
            axes[3, idx].set_ylabel("$f_y$ ($\mu$m$^{-1}$)", fontsize = 12)
            axes[3, idx].yaxis.set_label_coords(-0.20, 0.5)
            divider = make_axes_locatable(axes[3, idx])
            cax = divider.append_axes("right", size = "5%", pad = 0.05) 
            cbar = figure.colorbar(image, cax = cax)
            
        figure.tight_layout()

        