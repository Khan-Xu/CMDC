# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu"
__date__     = "Date : Wed Oct  8 19:31:56 2025"
__email__    = "xuhan@ihep.ac.cn"


"""
Description
"""

#-----------------------------------------------------------------------------#
# modules

from cat.wave_optics.optics import source_optic, screen, kb
from cat.wave_optics.propagate import fresnel_1d
from cat.wave_optics.widget import plot_optic

from scipy.linalg import svd
from copy import deepcopy

import numpy as np

import sys
sys.path.append(r'D:\Processing\Beamline\simulation\cat')
import os


#-----------------------------------------------------------------------------#
# parameters

file_header = r"D:\File\Paper\CMDC\simulation"
source_file_name = os.path.join(
    file_header, r"HEPS_phaseI_VIB_DL_configuration@12400eV_B4IVULB.h5"
    )

unit = '$\u03bc$m' 

#---------------------------------------------------------
# parameters for vkb

#-----------------------------------------------------------------------------#
# functions

#--------------------------------------------------------------------
# the decoupling function

def decouple_csd(synr_source, counts_mode = 110, threshold = 0.99):

    mode_analysis = list()
    decoupled_modes = list() 
    decoupled_modes_x = list()
    decoupled_modes_y = list()
    coupled_modes = list()
       
    # normalization
    synr_source.ratio /= np.sum(synr_source.ratio)
    total_energy = np.sum(np.array(synr_source.ratio[: counts_mode])**2)
    
    #---------------------------------------------------
    # decoupling of coherent modes
    
    for idx in range(counts_mode):

        coherent_mode = synr_source.cmode[idx]  
        energy_weight = synr_source.ratio[idx]**2
       
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
            
    
    #---------------------------------------------------
    # svd of coupling part
    
    coupled_matrix = np.array([c.flatten() for c in coupled_modes])
    u_vector, singular, vh_vector = svd(coupled_matrix, full_matrices = False)
    
    energy_threshold = (
        threshold - overall_separable_ratio
        ) / overall_coupled_ratio 
    norm_singular = singular / np.sum(np.abs(singular))
    cumulative_energy = (
        np.cumsum(norm_singular**2) / 
        np.sum(norm_singular**2)
        )
    n_modes = np.argmax(cumulative_energy >= energy_threshold) + 1
    
    print(f"Separate Ratio: {overall_separable_ratio:.4f}", flush=True)
    print("compressed modes number: %d" % n_modes, flush = True)
    
    coupled_modes_compressed = list()
    for idx in range(n_modes):
        mode_flat = vh_vector[idx, :]
        mode_2d = mode_flat.reshape(synr_source.xcount, synr_source.ycount)
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
    
    # synr_source.__setattr__("decoupling", decoupling)
    
    #---------------------------------------------------
    # construct new source with decoupling result
    
    synr_source_decoupling = deepcopy(synr_source)
    synr_source_decoupling.n = counts_mode
    synr_source_decoupling.ratio = np.ones(counts_mode)
    synr_source_decoupling.dim = 1
    
    synr_source_decoupling.cmode_x = list()
    synr_source_decoupling.cmode_y = list()
    
    for idx in range(counts_mode):
        
        synr_source_decoupling.n = counts_mode
        synr_source_decoupling.cmode_x.append(decoupled_modes_x[idx])
        synr_source_decoupling.cmode_y.append(decoupled_modes_y[idx])
    
    synr_source_coupling = deepcopy(synr_source)
    synr_source_coupling.cmode = list()
    synr_source_coupling.n = n_modes
    synr_source_coupling.ratio = np.ones(n_modes)
    
    for idx in range(n_modes):
        synr_source_coupling.cmode.append(
            singular[idx] * coupled_modes_compressed[idx]
            )
        
    return synr_source_decoupling, synr_source_coupling, result

#-----------------------------------------------------------------------------#
# classes

#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    count_mode = 94
    
    #---------------------------------------------------
    # source loading and CMDC process
    
    synr_source = source_optic(
        source_file_name = source_file_name, n_vector = int(count_mode), 
        position = 0
        )
    decoupling_source, coupling_source, result = decouple_csd(
        synr_source, counts_mode = int(count_mode)
        )
    
    beamline_calculation = True

    
    if beamline_calculation:
            
        #---------------------------------------------------
        # beamline propagation
        
        # from source screen to dcm screen
        
        source_screen = deepcopy(decoupling_source)
        source_screen.position = 0
        source_screen.n = int(count_mode)
        
        source_screen.interp_optic(pixel = [2.0e-6, 2.0e-6], coor = [2.0e-3, 2.0e-3])
        dcm_screen = screen(
            optic = source_screen, n_vector = 94, position = 40, 
            dim = 1
            )
        fresnel_1d(source_screen, dcm_screen)
        
        # from dcm screen to kb
        
        kb_acceptance = screen(
            optic = dcm_screen, n_vector = int(count_mode), 
            position = 79.1, dim = 1
            )
        fresnel_1d(dcm_screen, kb_acceptance)
        
        # coherent defining slit
        
        kb_acceptance.interp_optic(pixel = [0.14e-6, 0.14e-6], coor = [5.0e-4, 5.00e-4])
        kb_acceptance.mask(
            ycoor = [-1.52e-4/2, 1.52e-4/2], 
            xcoor = [-4.25e-4/2, 4.25e-4/2]
            )
        
        vkb_mirror = kb(
            optic = kb_acceptance, direction = 'h', 
            n_vector = int(count_mode), position = 79.62, pfocus = 79.620, 
            qfocus = 80 - 79.620, dim = 1
            )
        fresnel_1d(kb_acceptance, vkb_mirror)
        
        hkb_mirror = kb(
            optic = vkb_mirror, direction = 'v', n_vector = int(count_mode), 
            position = 79.840, pfocus = 79.840, qfocus = 80 - 79.840, dim = 1
            )
        fresnel_1d(vkb_mirror, hkb_mirror)
        hkb_mirror.interp_optic(
            pixel = [0.35e-8, 0.35e-8], 
            coor = [3.00e-4, 3.00e-4]
            )
        
        # focusing
        
        # aberation = -2.8e-4
            
        focus = screen(optic = hkb_mirror, position = 80, dim = 1)
        fresnel_1d(hkb_mirror, focus)
        
        #---------------------------------------------------
        # visulation
        
        focus.interp_optic(
            pixel = [0.35e-8, 0.35e-8], 
            coor = [5.00e-6, 5.00e-6]
            )
        focus.generate_2d()
        test_plot = plot_optic(focus)
        test_plot.intensity()
    
    
    
