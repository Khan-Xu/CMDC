# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu"
__date__     = "Date : Mon Nov 24 10:30:00 2025"
__email__    = "xuhan@ihep.ac.cn"


"""
Description: A refactored implementation of Lithography Simulation using Implicit SOCS/TCC.
Style aligned with HEPS B4 beamline coding standards.
"""

#-----------------------------------------------------------------------------#
# modules

from scipy.sparse.linalg import svds
from scipy.linalg import svd

import numpy as np

#-----------------------------------------------------------------------------#
# parameters

_DEFAULT_WAVELENGTH = 193.0

#-----------------------------------------------------------------------------#
# functions

#--------------------------------------------------
# support function

def aerial_image_2d_fullstack(image_2d):

    image_2d = np.stack(image_2d, axis = 0)
    areial_image = np.sum(
        image_2d.real * image_2d.real + 
        image_2d.imag * image_2d.imag,
        axis = 0
        )

    return areial_image

def aerial_image_1d_fullstack(
        image_1d, correction_intensity_map = None
        ):
    
    x_stack = np.stack([pair[0] for pair in image_1d], axis = 0)
    y_stack = np.stack([pair[1] for pair in image_1d], axis = 0)
    
    image_2d = np.matmul(np.swapaxes(x_stack, 1, 2), y_stack)
    areial_image = np.sum(
        image_2d.real * image_2d.real + 
        image_2d.imag * image_2d.imag, 
        axis = 0
        )
    areial_image += correction_intensity_map

    return areial_image

def calculate_image_resudual(correction):
    
    image_2d = np.stack(correction, axis = 0)
    areial_image = np.sum(
        image_2d.real * image_2d.real + 
        image_2d.imag * image_2d.imag,
        axis = 0
        )

    return areial_image

def aerial_image_cmdc_fullstack(
        image_1d, residual_convolution = None, u_vector = None, n_res = None
        ):

    x_stack = np.stack([pair[0] for pair in image_1d], axis = 0)
    y_stack = np.stack([pair[1] for pair in image_1d], axis = 0)

    n_modes = x_stack.shape[0]
    field_x = x_stack.shape[2]
    field_y = y_stack.shape[2]
    
    calc_dtype = x_stack.dtype
    real_dtype = x_stack.real.dtype

    aerial_image = np.zeros((field_x, field_y), dtype = real_dtype)
    sep_field = np.zeros((field_x, field_y), dtype = calc_dtype)
    intensity_buffer = np.zeros((field_x, field_y), dtype = real_dtype)

    has_residual = (residual_convolution is not None and u_vector is not None)

    if has_residual:
        k_res = len(residual_convolution) if n_res is None else min(n_res, len(residual_convolution))
        u_used = u_vector[:, :k_res].astype(calc_dtype, copy = False) 
        
        if u_used.shape[0] != n_modes:
            raise ValueError(
                f"u_vector.shape[0] = {u_used.shape[0]} does not match n_modes = {n_modes}"
            ) 
        res_conv_stack = np.array(residual_convolution[:k_res], dtype = calc_dtype)
        temp_res = np.zeros((field_x, field_y), dtype = calc_dtype)

    for i in range(n_modes):
        
        np.matmul(x_stack[i].T, y_stack[i], out=sep_field)

        if has_residual:
            for j in range(k_res):
                np.multiply(res_conv_stack[j], u_used[i, j], out = temp_res)
                sep_field += temp_res

        np.square(sep_field.real, out = intensity_buffer)
        np.square(sep_field.imag, out = intensity_buffer)
        aerial_image += intensity_buffer  
        aerial_image += intensity_buffer 

    return aerial_image

#-------------------------
# lithography simulation

def calculate_convolution_socs_1d(
        mask_layout_1d, eigen_vectors, n_modes, cutoff_x, cutoff_y, 
        output_kernels = False
        ):
    
    mask_layout_1dx, mask_layout_1dy = mask_layout_1d 
    eigen_vector_x, eigen_vector_y = eigen_vectors 
    
    field_x = len(mask_layout_1dx[0])
    field_y = len(mask_layout_1dy[0])
    
    center_y, center_x = field_y // 2, field_x // 2
    slice_y = slice(center_y - cutoff_y, center_y + cutoff_y + 1)
    slice_x = slice(center_x - cutoff_x, center_x + cutoff_x + 1)
    
    n_kernels = int(n_modes)
    
    mask_stack_x = np.array(mask_layout_1dx) 
    mask_stack_y = np.array(mask_layout_1dy)
    
    mask_fft_x = np.fft.fftshift(np.fft.fft(mask_stack_x, axis = -1), axes = -1)
    mask_fft_y = np.fft.fftshift(np.fft.fft(mask_stack_y, axis = -1), axes = -1)
    
    full_kernel_x = np.zeros(field_x, dtype = np.complex64)
    full_kernel_y = np.zeros(field_y, dtype = np.complex64)
    
    #---------------------------------------------------------
    # loop over SOCS Kernels
    
    coherent_image_list = list()
    
    for idx in range(n_kernels):
        
        full_kernel_x.fill(0) 
        full_kernel_y.fill(0)
        
        full_kernel_x[slice_x] = eigen_vector_x[idx]
        full_kernel_y[slice_y] = eigen_vector_y[idx]
        
        image_x = np.fft.ifft(np.fft.ifftshift(mask_fft_x * full_kernel_x, axes = -1), axis = -1)
        image_y = np.fft.ifft(np.fft.ifftshift(mask_fft_y * full_kernel_y, axes = -1), axis = -1)
        
        coherent_image_list.append([image_x, image_y])
        
    return coherent_image_list

def calculate_convolution_socs_2d(
        mask_layout, eigen_values, eigen_vectors, cutoff_x, cutoff_y, 
        output_kernels = False
        ):
    
    field_y, field_x = mask_layout.shape
    
    mask_fourier_transform = np.fft.fftshift(np.fft.fft2(mask_layout)) 

    center_y, center_x = field_y // 2, field_x // 2
    
    slice_y = slice(center_y - cutoff_y, center_y + cutoff_y + 1)
    slice_x = slice(center_x - cutoff_x, center_x + cutoff_x + 1)
    dim_freq_y = 2 * cutoff_y + 1
    dim_freq_x = 2 * cutoff_x + 1

    coherent_image_list = list()
    
    for idx, eigen_val in enumerate(eigen_values):
        
        kernel_2d = eigen_vectors[:, idx].reshape((dim_freq_y, dim_freq_x))
        full_field_kernel = np.zeros(
            (field_y, field_x), dtype = np.complex64
            )
        full_field_kernel[slice_y, slice_x] = kernel_2d
        
        convolution_freq_domain = mask_fourier_transform * full_field_kernel
        coherent_image_list.append(
            np.sqrt(eigen_val) * 
            np.fft.ifft2(np.fft.ifftshift(convolution_freq_domain))
            )
        
    return coherent_image_list

def calculate_convolution_resudual(
        mask_layout, coupling_result, n_correction_modes, 
        cutoff_x, cutoff_y
        ):

    field_y, field_x = mask_layout.shape
    mask_fourier_transform = np.fft.fftshift(np.fft.fft2(mask_layout)) 
    
    correction_convolution = list()
    
    center_y, center_x = field_y // 2, field_x // 2
    slice_y = slice(center_y - cutoff_y, center_y + cutoff_y + 1)
    slice_x = slice(center_x - cutoff_x, center_x + cutoff_x + 1)

    n_process = min(n_correction_modes, len(coupling_result['modes']))
    
    for idx in range(n_process):
        
        kernel_2d = coupling_result['modes'][idx]
        full_field_kernel = np.zeros(
            (field_y, field_x), dtype = np.complex64
            )
        full_field_kernel[slice_y, slice_x] = kernel_2d
        correction_convolution.append(np.fft.ifft2(
            np.fft.ifftshift(mask_fourier_transform * full_field_kernel)
            ))
            
    return correction_convolution

#-----------------------------------------------------------------------------#
# classes

class lithography_simulation(object):
    
    def __init__(self, wavelength = _DEFAULT_WAVELENGTH):
        
        self.wavelength = wavelength

    def _get_sigma_outer(self, source_intensity):
        
        non_zero_indices = np.argwhere(source_intensity > 1e-6)
        if non_zero_indices.size == 0:
            return 0.0
        
        center_coordinate = (source_intensity.shape[0] - 1) / 2.0
        
        pixel_distances = np.sqrt(
            (non_zero_indices[:, 1] - center_coordinate)**2 + 
            (non_zero_indices[:, 0] - center_coordinate)**2
            )
        max_distance = np.max(pixel_distances)
        
        return max_distance / center_coordinate
    
    def calculate_socs_implicit(
            self, source_intensity, numerical_aperture, defocus_distance, 
            field_pixel_count_x, field_pixel_count_y, sigma_outer, kernel_count,
            svd_method = "svd"
            ):
        
        source_grid_size = source_intensity.shape[0]
        wave_vector_k = 2 * np.pi / self.wavelength
        
        #-----------------------------------------------------------
        # determine frequency grid size
        
        xcount, ycount = field_pixel_count_x, field_pixel_count_y
        na = numerical_aperture
        
        frequency_cutoff_x, frequency_cutoff_y = [
            int(np.floor(na * count * (1 + sigma_outer) / self.wavelength)) 
            for count in [xcount, ycount]
            ]
        total_frequency_pixels = (
            (2*frequency_cutoff_x + 1) * (2*frequency_cutoff_y + 1)
            )
        
        #-----------------------------------------------------------
        # build system matrix H

        source_point_indices = np.argwhere(source_intensity > 0)
        source_point_count = len(source_point_indices)

        if kernel_count > source_point_count:
            kernel_count = source_point_count
        
        system_matrix_h = np.zeros(
            (total_frequency_pixels, source_point_count), 
            dtype = np.complex128
            )
        center_source_index = (source_grid_size - 1) / 2.0
        field_normalization_x, field_normalization_y = [
            count * na / self.wavelength for count in [xcount, ycount]
            ]
        # defocus_factor = defocus_distance / (
        #     numerical_aperture**2 / self.wavelength
        #     )
        
        frequency_range_x = np.arange(-frequency_cutoff_x, frequency_cutoff_x + 1)
        frequency_range_y = np.arange(-frequency_cutoff_y, frequency_cutoff_y + 1)
        grid_freq_x, grid_freq_y = np.meshgrid(
            frequency_range_x, frequency_range_y
            ) 
        
        for idx, point_coordinate in enumerate(source_point_indices):
            
            pixel_y, pixel_x = point_coordinate
            intensity_value = source_intensity[pixel_y, pixel_x]
            amplitude_value = np.sqrt(intensity_value)
            
            radius_squared = (
                (grid_freq_x / field_normalization_x + ((pixel_x - center_source_index) / center_source_index))**2 +
                (grid_freq_y / field_normalization_y + ((pixel_y - center_source_index) / center_source_index))**2
                )
            
            mask_pupil_valid = (radius_squared <= 1.0)
            pupil_column_vector = np.zeros_like(
                radius_squared, dtype = np.complex128
                )
            valid_phase_mask = mask_pupil_valid & (
                (1.0 - radius_squared * numerical_aperture**2) >= 0
                )
            
            if np.any(valid_phase_mask):
                sqrt_term = np.sqrt(
                    1.0 - radius_squared[valid_phase_mask] * numerical_aperture**2
                    )
                phase_phi = defocus_distance * wave_vector_k * (sqrt_term - 1)
                pupil_column_vector[valid_phase_mask] = (
                    np.cos(phase_phi) + 1j * np.sin(phase_phi)
                    )
            
            system_matrix_h[:, idx] = pupil_column_vector.flatten() * amplitude_value
            
        #-----------------------------------------------------------
        # kernel calculation
        
        print("SVD calculation start (Matrix H)..... ", flush = True)
        
        svd_rank_k = min(kernel_count, min(system_matrix_h.shape) - 1)
        if svd_rank_k < 1: svd_rank_k = 1
        
        if svd_method == "svd":
            
            left_singular_vec, singular_val, right_singular_vec = svd(
                system_matrix_h, full_matrices = False
                )
            left_singular_vec = left_singular_vec[:, :kernel_count]
            singular_val = singular_val[:kernel_count]
        
        # Warning! the random phase from SVDs introduce unstablilty for CMDC!
        elif svd_method == "svds":
            
            left_singular_vec, singular_val, right_singular_vec = svds(
                system_matrix_h, k = svd_rank_k, which = 'LM'
                )
            left_singular_vec = left_singular_vec[:, ::-1]
            singular_val = singular_val[::-1]
            
        eigen_values = singular_val**2
        eigen_vectors = left_singular_vec 
            
        return (
            eigen_values, eigen_vectors, 
            frequency_cutoff_x, frequency_cutoff_y
            )
    
#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    pass