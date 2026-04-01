#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.12.2021"
__version__  = "beta-0.3"
__email__    = "xuhan@ihep.ac.cn"

"""
_source_utils: Source construction support.

Functions: None
           
Classes  : _optic_plane - the geometry structure of optics
"""

#-----------------------------------------------------------------------------#
# library

import sys
import os
import math

import numpy as np
import h5py as h5

from scipy import interpolate
from scipy.linalg import svd
from copy import deepcopy
from scipy import fft

#-----------------------------------------------------------------------------#
# constant

#-----------------------------------------------------------------------------#
# function

def _locate(ticks, value):
    
    """
    This function searches through an array of 'ticks' and returns the index of the entry
    that is closest to the specified 'value'. If the 'value' is outside the range of 'ticks',
    a ValueError is raised.
    
    Args:
        ticks (np.ndarray): An array of numerical values.
        value (float): The value to locate within 'ticks'.
    
    Raises:
        ValueError: If 'value' is outside the range of 'ticks'.
    
    Returns:
        int: The index of the closest entry in 'ticks' to 'value'.
    """

    if value > np.max(ticks) or value < np.min(ticks):
        raise ValueError("Unsupported value range: {}".format(value))
    else:
        return np.argmin(np.abs(ticks - value)) 

def _geometry(start, end, count):
    
    """
   Calculate the geometry parameters based on the start, end, and count.

   Args:
       start (float): The starting value of the range.
       end (float): The ending value of the range.
       count (int): The number of elements to generate between start and end.

   Returns:
       list: A list containing the step size, the start and end values, and an array of
       linearly spaced values between start and end.

   Example:
       >>> _calculate_geometry_parameters(0, 10, 5)
       [2.0, [0, 10], [array([ 0.,  2.,  4.,  6.,  8., 10.])]]
   """
   
    return [(end - start) / (count - 1), [start, end], np.linspace(start, end, int(count))]

def one_dimensional_cmode(cmode, xcount, ycount):
    
    cmode_x, cmode_y = [list(), list()]
    
    for idx_cmode in cmode:
        
        cmode_x.append(
            idx_cmode[int(ycount//2), :] if ycount%2 == 1 else 
            (idx_cmode[int(ycount//2), :] + idx_cmode[int(ycount//2 - 1), :])/2
            )
        cmode_y.append(
            idx_cmode[:, int(xcount//2)] if ycount%2 == 1 else 
            (idx_cmode[:, int(xcount//2)] + idx_cmode[:, int(ycount//2 - 1)])/2
            )
        
    return cmode_x, cmode_y

def fourier_shift(data, offset):

    x_count, y_count = data.shape
    
    #------------------------------------------------------
    # construct y shift phase
    
    y_meshgrid = np.hstack([
        np.arange(np.floor(x_count/2), dtype = np.int),
        np.arange(np.floor(-x_count/2), 0, dtype = np.int)
        ])
    y_shift = np.exp(-1j * 2 * np.pi * offset[0] * y_meshgrid / y_count)
    y_shift = y_shift[None, :] 
    # if np.mod(y_count, 2) == 0:
    #     y_shift[:, y_count//2] = np.real(y_shift[:, y_count//2])
    
    #------------------------------------------------------
    # construct x shift phase
    
    x_meshgrid = np.hstack([
        np.arange(np.floor(y_count/2), dtype = np.int),
        np.arange(np.floor(-y_count/2), 0, dtype = np.int)
        ])
    x_shift = np.exp(-1j * 2 * np.pi * offset[1] * x_meshgrid / x_count)
    x_shift = x_shift[:, None]  
    # if np.mod(x_count, 2) == 0:
    #     x_shift[x_count//2] = np.real(x_shift[x_count//2])
    
    #------------------------------------------------------
    # shifted data
    
    shifted = np.fft.ifft2(
        np.fft.fft2(data) * (x_shift * y_shift)
        )
    
    return shifted


def decouple_csd(synr_source, counts_mode = 110):

    mode_analysis = list()
    decoupled_modes = list() 
    coupled_modes = list()
       
    total_energy = np.sum(np.array(synr_source.ratio[: counts_mode])**2)
      
    #---------------------------------------------------
    # decoupling of coherent modes
    
    for idx in range(counts_mode):

        coherent_mode = synr_source.cmode[idx]  
        energy_weight = synr_source.ratio[idx]**2 
       
        u_vector, singular, vh_vector = svd(coherent_mode, full_matrices = False)
        separable_part = singular[0] * np.outer(u_vector[:, 0], vh_vector[0, :])
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
       
        decoupled_modes.append(energy_weight * separable_part)
        coupled_modes.append(energy_weight * residual_part)
       
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
    
    # 99% total energy
    energy_threshold = (overall_coupled_ratio - 0.01) / overall_coupled_ratio 
    norm_singular = singular / np.sum(np.abs(singular))
    cumulative_energy = (
        np.cumsum(norm_singular**2) / 
        np.sum(norm_singular**2)
        )
    n_modes = np.argmax(cumulative_energy >= energy_threshold) + 1
       
    # 重构压缩后的耦合模式
    coupled_modes_compressed = list()
    for idx in range(n_modes):
        mode_flat = vh_vector[idx, :]
        mode_2d = mode_flat.reshape(synr_source.xcount, synr_source.ycount)
        coupled_modes_compressed.append(mode_2d)
       
    # 返回完整分析结果
    decoupling = {
        'mode_analysis': mode_analysis,
        'overall_analysis': {
            'total_energy': total_energy,
            'total_separable_energy': total_separable_energy,
            'total_coupled_energy': total_coupled_energy,
            'separable_ratio': overall_separable_ratio,
            'coupled_ratio': overall_coupled_ratio
            },
        'decoupled_modes': decoupled_modes,
        'coupled_modes_compressed': coupled_modes_compressed,
        'coupled_singular_values': singular,
        'coupled_cumulative_energy': cumulative_energy
        }
    
    synr_source.__setattr__("decoupling", decoupling)
    
    return synr_source

#-----------------------------------------------------------------------------#
# class

class _op(object):
    
    #--------------------------------------------------------------------------
    # the initialization of the optic plane

    def __init__(
        self, 
        optic = None, optic_file = None, name = "optic", n_vector = 0, 
        i_vector = None, position = None, dim = 2 
        ):
        
        """
        Initialize the optic_class with the provided parameters.
        
        Parameters:
            - optic: instance of optic_class, optional
            - optic_file: str, optional
            - name: str, default='optic', the name of the optic
            - n_vector: int, default=0, the number of coherent mode to propagate
            - i_vector: int, optional, the sequence number of coherent mode to propagate
            - position: int, default=0, the position of the optic plane
        
        Raises:
            - ValueError: if optic and optic_file are both None
        
        Notes:
            - If optic is provided, it initializes the optic class with the previous optic class.
            - Initializes the optic class with HDF5 optic file or optic_class.
            - If optic_file is provided, it constructs the optic class with the file content.
        """

        #------------------------------------------------------
        # parameter check
        
        if optic == None and optic_file == None:
            raise ValueError("initialize the optic_class with hdf5 optic file or optic_class.")
        
        #------------------------------------------------------
        # initalize the optic class with hdf5 optic file
        
        if optic_file != None and optic == None:
            
            if not os.path.isfile(optic_file):
                raise ValueError("The optic file: {%s} dose not exist!" % (optic_file))
        
            self.optic_name = optic_file
            self.n = n_vector
            self.n_vector = n_vector
            self.dim = dim
            
            with h5.File(self.optic_name, 'r') as optic_file:
                
                #------------------------------------------------------
                # construct geomtry parameters
                
                self.position = np.array(optic_file["optic_plane/position"]) if position is None else position

                geometry_x_path_list = ["xstart", "xend", "xcount"]
                self.xstart, self.xend, self.xcount = [
                    np.array(optic_file["optic_plane/" + x_path]) 
                    for x_path in geometry_x_path_list
                    ]
                
                # construct geometry y parameters
                
                geometry_y_path_list = ["ystart", "yend", "ycount"]
                self.ystart, self.yend, self.ycount = [
                    np.array(optic_file["optic_plane/" + y_path]) 
                    for y_path in geometry_y_path_list
                    ]
                
                #------------------------------------------------------
                # coherence parameters
                
                self.wavelength = np.array(optic_file["optic_plane/wavelength"])
                self.ratio = np.array(optic_file["coherence/ratio"]).tolist()
                
                try:
                    self.evolution = np.array(optic_file["coherence/evolution"])
                except:
                    self.evolution = list()
                
                self.cmode = list()
                self.cmode_x = list()
                self.cmode_y = list()
                
                # construct coherence parameters
                coherence_csd_path_list = ["csd2x", "csd1x", "csd2y", "csd1y"]
                self.csd2x, self.csd1x, self.csd2y, self.csd1y = [
                    np.array(optic_file["coherence/" + coherence_csd_path]) 
                    for coherence_csd_path in coherence_csd_path_list
                    ]
                coherence_sdc_path_list = ["sdc2x", "sdc1x", "sdc2y", "sdc1y"]
                self.sdc2x, self.sdc1x, self.sdc2y, self.sdc1y = [
                    np.array(optic_file["coherence/" + coherence_sdc_path]) 
                    for coherence_sdc_path in coherence_sdc_path_list
                    ]
                
                if self.dim == 2:
                    
                    cmode_path = "coherence/coherent_mode"
                    if i_vector != None:
                        self.n = 1
                        self.cmode = [np.array(
                            optic_file[cmode_path][int(i_vector), :, :], 
                            dtype = np.complex64
                            )]
                    elif i_vector == None:
                        self.cmode = [
                            np.array(optic_file[cmode_path][i, :, :], dtype = np.complex64) 
                            for i in range(n_vector)
                            ]
                    self.xpixel, self.xcoor, self.xtick = _geometry(self.xstart, self.xend, self.xcount)
                    self.ypixel, self.ycoor, self.ytick = _geometry(self.ystart, self.yend, self.ycount)
                    self.xgrid, self.ygrid = np.meshgrid(self.xtick, self.ytick)
                
                elif self.dim == 1:

                    cmodex_path = "coherence/coherent_modex"
                    cmodey_path = "coherence/coherent_modey"
                    
                    if i_vector != None:
                        self.n = 1
                        self.cmode_x = [np.array(
                            optic_file[cmodex_path][int(i_vector), :], 
                            dtype = np.complex64
                            )]
                        self.cmode_y = [np.array(
                            optic_file[cmodey_path][int(i_vector), :], 
                            dtype = np.complex64
                            )]
                    elif i_vector == None:
                        
                        self.cmode_x = [
                            np.array(optic_file[cmodex_path][i, :], dtype = np.complex64) 
                            for i in range(n_vector)
                            ]
                        self.cmode_y = [
                            np.array(optic_file[cmodey_path][i, :], dtype = np.complex64) 
                            for i in range(n_vector)
                            ]
                    
                    self.xpixel, self.xcoor, self.xtick = _geometry(self.xstart, self.xend, self.xcount)
                    self.ypixel, self.ycoor, self.ytick = _geometry(self.ystart, self.yend, self.ycount)
                        
        #------------------------------------------------------
        # initalize the optic class with previous optic class
                
        if optic_file == None and optic != None:
            
            self.optic_name = name
            self.n = n_vector
            self.n_vector = n_vector
            self.dim = dim
            
            #------------------------------------------------------
            # construct geomtry parameters
            
            self.xstart, self.xend, self.xcount = [
                getattr(optic, name) for name in ["xstart", "xend", "xcount"]
                ]
            self.ystart, self.yend, self.ycount = [
                getattr(optic, name) for name in ["ystart", "yend", "ycount"]
                ]
            
            parameters_to_initalize = [
                "csd2x", "csd1x", "csd2y", "csd1y", 
                "sdc2x", "sdc1x", "sdc2y", "sdc1y", "evolution"
                ]
            for name in parameters_to_initalize: setattr(self, name, 0)
            
            for name in ["wavelength", "n", "ratio"]:
                setattr(self, name, getattr(optic, name))
                
            if hasattr(optic, "cmode_index"): self.cmode_index = optic.cmode_index
            self.position = position
            
            #------------------------------------------------------
            # load coherent modes
            
            if dim == 2:
                
                self.cmode = list()
                for i in range(self.n): 
                    self.cmode.append(np.ones((int(self.ycount), int(self.xcount)), dtype = np.complex64))
                
                self.xpixel, self.xcoor, self.xtick = _geometry(self.xstart, self.xend, self.xcount)
                self.ypixel, self.ycoor, self.ytick = _geometry(self.ystart, self.yend, self.ycount)
                self.xgrid, self.ygrid = np.meshgrid(self.xtick, self.ytick)
                
            elif dim == 1:
                
                self.cmode_x, self.cmode_y = [list(), list()]
                for i in range(self.n):
                    self.cmode_x.append(np.ones(self.xcount, dtype = np.complex64))
                    self.cmode_y.append(np.ones(self.ycount, dtype = np.complex64))
                
                self.xpixel, self.xcoor, self.xtick = _geometry(self.xstart, self.xend, self.xcount)
                self.ypixel, self.ycoor, self.ytick = _geometry(self.ystart, self.yend, self.ycount)
                
        # self.dim = dim
            
        #------------------------------------------------------
        # calculate the geometry parameters
        
        self.xcount = int(self.xcount)
        self.ycount = int(self.ycount)
        
        self.n_column = np.copy(self.xcount)
        self.n_row = np.copy(self.ycount)
        
    #--------------------------------------------------------------------------
    # get and set method
    
    def __getattr__(self, name: str):
        
        if (names := self.__dict__.get('names')) and name in names:
            return self[name]
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )

    def __setattr__(self, name: str, value):
        
        if (names := self.__dict__.get('names')) and name in names:
            self[name] = value
        super().__setattr__(name, value)
    
    #--------------------------------------------------------------------------
    # the geometry operation on the optic plane
    
    #------------------------------------------------------
    # shift pixel
    
    def shift_pixel_geometry(self, distance, direction = "x"):
        
        """
        Shifts the pixel geometry attributes in the specified direction by n pixels.
        
        Parameters:
            - distance (float): The distance to shift the geometry by.
            - direction (str, optional): The direction in which to shift the geometry. 
              Can be either "x" for horizontal or "y" for vertical. Default is "x".
        
        Returns:
            None
        
        Actions:
            - If direction is "x", shifts attributes xstart, xend, and xtick by distance.
            - If direction is "y", shifts attributes ystart, yend, and ytick by distance.
        
        Also Updates:
            - xgrid (ndarray): Meshgrid of xtick values.
            - ygrid (ndarray): Meshgrid of ytick values.
        """

        if direction == "x":
            for attribute in ["xstart", "xend", "xtick"]:
                setattr(self, attribute, getattr(self, attribute) - distance)
        elif direction == "y":
            for attribute in ["ystart", "yend","ytick"]:
                setattr(self, attribute, getattr(self, attribute) - distance)
                
        self.xgrid, self.ygrid = np.meshgrid(self.xtick, self.ytick)
    
    #------------------------------------------------------
    # update pixel, coor of the optic plane
    
    def interp_optic(
            self, pixel = None, coor = None, power2 = False, method = "phase_unwrap", 
            update_geometry_parameters = True, even = True
            ):
        
        """
        Interpolate the optic plane data based on the specified method.
        
        Parameters:
            pixel (list): A list of two elements [xpixel, ypixel] representing the pixel values.
            coor (list): A list of two elements [xcoor, ycoor] representing the coordinates.
            method (str): Method of interpolation, can be 'phase_unwrap' (default), 'ri', 'ap', or 'unwrap'.
        
        Raises:
            ValueError: If both pixel and coor are None, or if pixel or coor format is incorrect.
        
        Returns:
            None. Updates the self.cmode attribute in place.
        """

        #------------------------------------------------------
        # check parameters
        
        if pixel == None and coor == None and not power2:
            raise ValueError(
                "The interpolation of optic_plane {%s} is not required" % (self.optic_name)
                )
        
        elif pixel != None and coor == None and not power2:
            
            if isinstance(pixel, list) and len(pixel) == 2:
                xcoor = getattr(self, "xcoor")
                ycoor = getattr(self, "ycoor")
                ypixel, xpixel = pixel
            else:
                raise ValueError("Require argument {pixel} as [xpixel, ypixel] list")
                
        elif coor != None and pixel == None and not power2:
            
            if isinstance(coor, list) and len(coor) == 2:
                
                xpixel = getattr(self, "xpixel")
                ypixel = getattr(self, "ypixel")
                
                if isinstance(coor[0], list) and isinstance(coor[1], list): 
                    ycoor, xcoor = coor
                else:
                    xcoor = [-0.5 * coor[0], 0.5 * coor[0]]
                    ycoor = [-0.5 * coor[1], 0.5 * coor[1]]
                    
            else:
                raise ValueError("Require argument {coor} as [xcoor, ycoor] list")
        
        elif coor == None and pixel == None and power2:
            
            if isinstance(power2, bool):
                
                xcoor = getattr(self, "xcoor")
                xcount = int(2**int(np.log(self.xcount) / np.log(2)))
                xpixel = (xcoor[1] - xcoor[0]) / (xcount - 1)
                
                ycoor = getattr(self, "ycoor")
                ycount = int(2**int(np.log(self.ycount) / np.log(2)))
                ypixel = (ycoor[1] - ycoor[0]) / (ycount - 1)
            
            else:
                raise ValueError("Require argument {power2} as bool")
                
        elif coor != None and pixel != None and not power2:
            
            if isinstance(pixel, list) and len(pixel) == 2:
                if isinstance(coor, list) and len(coor) == 2:
                    ypixel, xpixel = pixel
                    
                    if isinstance(coor[0], list) and isinstance(coor[1], list): 
                        ycoor, xcoor = coor
                    else:
                        xcoor = [-0.5 * coor[0], 0.5 * coor[0]]
                        ycoor = [-0.5 * coor[1], 0.5 * coor[1]]
                        
                else: raise ValueError("Require argument {coor} as [xcoor, ycoor] list")
            else: raise ValueError("Require argument {pixel} as [xpixel, ypixel] list")

        else: raise ValueError("No interp arguments!")
                    
        # calculate geometry parameters
        
        xcount = int((xcoor[1] - xcoor[0]) / xpixel + 1)
        ycount = int((ycoor[1] - ycoor[0]) / ypixel + 1)
        
        if even:
            xcount = xcount - xcount%2
            ycount = ycount - ycount%2
        else:
            xcount = xcount + (xcount%2 - 1)
            ycount = ycount + (ycount%2 - 1)
        
        self.even = even
        
        xpixel = (xcoor[1] - xcoor[0]) / (xcount - 1)
        ypixel = (ycoor[1] - ycoor[0]) / (ycount - 1)
        
        xtick = np.linspace(xcoor[0], xcoor[1], xcount)
        ytick = np.linspace(ycoor[0], ycoor[1], ycount)
        
        if self.dim == 2:
            
            pts = np.meshgrid(xtick, ytick, indexing = 'xy')
            pts = (
                np.reshape(pts[0], (1, int(xcount * ycount))), 
                np.reshape(pts[1], (1, int(xcount * ycount)))
                )
            
            #------------------------------------------------------
            # interpolate real and image data
            
            if method == 'ri':
                
                for i in range(self.n):
                    
                    freal = interpolate.RegularGridInterpolator(
                        (self.xtick, self.ytick), np.real(self.cmode[i]), method = "linear",
                        bounds_error = False, fill_value = 0
                        )
                    fimag = interpolate.RegularGridInterpolator(
                        (self.xtick, self.ytick), np.imag(self.cmode[i]), method = "linear",
                        bounds_error = False, fill_value = 0
                        )
                    self.cmode[i] = np.array(np.rot90(np.reshape(
                        freal(pts) + 1j * fimag(pts), (ycount, xcount)
                        )), dtype = np.complex64)
            
            #------------------------------------------------------
            # interpolate abs and phase data
    
            elif method == 'ap':
                
                for i in range(self.n):
                
                    fabs = interpolate.RegularGridInterpolator(
                        (self.xtick, self.ytick), np.abs(self.cmode[i]), method = "linear",
                        bounds_error = False, fill_value = 0
                        )
                    fangle = interpolate.RegularGridInterpolator(
                        (self.xtick, self.ytick), np.angle(self.cmode[i]), method = "linear",
                        bounds_error = False, fill_value = 0
                        )
                    self.cmode[i] = np.array(np.rot90(np.reshape(
                        fabs(pts) * np.exp(1j * fangle(pts)), (ycount, xcount)
                        )), dtype = np.complex64)
            
            #------------------------------------------------------
            # interpolate abs and phase data after unwrap data
    
            elif method == 'phase_unwrap':
                    
                from skimage.restoration import unwrap_phase
                
                for i in range(self.n):
                    
                    unwraped_phase = unwrap_phase(np.angle(self.cmode[i]))
                    
                    fabs = interpolate.RegularGridInterpolator(
                        (self.xtick, self.ytick), np.abs(self.cmode[i]), method = "linear",
                        bounds_error = False, fill_value = 0
                        )
                    fangle = interpolate.RegularGridInterpolator(
                        (self.xtick, self.ytick), unwraped_phase, method = "linear",
                        bounds_error = False, fill_value = 0
                        )
                    self.cmode[i] = np.array(np.rot90(np.reshape(
                        fabs(pts) * np.exp(1j * fangle(pts)), (ycount, xcount)
                        )), dtype = np.complex64)
                    
            elif method == 'mask':
                
                from skimage.restoration import unwrap_phase
                
                for i in range(self.n):
                    
                    unwraped_phase = unwrap_phase(np.angle(self.cmode[i]))
                    
                    fabs = interpolate.RegularGridInterpolator(
                        (self.xtick, self.ytick), np.abs(self.cmode[i]), method = "nearest",
                        bounds_error = False, fill_value = 0
                        )
                    fangle = interpolate.RegularGridInterpolator(
                        (self.xtick, self.ytick), unwraped_phase, method = "nearest",
                        bounds_error = False, fill_value = 0
                        )
                    self.cmode[i] = np.array(np.rot90(np.reshape(
                        fabs(pts) * np.exp(1j * fangle(pts)), (ycount, xcount)
                        )), dtype = np.complex64)
                    
        elif self.dim == 1:
            
            #------------------------------------------------------
            # interpolate abs and phase data after unwrap data

            kind = "linear" if method == "phase_unwrap" else "nearest"
                    
            from skimage.restoration import unwrap_phase
                
            for i in range(self.n):
                
                # interpolation along direction x
                
                unwraped_phase_x = unwrap_phase(np.angle(self.cmode_x[i]))
                
                fabs_x = interpolate.interp1d(
                    self.xtick, np.abs(self.cmode_x[i]), kind = kind, 
                    bounds_error = False, fill_value = 0
                    )
                fangle_x = interpolate.interp1d(
                    self.xtick, unwraped_phase_x, kind = kind, bounds_error = False, fill_value = 0
                    )
                self.cmode_x[i] = fabs_x(xtick) * np.exp(1j * fangle_x(xtick))  
                
                # interpolation along direction y
                
                unwraped_phase_y = unwrap_phase(np.angle(self.cmode_y[i]))
                
                fabs_y = interpolate.interp1d(
                    self.ytick, np.abs(self.cmode_y[i]), kind = kind, 
                    bounds_error = False, fill_value = 0
                    )
                fangle_y = interpolate.interp1d(
                    self.ytick, unwraped_phase_y, kind = kind, bounds_error = False, fill_value = 0
                    )
                self.cmode_y[i] = fabs_y(ytick) * np.exp(1j * fangle_y(ytick)) 
                        
        #------------------------------------------------------
        # update the optic plane parameters
        
        if update_geometry_parameters:
 
            self.xstart, self.xend = ycoor
            self.xcoor = deepcopy(ycoor)
            self.xtick = xtick
            self.xcount = int(xcount)
            self.n_column = self.xcount
            self.xpixel = xpixel
            
            self.ystart, self.yend = xcoor
            self.ycoor = deepcopy(xcoor)
            self.ytick = ytick
            self.ycount = int(ycount)
            self.n_row = self.ycount
            self.ypixel = ypixel
            
            if self.dim == 2:
                self.xgrid, self.ygrid = np.meshgrid(self.xtick, self.ytick)
            
    #------------------------------------------------------
    # add the mask of to the coheret mode
            
    def mask(
            self, xcoor = None, ycoor = None, r = None, shape = "b", 
            even = True, apply = True
            ):
        
        """
        Generate a mask for specific regions of a grid.
        
        Parameters:
            xcoor (int or None): x-coordinate for the center of the region. Default is None.
            ycoor (int or None): y-coordinate for the center of the region. Default is None.
            r (int or None): radius of the circular region. Default is None.
            shape (str): Shape of the region, "b" for box region or "c" for circular region. Default is "b".
        
        Returns:
            None
        
        Generates a mask for specific regions of a grid based on the parameters provided. 
        If shape is "b", the mask will be created based on the interpolated optic values at the given coordinates. 
        If shape is "c", a circular mask with radius r will be created and applied to the grid.
        
        Examples:
            mask(xcoor=10, ycoor=5, shape="b")  # Generates a mask for a box region centered at (10, 5)
            mask(r=7, shape="c")  # Generates a mask for a circular region with radius 7
        """
        
        xcount = int((xcoor[1] - xcoor[0]) / self.xpixel + 1)
        ycount = int((ycoor[1] - ycoor[0]) / self.ypixel + 1)
        
        if even:
            xcount = xcount - xcount%2
            ycount = ycount - ycount%2
        else:
            xcount = xcount + (xcount%2 - 1)
            ycount = ycount + (ycount%2 - 1)
        
        xtick, ytick = (
            np.linspace(xcoor[0], xcoor[1], xcount), np.linspace(ycoor[0], ycoor[1], ycount)
            )
        mesh_points = np.meshgrid(xtick, ytick)
        mesh_points = (
            np.reshape(mesh_points[0], (1, int(xcount * ycount))), 
            np.reshape(mesh_points[1], (1, int(xcount * ycount)))
            )
        
        if self.dim == 2:
            
            if shape == "b":
            
                fmask_before = interpolate.RegularGridInterpolator(
                    (self.xtick, self.ytick), np.ones((self.xcount, self.ycount)), method = "nearest",
                    bounds_error = False, fill_value = 0
                    )
                mask = fmask_before(mesh_points).reshape((xcount, ycount))
                
                # return mask
            
                fmask_after = interpolate.RegularGridInterpolator(
                    (xtick, ytick), mask, method = "nearest", 
                    bounds_error = False, fill_value = 0
                    )
                self.mask_2d = fmask_after((
                    np.reshape(self.xgrid, (1, int(self.xcount * self.ycount))),
                    np.reshape(self.ygrid, (1, int(self.xcount * self.ycount)))
                    )).reshape(self.xcount, self.ycount)
                
            elif shape == "c":
                
                self.mask_2d = np.zeros((self.n_row, self.n_column))
                self.mask_2d[np.sqrt(self.xgrid**2 + self.ygrid**2) < r] = 1
              
            if apply:
                for i in range(self.n): 
                    self.cmode[i] = np.array(
                        self.cmode[i] * self.mask_2d, dtype = np.complex64
                        )
                
        elif self.dim == 1:
            
            fmask_x_before = interpolate.interp1d(
                self.xtick, np.ones(self.xcount), kind = "nearest", 
                bounds_error = False, fill_value = 0
                )
            fmask_y_before = interpolate.interp1d(
                self.ytick, np.ones(self.ycount), kind = "nearest", 
                bounds_error = False, fill_value = 0
                )
            
            mask_x_before = fmask_x_before(xtick)
            mask_y_before = fmask_y_before(ytick)
            
            fmask_x_after = interpolate.interp1d(
                xtick, mask_x_before, kind = "nearest", bounds_error = False, fill_value = 0
                )
            fmask_y_after = interpolate.interp1d(
                ytick, mask_y_before, kind = "nearest", bounds_error = False, fill_value = 0
                )
            
            self.mask_x = fmask_x_after(self.xtick)
            self.mask_y = fmask_y_after(self.ytick)
            
            for i in range(self.n): 
                self.cmode_x[i] = self.cmode_x[i] * self.mask_x
                self.cmode_y[i] = self.cmode_y[i] * self.mask_y
                     
    #------------------------------------------------------
    # generate 2d coherent modes based on the 1d cm
            
    def generate_2d_gaussian(self):
        
        self.dim = 2
        self.xgrid, self.ygrid = np.meshgrid(self.xtick, self.ytick)
        
        self.cmode = [
            np.array(np.dot(
                np.matrix(self.cmode_y[idy]).T, np.matrix(self.cmode_x[idx])
                ), dtype = np.complex64
                )
            for idx, idy in self.cmode_index[0 : int(self.n), :]
            ]
        
    def generate_2d(self):
        
        self.dim = 2
        self.xgrid, self.ygrid = np.meshgrid(self.xtick, self.ytick)
        
        # self.cmode = [
        #     np.outer(self.cmode_x[idx], self.cmode_y[idx]) / self.ratio[idx]
        #     for idx in range(self.n)
        #     ]
        self.cmode = [
            np.outer(self.cmode_x[idx], self.cmode_y[idx]) 
            for idx in range(self.n)
            ]
    
    #--------------------------------------------------------------------------
    # coherence property calculation

    #------------------------------------------------------
    # decoupling process
    
    def decoupling(self, total_energy = 0.99):
        
        mode_analysis = list()
        decoupled_modes = list() 
        coupled_modes = list()
           
        total_energy = np.sum(np.array(self.ratio)**2)
          
        #---------------------------------------------------
        # decoupling of coherent modes
        
        for idx in range(self.n):

            coherent_mode = self.cmode[idx]  
            energy_weight = self.ratio[idx]**2 
           
            u_vector, singular, vh_vector = svd(coherent_mode, full_matrices = False)
            separable_part = singular[0] * np.outer(u_vector[:, 0], vh_vector[0, :])
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
           
            decoupled_modes.append(energy_weight * separable_part)
            coupled_modes.append(energy_weight * residual_part)
           
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
        
        # 99% total energy
        energy_threshold = (overall_coupled_ratio - 0.01) / overall_coupled_ratio 
        norm_singular = singular / np.sum(np.abs(singular))
        cumulative_energy = (
            np.cumsum(norm_singular**2) / 
            np.sum(norm_singular**2)
            )
        n_modes = np.argmax(cumulative_energy >= energy_threshold) + 1
           
        coupled_modes_compressed = list()
        for idx in range(n_modes):
            mode_flat = vh_vector[idx, :]
            mode_2d = mode_flat.reshape(self.xcount, self.ycount)
            coupled_modes_compressed.append(mode_2d)
           
        decoupling = {
            'mode_analysis': mode_analysis,
            'overall_analysis': {
                'total_energy': total_energy,
                'total_separable_energy': total_separable_energy,
                'total_coupled_energy': total_coupled_energy,
                'separable_ratio': overall_separable_ratio,
                'coupled_ratio': overall_coupled_ratio
                },
            'decoupled_modes': decoupled_modes,
            'coupled_modes_compressed': coupled_modes_compressed,
            'coupled_singular_values': singular,
            'coupled_cumulative_energy': cumulative_energy
            }
        
        self.__setattr__("decoupling", decoupling)
        
    def cal_csd(self, direction = "x"):
        
        """
        Calculate cross spectral density (csd) and spectral degree of coherence (sdc) along x or y direction.
        
        Parameters:
            direction (str): Direction along which to calculate csd and sdc, 
            "x" for x direction, "y" for y direction. Default is "x".
        
        Returns:
            None
        
        Calculates cross spectral density (csd) and spectral degree of coherence (sdc) along the specified direction.
        If direction is "x", csd and sdc will be calculated along the x direction.
        If direction is "y", csd and sdc will be calculated along the y direction.
        
        Examples:
            cal_csd()  # Calculates csd and sdc along the default direction "x"
            cal_csd(direction="y")  # Calculates csd and sdc along the y direction
        """

        #------------------------------------------------------
        # calcualte csd and sdc along x direction
        
        if direction == "x":
            
            cmode_x = np.zeros((self.n, self.n_column), dtype = np.complex64)
            
            for i in range(self.n):
                cmode_x[i, :] = (
                    self.cmode[i][math.ceil(self.n_row / 2), :] + 
                    self.cmode[i][math.ceil(self.n_row / 2), :]  
                    ) * self.ratio[i] / 2
                
            self.csd2x = np.dot(cmode_x.T.conj(), cmode_x)
            self.csd1x = np.abs(np.diagonal(np.fliplr(self.csd2x)))
            ix2 = np.tile(np.diagonal(np.abs(self.csd2x)), (int(self.xcount), 1))
            
            self.csd2x = self.csd2x / np.abs(self.csd2x).max()
            ix2 = ix2 / ix2.max()
            self.sdc2x = self.csd2x / np.sqrt(ix2 * ix2.T + 1e-9)
            self.sdc1x = np.abs(np.diagonal(np.fliplr(self.sdc2x)))
        
        #------------------------------------------------------
        # calcualte csd and sdc along y direction
        
        elif direction == "y":
            
            cmode_y = np.zeros((self.n_row, self.n), dtype = np.complex64)
            
            for i in range(self.n):
                cmode_y[:, i] = (
                    self.cmode[i][:, math.ceil(self.n_column / 2)] + 
                    self.cmode[i][:, math.ceil(self.n_column / 2)] 
                    ) * self.ratio[i] / 2
                
            self.csd2y = np.dot(cmode_y.conj(), cmode_y.T)
            self.csd1y = np.abs(np.diagonal(np.fliplr(self.csd2y)))
            iy2 = np.tile(np.diag(np.abs(self.csd2y)), (int(self.ycount), 1))
            self.csd2y = self.csd2y / np.abs(self.csd2y).max()
            
            iy2 = iy2 / iy2.max()
            self.sdc2y = self.csd2y / np.sqrt(iy2 * iy2.T + 1e-9)
            self.sdc1y = np.abs(np.diagonal(np.fliplr(self.sdc2y)))
            
        else:
            raise ValueError("Unsupported direction: {}".format(direction))
            
    #------------------------------------------------------
    # calcualte the intensity
    
    def cal_flux(self):
        
        self.cal_intensity()
        self.flux = self.intensity * self.xpixel * self.ypixel
        
    
    def cal_intensity(self):
        
        """
        Calculate the intensity distribution based on the stored mode coefficients and ratios.
        
        Parameters:
            None
        
        Returns:
            None
        
        Calculates the intensity distribution on a grid based on the stored mode coefficients and their corresponding ratios.
        The calculated intensity is stored in the 'intensity' attribute of the object.
        
        Examples:
            cal_intensity()  # Calculates the intensity distribution based on stored mode coefficients and ratios
        """

        self.intensity = np.zeros((self.n_column, self.n_row))
        
        for i in range(self.n):  
            self.intensity = self.intensity + self.ratio[i]**2 * np.abs(self.cmode[i])**2

    #------------------------------------------------------
    # svd process

    def decomposition(self, method = "svd", sparse_n = 0):

        """
        Perform matrix decomposition using Singular Value Decomposition (SVD) 
        or Coherent Mode Decomposition (CMD).
        
        Parameters:
            - method (str): 
                The method used for decomposition, can be "svd" for Singular Value Decomposition
                or "cmd" for Covariance Matrix Diagonalization. Defaults to "svd".
        
        Returns:
            None
        
        Raises:
            ValueError: If an unsupported method is provided.
        """

        #------------------------------------------------------
        # calcualte matrix for svd process
        
        cmodes = np.zeros((self.n, self.xcount * self.ycount), dtype = np.complex64)
        for i in range(self.n):
            cmodes[i, :] = np.reshape(
                self.cmode[i], (self.xcount * self.ycount)
                ) * self.ratio[i]
        cmode_matrix = cmodes.T
        
        #------------------------------------------------------
        # svd process
        
        from scipy import linalg
        
        if method == "svd":
            
            self.n = int(self.n - 2)
            eigen_vector, eigen_value, eigen_evolution = linalg.svd(cmode_matrix, full_matrices = False)
            self.cmode = [
                np.reshape(eigen_vector[:, i], (self.xcount, self.ycount)) 
                for i in range(self.n)
                ]
            self.ratio = [eigen_value[i] for i in range(self.n)]
            self.evolution = eigen_evolution
            
        elif method == "sparse_svd":

            from scipy.sparse.linalg import svds
    
            # self.n = int(self.n - 2)
            if sparse_n == 0: sparse_n = int(self.n/10)
            eigen_vector, eigen_value, eigen_evolution = svds(cmode_matrix, k = sparse_n)
            self.cmode = [
                np.reshape(eigen_vector[:, sparse_n - 1 - i], (self.xcount, self.ycount)) 
                for i in range(sparse_n)
                ]
            self.ratio = [eigen_value[sparse_n - 1 - i] for i in range(sparse_n)]
            self.evolution = np.flipud(eigen_evolution)
                
        #------------------------------------------------------
        # cmd process
        
        elif method == "cmd":
            
            csd_matrix = np.dot(cmodes.T.conj(), cmodes)
            
            eigen_value, eigen_vector = linalg.eigsh(csd_matrix, k = self.n)
            self.cmode = [np.array(np.reshape(
                eigen_vector[:, i], (self.n_row, self.n_column)
                ), dtype = np.complex64) for i in range(self.n)]
            self.ratio = [eigen_value[i] for i in range(self.n)]
            
        else:
            raise ValueError("Unsupported method: {}".format(method))
            
        self.cmode_x = list()
        self.cmode_y = list()
        
        for idx_cmode in self.cmode:
            
            self.cmode_x.append(
                idx_cmode[int(self.xcount//2), :] if self.ycount%2 == 1 else 
                (idx_cmode[int(self.xcount//2), :] + idx_cmode[int(self.xcount//2 - 1), :]) / 2
                )
            self.cmode_y.append(
                idx_cmode[:, int(self.ycount//2)] if self.ycount%2 == 1 else 
                (idx_cmode[:, int(self.ycount//2)] + idx_cmode[:, int(self.ycount//2 - 1)]) / 2
                )
    
    #--------------------------------------------------------------------------
    # the vibration methods
    
    def shift(self, offx = 0, offy = 0):
        
        """
        Shift the complex mode(s) stored in self.cmode by specified offsets in x and y directions.
        
        Parameters:
            offx (float): Offset in the x direction. Default is 0.
            offy (float): Offset in the y direction. Default is 0.
        
        Notes:
            This method shifts the complex mode(s) stored in self.cmode by applying phase shifts in the Fourier domain
            corresponding to the specified offsets offx and offy. The sampling of the wavefront should be large enough
            or some periodic defects will be introduced!
        
        Returns:
            None
        """
        
        offset = [offx / self.xpixel, offy / self.ypixel]

        for idx in range(len(self.cmode)):
            self.cmode[idx] = fourier_shift(self.cmode[idx], offset)


    def tilt(self, rotx = 0, roty = 0, kind = "refrection", degree = False):
        
        """
        Apply tilt to the complex mode(s) stored in self.cmode.
        
        Parameters:
            rotx (float): Tilt angle in the x direction in radians. Default is 0.
            roty (float): Tilt angle in the y direction in radians. Default is 0.
            kind (str): Type of tilt. It can be "source", "refrection", or "reflection". Default is "refrection".
        
        Notes:
            This method applies tilt to the complex mode(s) stored in self.cmode based on the specified tilt angles and kind of tilt.
            Tilt can be applied for different purposes such as source tilt, refraction tilt, or reflection tilt.
        
        Raises:
            ValueError: If kind is not one of "source", "refrection", or "reflection".
        
        Returns:
            None
        """

        k_vector = 2 * np.pi / self.wavelength
        xgrid, ygrid = np.meshgrid(self.ytick, self.xtick)
        
        if degree:
            
            rotx = np.deg2rad(rotx)
            roty = np.deg2rad(roty)
        
        if kind == "source":

            self.shift(
                offx = np.sin(rotx) * self.position, offy = np.sin(roty) * self.position
                )
            rot_phase_x = rotx * xgrid - (1 - np.cos(rotx)) * self.position
            rot_phase_y = roty * ygrid - (1 - np.cos(roty)) * self.position
        
        elif kind == "reflection":
            
            rot_phase_x = np.sin(2 * rotx) * xgrid
            rot_phase_y = np.sin(2 * roty) * ygrid 
            
        elif kind == "refrection":
            
            rot_phase_x = np.sin(rotx) * xgrid
            rot_phase_y = np.sin(roty) * ygrid 
        
        else:
            raise ValueError("kind {%s} should be: source, refrection or reflection" % (kind))

        for idx in range(len(self.cmode)):
            self.cmode[idx] *= (
                np.exp(-1j * k_vector * rot_phase_x) * 
                np.exp(-1j * k_vector * rot_phase_y)
                )
    
    #--------------------------------------------------------------------------
    # the file operation
    
    def save_h5(self):
        
        """
        Save optic plane and coherence parameters to an HDF5 file.
        
        This function saves the optic plane parameters and coherence parameters 
        of the current object to an HDF5 file.
        
        Returns:
            None
        """


        if os.path.isfile(self.name): os.remove(self.name)

        with h5.File(self.name, 'a') as optic_file:
            
            #------------------------------------------------------
            # save optic plane parameters
            
            parameters = optic_file.create_group("optic_plane")
            parameters_list = [
                "xstart", "xend", "xcount", "ystart", "yend", "ycount", 
                "wavelength", "position"
                ]
            for parameter in parameters_list:
                parameters.create_dataset(parameter, data = getattr(self, parameter))
                
            try:
                parameters.create_dataset('n_vector', data = self.n_vector)
            except:
                parameters.create_dataset('n', data = self.n)
            
            #------------------------------------------------------
            # save coherence parameters
            
            coherence = optic_file.create_group("coherence")
            
            if self.dim == 2:
                coherence.create_dataset('coherent_mode', data = np.array(self.cmode))
            elif self.dim == 1:
                coherence.create_dataset('coherent_modex', data = np.array(self.cmode_x))
                coherence.create_dataset('coherent_modey', data = np.array(self.cmode_y))

            parameters_list = [
                "ratio", "evolution", "csd2x", "csd2y", "csd1x", "csd1y", "sdc2x", 
                "sdc2y", "sdc1x", "sdc1y"
                ]
            for parameter in parameters_list:
                coherence.create_dataset(
                    parameter, data = getattr(self, parameter)
                    )
