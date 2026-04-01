#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Fri Mar 22 18:02:22 2024"
__email__    = "xuhan@ihep.ac.cn"


"""
Description: the wave_optics propgation method
"""

#-----------------------------------------------------------------------------#
# library

import sys
import os
import time

import numpy as np

from scipy import fft
from scipy import interpolate
from copy import deepcopy
from numpy import matlib 

#-----------------------------------------------------------------------------#
# constant

bar = "▋" 

#-----------------------------------------------------------------------------#
# function

#-----------------------------------------------------------------
# function - fft shift functions

# FFT pixel shift - scipy.fft. (np.fft is different)
# for odd number: 0 pixel shift. for even number: 1 pixel shift.
# fresnel dfft: (1 odd, 0 even); asm sfft: (1 odd, 1.5 even)
# bluestein fft: (1 odd, power2 is required)

def shift_pixel_geometry(optic_class, n, direction = "x"):
    
    if direction == "x":
        for attribute in ["xstart", "xend", "xtick"]:
            setattr(
                optic_class, attribute, 
                getattr(optic_class, attribute) + n * optic_class.xpixel
                )
    elif direction == "y":
        for attribute in ["ystart", "yend","ytick"]:
            setattr(
                optic_class, attribute, 
                getattr(optic_class, attribute) + n * optic_class.ypixel
                )
    optic_class.xgrid, optic_class.ygrid = np.meshgrid(
        optic_class.xtick, optic_class.ytick
        )
    
    return optic_class

def shift_impluse_geometry(impluse_q):
    
    from skimage.restoration import unwrap_phase
    
    unwraped_phase = unwrap_phase(np.angle(impluse_q))
    
    ycount, xcount = impluse_q.shape
    x_range = np.arange(ycount)
    y_range = np.arange(xcount)
    counts = int(impluse_q.shape[0] * impluse_q.shape[1])
    
    pts = np.meshgrid(x_range + 0.5, y_range + 0.5, indexing = 'xy')
    pts = (
        np.reshape(pts[0], (1, counts)), 
        np.reshape(pts[1], (1, counts))
        )
    
    fabs = interpolate.RegularGridInterpolator(
        (x_range, y_range), np.abs(impluse_q), method = "linear", bounds_error = False, 
        fill_value = 0
        )
    fangle = interpolate.RegularGridInterpolator(
        (x_range, y_range), unwraped_phase, method = "linear", 
        bounds_error = False, fill_value = 0
        )
    return np.rot90(np.reshape(
        fabs(pts) * np.exp(1j * fangle(pts)), 
        (impluse_q.shape[0], impluse_q.shape[1])
        ))
        
#-----------------------------------------------------------------
# function - propagation functions
        
#------------------------------------------------------
# fresnel propagate function

def fresnel(start_optic, end_optic):
    
    propagate_class = _propagator(start_optic, end_optic)
    
    if propagate_class.distance == 0:
        for idx in range(start_optic.n):
            end_optic.cmode[idx] *= start_optic.cmode[idx]
    else:
        end_optic_cmode_list = propagate_class._fresnel_dfft()
        for idx in range(start_optic.n):
            end_optic.cmode[idx] *= end_optic_cmode_list[idx]

# #------------------------------------------------------
# # fresnel propagate function

# def fresnel_slit(start_optic, end_optic, xcoor, ycoor):
    
#     propagate_class = _propagator(start_optic, end_optic)
    
#     if propagate_class.distance == 0:
#         for idx in range(start_optic.n):
#             end_optic.cmode[idx] *= start_optic.cmode[idx]
#     else:
#         end_optic_cmode_list = propagate_class._fresnel_slit(xcoor, ycoor)
#         for idx in range(start_optic.n):
#             end_optic.cmode[idx] *= end_optic_cmode_list[idx]
            
#------------------------------------------------------
# angular spectrum propagate function

def asm(start_optic, end_optic):
    
    propagate_class = _propagator(start_optic, end_optic)
    
    if propagate_class.distance == 0:
        end_optic.cmode *= deepcopy(start_optic.cmode)
        
    else:
        end_optic_cmode_list = propagate_class._asm_sfft()
        for idx in range(start_optic.n):
            end_optic.cmode[idx] *= end_optic_cmode_list[idx]
        
#------------------------------------------------------
# chirp-z transform propagate function

def czt(start_optic, end_optic):
    
    propagate_class = _propagator(start_optic, end_optic)
    
    if propagate_class.distance == 0:
        end_optic.cmode *= deepcopy(start_optic.cmode)

    else:
        end_optic_cmode_list = propagate_class._bluestein_fft()
        for idx in range(start_optic.n):
            end_optic.cmode[idx] *= end_optic_cmode_list[idx]
        
        shift_pixel_geometry(end_optic, 1.0, direction = "x")
        shift_pixel_geometry(end_optic, 1.0, direction = "y")

#------------------------------------------------------
# fractional fresnel transform propagate function

def frfft(start_optic, end_optic):
    
    propagate_class = _propagator(start_optic, end_optic)
    
    if propagate_class.distance == 0:
        end_optic.cmode *= deepcopy(start_optic.cmode)

    else:
        end_optic_cmode_list = propagate_class._fractional_fft()
        
        # print(propagate_class.ax, flush = True)
        
        attribute_x = ["start", "end", "count", "tick", "pixel"]
        for name in attribute_x: 
            for direction in ["x", "y"]:
                setattr(
                    end_optic, direction + name, 
                    getattr(propagate_class, "a" + direction) * getattr(end_optic, direction + name)
                    )
        end_optic.xgrid, end_optic.ygrid = np.meshgrid(end_optic.xtick, end_optic.ytick)

        for idx in range(start_optic.n):
            end_optic.cmode[idx] *= end_optic_cmode_list[idx]
        
        shift_pixel_geometry(end_optic, 1.0, direction = "x")
        shift_pixel_geometry(end_optic, 1.0, direction = "y")
        
#-----------------------------------------------------------------
# function - propagate beamline

def processing_bar(current_count, total_count, interval_time, start_time):
    
    print("\r", end = "")
    print(
        "propagate processs: {}%: ".format(int(100*(current_count + 1) / total_count)), 
        "▋" * int(1 + 10*current_count / total_count), end = ""
        )
    print("  time cost: %.2f min" % (np.abs(interval_time - start_time)/60), end = "")
    
    sys.stdout.flush()
    time.sleep(0.005)
    
#------------------------------------------------------
# propagate different coherent modes

def propagate_mode(n, beamline_func):
    
    start_time = time.time()
    
    for idx in range(n):
        
        interval_time = time.time()
        processing_bar(idx, n, interval_time, start_time)
        
        if idx == 0:
            final_optic = beamline_func(idx)
        else:
            optic_cache = beamline_func(idx)
            final_optic.cmode.append(optic_cache.cmode[0])
    
    final_optic.n = n
    
    return final_optic

#------------------------------------------------------
# propagate to different location

def propagate_depth(depth_tick, beamline_func):
    
    start_time = time.time()
    
    for idx, position in enumerate(depth_tick):
        
        interval_time = time.time()
        processing_bar(idx, depth_tick.shape[0], interval_time, start_time)
        
        if idx == 0:
            final_optic = beamline_func(position)
        else:
            optic_cache = beamline_func(position)
            final_optic.cmode.append(optic_cache.cmode[0])
    
    return final_optic

#-----------------------------------------------------------------
# function - one dimensional propagate 

def fresnel_1d(start_optic, end_optic):
    
    propagate_class = _propagator(start_optic, end_optic)
    
    if propagate_class.distance == 0:
        for idx in range(start_optic.n):
            end_optic.cmode_x[idx] *= start_optic.cmode_x[idx]
            end_optic.cmode_y[idx] *= start_optic.cmode_y[idx]
    else:
        end_optic_cmode_list = propagate_class._fresnel_fft1d()
        for idx in range(start_optic.n):
            end_optic.cmode_x[idx] *= end_optic_cmode_list[0][idx]
            end_optic.cmode_y[idx] *= end_optic_cmode_list[1][idx]
            
def asm_1d(start_optic, end_optic):
    
    propagate_class = _propagator(start_optic, end_optic)
    
    if propagate_class.distance == 0:
        for idx in range(start_optic.n):
            end_optic.cmode_x[idx] *= start_optic.cmode_x[idx]
            end_optic.cmode_y[idx] *= start_optic.cmode_y[idx]
    else:
        end_optic_cmode_list = propagate_class._asm_fft1d()
        for idx in range(start_optic.n):
            end_optic.cmode_x[idx] *= end_optic_cmode_list[0][idx]
            end_optic.cmode_y[idx] *= end_optic_cmode_list[1][idx]

def propagate_mode_1d(n, beamline_func):
    
    start_time = time.time()
    
    for idx in range(n):
        
        interval_time = time.time()
        processing_bar(idx, n, interval_time, start_time)
        
        if idx == 0:
            final_optic = beamline_func(idx)
        else:
            optic_cache = beamline_func(idx)
            
            final_optic.cmode_x.append(optic_cache.cmode_x[0])
            final_optic.cmode_y.append(optic_cache.cmode_y[0])
    
    final_optic.n = n
    
    return final_optic

def propagate_depth_1d(depth_tick, beamline_func):
    
    start_time = time.time()
    
    for idx, position in enumerate(depth_tick):
        
        interval_time = time.time()
        processing_bar(idx, depth_tick.shape[0], interval_time, start_time)
        
        if idx == 0:
            final_optic = beamline_func(position)
        else:
            optic_cache = beamline_func(position)
            final_optic.cmode_x.append(optic_cache.cmode_x[0])
            final_optic.cmode_y.append(optic_cache.cmode_y[0])
    
    return final_optic

#--------------------------------------------------------------------------
# propagate to different vibration

#--------------------------------------------------------------------------
# propagate to different parameters

#------------------------------------------------------------------------------
# class

class _propagator(object):

    def __init__(self, start_optic, end_optic):
        
        self.so = start_optic
        self.eo = end_optic
        self.distance = self.eo.position - self.so.position
        
        power_check = lambda number: (np.log(number) / np.log(2)).is_integer()
        count_list = [self.so.xcount, self.so.ycount, self.eo.xcount, self.eo.ycount]
        self.power_flag = all([power_check(number) for number in count_list])
        
        self.equal_flag = all((
            getattr(self.so, name) == getattr(self.eo, name) 
            for name in ["xstart", "xend", "xcount", "ystart", "yend", "ycount"]
            ))
        self.even_flag = all(
            (getattr(self.so, name)%2 == 0 and getattr(self.eo, name)%2 == 0 
             for name in ["xcount", "ycount"])
            )
    
    #------------------------------------------------------
    # fresnel tfft propagator 
    
    def shift_impluse_geometry(self, impluse_q):
        
        from skimage.restoration import unwrap_phase
      
        x_range = np.arange(self.so.ycount)
        y_range = np.arange(self.so.xcount)

        points = np.meshgrid(x_range + 0.5, y_range - 0.5, indexing = 'xy')
        points = (
            np.reshape(points[0], (1, int(self.so.ycount * self.so.ycount))), 
            np.reshape(points[1], (1, int(self.so.ycount * self.so.ycount)))
            )
        
        func_abs = interpolate.RegularGridInterpolator(
            (x_range, y_range), np.abs(impluse_q), method = "linear", bounds_error = False, 
            fill_value = 0
            )
        func_ang = interpolate.RegularGridInterpolator(
            (x_range, y_range), unwrap_phase(np.angle(impluse_q)), 
            method = "linear", bounds_error = False, fill_value = 0
            )
        
        shifted_impulse_q = func_abs(points) * np.exp(1j * func_ang(points))
        shifted_impulse_q = np.rot90(np.reshape(
            shifted_impulse_q, (self.so.xcount, self.so.ycount)
            ))
        
        return shifted_impulse_q 
    
    def _fresnel_dfft(self):
        
        # check optic parameters
        
        if not self.equal_flag: 
            raise ValueError("The goemetry between two optic plane should be equal!")
            
        elif not self.even_flag:
            raise ValueError("The xcount and ycount shoud be even number!")
        
        # perform double fft fresnel transform
        
        else:
            
            # the grid of the frequency
            
            qx_tick = np.linspace(0.25/self.so.xstart, 0.25/self.so.xend, self.so.xcount) * self.so.xcount
            qy_tick = np.linspace(0.25/self.so.ystart, 0.25/self.so.xend, self.so.ycount) * self.so.ycount
            qx_grid, qy_grid = np.meshgrid(qx_tick, qy_tick)

            # propagation function
            
            impulse_q = np.array(np.exp(
                (-1j * 2*np.pi / self.so.wavelength * self.distance) * 
                (1 - self.so.wavelength**2 * (qx_grid**2 + qy_grid**2) / 2)
                ), dtype = np.complex64)
            impulse_q = np.array(
                self.shift_impluse_geometry(impulse_q), dtype = np.complex64
                )

            eo_cmode = list()
            for idx in range(self.so.n):
                eo_cmode.append(
                    np.array(fft.ifft2(
                        fft.fft2(self.so.cmode[idx]) * fft.ifftshift(impulse_q)
                        ), dtype = np.complex64)
                    )

        return eo_cmode
    
    # # def _fresnel_slit(self, xcoor, ycoor):
        
    #     #---------------------------------------------------
    #     # the calculation of slit diffraction
        
    #     # constant calculation
        
    #     fresnel_number_x = 0.25 * (xcoor[1] - xcoor[0])**2 / (self.so.wavelength * self.distance)
    #     fresnel_number_y = 0.25 * (ycoor[1] - ycoor[0])**2 / (self.so.wavelength * self.distance)
    #     norm_x = self.so.xtick / np.sqrt(self.so.wavelength * self.distance)
    #     norm_y = self.so.ytick / np.sqrt(self.so.wavelength * self.distance)
    #     norm_x, norm_y = np.meshgrid(norm_x, norm_y)
        
    #     # fresnel integral parameters
        
    #     def fresnel_integral(norm, fresnel_number):
            
    #         parameter1 = 2**0.5 * (fresnel_number**0.5 - norm)
    #         parameter2 = -2**0.5 * (norm + fresnel_number**0.5)
            
    #         from scipy import special
            
    #         sin_parameter1, cos_parameter1 = special.fresnel(parameter1)
    #         sin_parameter2, cos_parameter2 = special.fresnel(parameter2)
    #         integral = (cos_parameter2 - cos_parameter1) + 1j * (sin_parameter2 - sin_parameter1)
            
    #         return integral
        
    #     slit_integral = (
    #         (np.exp(1j * 20 * 2 * np.pi / self.so.wavelength) / (2j)) *
    #         fresnel_integral(norm_x, fresnel_number_x) *
    #         fresnel_integral(norm_y, fresnel_number_y)
    #         )
        
    #     #---------------------------------------------------
    #     # the calculation of the wavefront propagation
        
    #     qx_tick = np.linspace(0.25/self.so.xstart, 0.25/self.so.xend, self.so.xcount) * self.so.xcount
    #     qy_tick = np.linspace(0.25/self.so.ystart, 0.25/self.so.xend, self.so.ycount) * self.so.ycount
    #     qx_grid, qy_grid = np.meshgrid(qx_tick, qy_tick)

    #     # propagation function
        
    #     impulse_q = np.exp(
    #         (-1j * 2*np.pi / self.so.wavelength * self.distance) * 
    #         (1 - self.so.wavelength**2 * (qx_grid**2 + qy_grid**2) / 2)
    #         )
    #     impulse_q = self.shift_impluse_geometry(impulse_q)

    #     eo_cmode = list()
    #     for idx in range(self.so.n):
            
    #         fresnel_wavefront = fft.ifft2(
    #             fft.fft2(self.so.cmode[idx]) * fft.ifftshift(impulse_q)
    #             )
    #         eo_cmode.append(
    #             np.fft.ifftshift(np.fft.ifft2(fft.fft2(fresnel_wavefront) * fft.fft2(slit_integral)))
    #             )
    #         # eo_cmode.append(slit_integral)
            
    #     return eo_cmode
    
    #------------------------------------------------------
    # angular spectrum fft propagator 

    def _asm_sfft(self):
        
        # check optic parameters
        
        if not self.equal_flag: 
            raise ValueError("The goemetry between two optic plane should be euqal!")
            
        elif not self.even_flag:
            raise ValueError("The xcount and ycount shoud be even number!")
            
        # perform asm fft fresnel transform
        
        else:
            
            qx_tick = np.linspace(-1 / (2 * self.so.xpixel), 1 / (2 * self.so.xpixel), self.so.xcount)
            qy_tick = np.linspace(-1 / (2 * self.so.ypixel), 1 / (2 * self.so.ypixel), self.so.ycount)
            qx_grid, qy_grid = np.meshgrid(qx_tick, qy_tick)
            
            impulse = np.array(np.exp(
                -1j * 2*np.pi * self.distance * 
                np.sqrt(1 / self.so.wavelength**2 - (qx_grid**2 + qy_grid**2))
                ), dtype = np.complex64)
            impulse = np.array(
                self.shift_impluse_geometry(impulse),
                dtype = np.complex64
                )
            
            eo_cmode = list()
            for idx in range(self.so.n):
                eo_cmode.append(np.array(fft.ifft2(
                    fft.ifftshift(impulse * fft.ifftshift(fft.fft2(self.so.cmode[idx])))
                    ), dtype = np.complex64))
            
        return eo_cmode
    
    def _bluestein_fft(self):
        
        if not self.power_flag:
            raise ValueError("The size of the geometry should be the power of 2!")
        
        else:
            
            #---------------------------------------------------
            # bluestein fft method

            def _bluestein_fft_1d(bluestein_input, qstart, count, start, end, distance):
                
                
                vcount, hcount = np.shape(bluestein_input)
                
                #---------------------------------------------------
                # the phase calculation for bluestein fft
                
                start_index = start + qstart + 1/2 * (end - start) / count
                step_phase = np.exp(1j * 2*np.pi * (end - start) / (count * qstart))
                
                start_phase_neg_n = list()
                step_phase_n2 = list()
                for idx in range(vcount):
                    start_phase_neg_n.append(np.exp(-1j * 2*np.pi * start_index / qstart)**(-1 * idx))
                    step_phase_n2.append(np.exp(1j * 2*np.pi * (end - start) / (count * qstart))**(idx**2/2))
                step_phase_n2 = np.array(step_phase_n2) * np.array(start_phase_neg_n)
                
                step_phase_k2 = np.array([step_phase**(idx**2/2) for idx in range(count)])
                step_phase_pos_nk2 = np.array(
                    [step_phase**(idx**2/2) 
                     for idx in range(-vcount + 1, max(vcount, count))]
                    )
                step_phase_neg_nk2 = step_phase_pos_nk2**(-1)
                
                #---------------------------------------------------
                # bluestein fft count
                
                bluestein_fft_count = vcount + count
                power_n = 0
                while bluestein_fft_count <= vcount + count:
                    bluestein_fft_count = 2**power_n
                    power_n += 1
                
                #---------------------------------------------------
                # convolution of bluestein fft
                
                conv_phase_n2 = np.repeat(step_phase_n2[:, np.newaxis], hcount, axis = 1)
                conv_phase_neg_nk2 = np.repeat(step_phase_neg_nk2[:, np.newaxis], hcount, axis = 1)
                conved = (
                    fft.fft(bluestein_input * conv_phase_n2, bluestein_fft_count, axis = 0) * 
                    fft.fft(conv_phase_neg_nk2, bluestein_fft_count, axis = 0)
                    )
                
                # TODO: The flux was disturbed by convolution.
                
                bluestein_output = fft.ifft(conved, axis = 0)
                bluestein_output = (
                    bluestein_output[vcount : vcount + count, :] * 
                    np.repeat(step_phase_k2[:, np.newaxis], hcount, axis = 1)
                    )
                
                shift_distance = (end - start) * np.linspace(0, count - 1, count) / count + start_index 
                shift_phase = matlib.repmat(np.exp(
                    1j * 2*np.pi * shift_distance * (-vcount / 2 + 1/2) / qstart
                    ), hcount, 1)
                bluestein_output = bluestein_output.T * shift_phase 

                return bluestein_output

            #---------------------------------------------------
            # set plane geometry structure

            k_vector = 2 * np.pi / self.so.wavelength
            
            fresnel_so = (
                np.exp(1j * k_vector * self.distance) * 
                np.exp(-0.5 * 1j * k_vector * (self.so.xgrid**2 + self.so.ygrid**2) / self.distance)
                )
            fresnel_eo = np.exp(
                -0.5 * 1j * k_vector * (self.eo.xgrid**2 + self.eo.ygrid**2) / 
                self.distance
                )
            
            eo_cmode = list()
            
            for idx in range(self.so.n):
                
                bluestein_input = self.so.cmode[idx] * fresnel_so
                qy_pixel = self.so.wavelength * self.distance / self.so.ypixel
                qx_pixel = self.so.wavelength * self.distance / self.so.xpixel
                
                bluestein_input = _bluestein_fft_1d(
                    bluestein_input, qy_pixel, self.eo.ycount, self.eo.ystart, 
                    self.eo.yend, self.distance
                    )
                bluestein_input = _bluestein_fft_1d(
                    bluestein_input, qx_pixel, self.eo.xcount, self.eo.xstart, 
                    self.eo.xend, self.distance
                    )
                
                bluestein_output = bluestein_input * fresnel_eo
                normalizaton = (
                    np.sum(np.abs(self.so.cmode[idx])**2 * self.so.xpixel * self.so.ypixel) / 
                    np.sum(np.abs(bluestein_output)**2 * self.eo.xpixel * self.eo.ypixel)
                    )
                eo_cmode.append(bluestein_output * normalizaton)
                
            return eo_cmode
        
    def _fresnel_fft1d(self):
        
        # check optic parameters
        
        if not self.equal_flag: 
            raise ValueError("The goemetry between two optic plane should be euqal!")
            
        # perform double fft fresnel transform
        
        else:
            
            # the grid of the frequency
            
            qx_tick = np.linspace(0.25/self.so.xstart, 0.25/self.so.xend, self.so.xcount) * self.so.xcount
            qy_tick = np.linspace(0.25/self.so.ystart, 0.25/self.so.yend, self.so.ycount) * self.so.ycount
            
            # propagation function
            
            impulse_qx = np.exp(
                (-1j * 2*np.pi / self.so.wavelength * self.distance) * 
                (1 - self.so.wavelength**2 * qx_tick**2 / 2)
                )
            impulse_qy = np.exp(
                (-1j * 2*np.pi / self.so.wavelength * self.distance) * 
                (1 - self.so.wavelength**2 * qy_tick**2 / 2)
                )
            
            eo_cmode_x, eo_cmode_y = [list(), list()]
            
            for idx in range(self.so.n):
                
                eo_cmode_x.append(
                    fft.ifft(fft.fft(self.so.cmode_x[idx]) * fft.ifftshift(impulse_qx))
                    )
                eo_cmode_y.append(
                    fft.ifft(fft.fft(self.so.cmode_y[idx]) * fft.ifftshift(impulse_qy))
                    )
        
        return [eo_cmode_x, eo_cmode_y]
    
    # def _fresnel_fft1d(self):
        
    #     if not self.equal_flag:
    #         raise ValueError("The geometry between two optic plane should be equal!")

    #     k_vector = 2 * np.pi / self.so.wavelength
    
    #     def _propagate_1d(wavefront, pixel, count):
            
    #             freq = fft.fftfreq(count, d = pixel)
    #             impulse_func = np.exp(1j * k_vector * self.distance) * np.exp(
    #                 -1j * np.pi * self.so.wavelength * 
    #                 self.distance * freq**2
    #                 )
    #             return fft.ifft(fft.fft(wavefront) * impulse_func)
    
    #     eo_cmode_x = list()
    #     eo_cmode_y = list()
    
    #     for idx in range(self.so.n):
    #         eo_cmode_x.append(_propagate_1d(
    #             self.so.cmode_x[idx], self.so.xpixel, self.so.xcount
    #             ))
    #         eo_cmode_y.append(_propagate_1d(
    #             self.so.cmode_y[idx], self.so.ypixel, self.so.ycount
    #             ))
    
    #     return [eo_cmode_x, eo_cmode_y]
    
    def _asm_fft1d(self):
        
        if not self.equal_flag:
            raise ValueError("The geometry between two optic plane should be equal!")
    
        wavelength = self.so.wavelength
        k_vector = 2 * np.pi / wavelength
    
        def _propagate_1d(wavefront, pixel, count):
            
            freq = fft.fftfreq(count, d = pixel)
            kz = np.sqrt(np.array(
                k_vector**2 - (2 * np.pi * freq)**2, dtype = np.complex128
                ))
            impulse_func = np.exp(1j * kz * self.distance)
            
            return fft.ifft(fft.fft(wavefront) * impulse_func)
    
        eo_cmode_x = list()
        eo_cmode_y = list()
    
        for idx in range(self.so.n):
            eo_cmode_x.append(_propagate_1d(
                self.so.cmode_x[idx], self.so.xpixel, self.so.xcount
                ))
            eo_cmode_y.append(_propagate_1d(
                self.so.cmode_y[idx], self.so.ypixel, self.so.ycount
                ))
    
        return [eo_cmode_x, eo_cmode_y]
        