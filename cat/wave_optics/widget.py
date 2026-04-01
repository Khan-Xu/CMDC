#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Thu Mar 21 18:50:03 2024"
__email__    = "xuhan@ihep.ac.cn"


"""
Description
"""

#-----------------------------------------------------------------------------#
# modules

import sys
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import h5py as  h5

from scipy import interpolate

#-----------------------------------------------------------------------------#
# parameters

from matplotlib.colors import LinearSegmentedColormap

temperture_cmap = LinearSegmentedColormap(
    'temperture',
    {'red': ((0.00, 0.00, 0.00), (0.50, 0.00, 0.00), (0.75, 1.00, 1.00), (1.00, 1.00, 1.00)),
     'green': ((0.00, 0.00, 0.00), (0.25, 1.00, 1.00), (0.75, 1.00, 1.00), (1.00, 0.00, 0.00)),
     'blue': ((0.00, 1.00, 1.00), (0.25, 1.00, 1.00), (0.50, 0.00, 0.00), (1.00, 0.00, 0.00))
     }, 256
    )

# the radius of Be crl

radius_crl = [1e-3, 0.5e-3, 0.2e-3, 0.1e-3]

# the delta of Be

Be_delta = [[01.000, 3.36218684e-4], [01.995, 8.66195696e-5],
            [03.019, 3.75788877e-5], [03.981, 2.15719519e-5],
            [05.011, 1.35925748e-5], [05.888, 9.84020153e-6],
            [07.079, 6.80375706e-6], [07.962, 5.37693040e-6],
            [10.102, 3.33848720e-6], [12.390, 2.21914661e-6],
            [14.195, 1.69029488e-6], [17.409, 1.12369605e-6],
            [20.636, 7.99653378e-7], [23.644, 6.09130609e-7],
            [30.000, 3.78344765e-7]]

# label

micrometer_label_unit = '$\u03bc$m' 

#-----------------------------------------------------------------------------#
# functions


#-------------------------------------------------------
# print the h5 file structure

def h5print(file_name):
    
    "source: https://www.cnblogs.com/osnosn/"
    
    def h5list(f, tab):
        
        print(tab, 'Group: ', f.name, 'len: %d' % len(f))
        mysp2 = tab[:-1] + '  |-*'
        
        for vv in f.attrs.keys(): 
            print(mysp2, end = ' ')
            print('%s = %s' % (vv,f.attrs[vv]))
        
        mysp = tab[:-1] + '  |-'
        
        for k in f.keys():
            d = f[k]
            
            if isinstance(d, h5.Group):
                h5list(d, mysp)
            
            elif isinstance(d, h5.Dataset):
              
                print(mysp, 'Dataset: ', d.name, '(size: %d)' % d.size)
                mysp1 = mysp[:-1] + '  |-'
                print(mysp1, '(dtype = %s)' % d.dtype)
                
                if d.dtype.names is not None:
                    print(mysp, end = ' ')
                    for vv in d.dtype.names:
                        print(vv, end = ',')
                        
                    print()
                
                mysp2 = mysp1[:-1]+ '  |-*'
                
                for vv in d.attrs.keys(): 
                    print(mysp2, end = ' ')
                   
                    try:
                        print('%s = %s' % (vv, d.attrs[vv]))
                    except TypeError as e:
                        print('%s = %s' % (vv, e))
                    except:
                        print('%s = ?? Other ERR' % (vv,))
                        
            else:
                print('??->', d, 'Unkown Object!')
        
    with h5.File(file_name, 'r') as f: h5list(f, '')

#-------------------------------------------------------
# calculate delta of Be based on energy

def delta_calculation(delta_or_energy, mode = 'e2d'):
    
    from scipy import optimize
    
    Be_delta_data = np.array(Be_delta)
    
    if mode == 'e2d':
    
        def delta_linear(x, para_0, para_1): return para_0 * x * 1e3 + para_1
        
        parameters, cov_matrix = optimize.curve_fit(
            delta_linear, 
            np.log10(Be_delta_data[:, 0] * 1e3), 
            np.log10(Be_delta_data[:, 1])
            )
    
        return 10**delta_linear(np.log10(delta_or_energy*1e3), *parameters)
    
    elif mode == 'd2e':
        
        def energy_linear(x, para_0, para_1): return para_0 * x + para_1
        
        parameters, cov_matrix = optimize.curve_fit(
            energy_linear,
            np.log10(Be_delta_data[:, 1]),
            np.log10(Be_delta_data[:, 0] * 1e3) 
            )  
        
        return 10**energy_linear(np.log10(delta_or_energy), *parameters)

#-------------------------------------------------------
# 2d gaussian fitting

def gaussian_fit(xtick, intensity):
    
    # init the gaussian paramter
    
    peak0 = np.max(intensity)
    center0 = np.sum(xtick * intensity) / np.sum(intensity)
    sigma0 = np.sqrt(np.sum(intensity * (xtick - center0)**2) / np.sum(intensity))
    
    # gauss function
    
    def gauss(x, peak, center, sigma): 
        return  peak * np.exp(-(xtick - center)**2 / (2 * sigma**2))
    
    from scipy.optimize import curve_fit
    
    # the fitting
    
    popt, pcov = curve_fit(
        gauss, xtick, intensity, p0 = [peak0, center0, sigma0]
        )
    
    return popt, gauss(xtick, *popt)

def cal_fwhm(x, intensity_1d):

    above_half_max = intensity_1d >= (
        intensity_1d[np.argmax(intensity_1d)] / 2
        )
    indices = np.where(above_half_max)[0]
    
    if len(indices) == 0:
        raise ValueError("The data can't identify a FWHM")
    else:
        return x[indices[-1]] - x[indices[0]]
    
#-------------------------------------------------------
# complex z-plane visulation

# def vector_complex(complex_2d_dataset):

#     from colorsys import hls_to_rgb
#     from skimage.restoration import unwrap_phase
    
#     amplitude = np.abs(complex_2d_dataset)
#     angle = unwrap_phase(np.angle(complex_2d_dataset))
#     angle /= np.abs(angle).max()
#     amplitude /= np.abs(amplitude).max()
    
#     # vector_dataset = np.array(np.vectorize(hls_to_rgb) (
#     #     (angle + np.pi) / (2 + np.pi) + 0.5, 1 - 1 / (1 + amplitude**0.3), 0.8
#     #     )).swapaxes(0, 2)
    
#     vector_dataset = np.array(np.vectorize(hls_to_rgb) (
#         angle, 1 - 1 / (1 + amplitude), 0.8
#         )).swapaxes(0, 2)
    
#     return vector_dataset

#-----------------------------------------------------------------------------#
# classes

class plot_optic(object):

    def __init__(self, optic_class):

        self.optic = optic_class
        self.extent = [
            np.min(self.optic.xtick * 1e6), np.max(self.optic.xtick * 1e6),
            np.min(self.optic.ytick * 1e6), np.max(self.optic.ytick * 1e6)
            ]
        
        # take intensity data 
        
        self.optic.cal_intensity()
        idx_max = np.where(self.optic.intensity == np.max(self.optic.intensity))
        self.idx_max_y, self.idx_max_x = (
            (idx_max[0][0], idx_max[1][0]) if idx_max[0].shape[0] != 1 else idx_max
            )
        
        self.optic_intensity = self.optic.intensity / self.optic.intensity.max()
        self.intensity_1dy = (
            np.sum(self.optic.intensity, 1) / 
            np.max(np.sum(self.optic.intensity, 1))
            )
        intensity_cuty = np.reshape(
            self.optic.intensity[:, self.idx_max_x], (self.optic.xcount)
            )
        
        self.intensity_1dx = (
            np.sum(self.optic.intensity, 0) / 
            np.max(np.sum(self.optic.intensity, 0))
            )
        intensity_cutx = np.reshape(
            self.optic.intensity[self.idx_max_y, :], (self.optic.ycount)
            )
        
        self.intensity_cuty = intensity_cuty / intensity_cuty.max()
        self.intensity_cutx = intensity_cutx / intensity_cutx.max()
        
    #---------------------------------------------------
    # plot the intensity, 2d and 1d

    def intensity(self, fit = 0, fit_mode = "sigma", mode = 'sum', log_intensity = False):
        
        """
        A function for calculating and plotting intensity data with optional Gaussian fitting.
        
        Parameters:
            - mode (str): The mode of calculation, can be 'sum' or 'section'.
            - fit (int): Whether to perform Gaussian fitting and plot the result. Default is 0.
        
        Returns:
            None
        
        Notes:
            This function constructs a figure with three subplots:
            1. 2D intensity plot.
            2. 1D intensity plot along the y-axis.
            3. 1D intensity plot along the x-axis.
        
        If mode is 'sum':
            - Gaussian fitting is performed on the 1D intensity data along the y and x axes.
        
        If mode is 'section':
            - Gaussian fitting is performed on the sectioned intensity data along the y and x axes.
        
        The Gaussian fitting results are displayed on the corresponding plots with labels and titles.
        
        """

        # a internal function for gaussian fit and plot
        
        def gaussianfit_and_plot(
                fit, fit_mode, xtick, intensity_1d, ax_handle, title0, title1, 
                xlabel, ylabel
                ):
            
            if fit == 1:
                
                ax_handle.scatter(xtick, intensity_1d, s = 8, alpha = 0.5)
                
                if fit_mode == 'sigma':
                    para, curve = gaussian_fit(xtick, intensity_1d)
                    ax_handle.plot(xtick, curve)
                    ax_handle.set_title(title0 % (para[2], 2.35 * para[2]))
                    
                elif fit_mode == "fwhm":
                    
                    fwhm = cal_fwhm(xtick, intensity_1d)
                    ax_handle.set_title(title0 % (0, fwhm))
                    
                ax_handle.set_xlabel(xlabel % (micrometer_label_unit), fontsize = 12)
                ax_handle.set_ylabel(ylabel, fontsize = 12)
                # ax_handle.set_box_aspect(1) 
                
            else: 
                ax_handle.scatter(xtick, intensity_1d, s = 8, alpha = 0.5)
                ax_handle.set_title(title1)
                ax_handle.set_xlabel(xlabel % (micrometer_label_unit), fontsize = 12)
                ax_handle.set_ylabel(ylabel, fontsize = 12)
                # ax_handle.set_box_aspect(1) 
        
        # construct the figure handle
        figure, axes_handle = plt.subplots(1, 3, figsize = (11.5, 3.5))
        
        # 2d figure
        
        if log_intensity:
            intensity_2d = axes_handle[0].imshow(
                np.log(self.optic_intensity + 1e-9), extent = self.extent,
                cmap = 'coolwarm',
                )
        else:
            intensity_2d = axes_handle[0].imshow(
                self.optic_intensity, extent = self.extent, cmap = 'coolwarm',
                )
            
        axes_handle[0].set_title('Intenisty')
        axes_handle[0].set_xlabel('x (%s)' % (micrometer_label_unit), fontsize = 12)
        axes_handle[0].set_ylabel('y (%s)' % (micrometer_label_unit), fontsize = 12)
        axes_handle[0].set_box_aspect(1) 
        figure.colorbar(intensity_2d, ax = axes_handle[0])
        
        # 1d figure
        if mode == 'sum':
            gaussianfit_and_plot(
                fit, fit_mode, self.optic.xtick * 1e6, self.intensity_1dy, 
                axes_handle[1], "intensity_y \n $\u03B4$ = %.3f um   FWHM = %.3f um", 
                "Intensity_y", 'y (%s)', 'Intensity (a. u.)'
                )
            gaussianfit_and_plot(
                fit, fit_mode, self.optic.ytick * 1e6, self.intensity_1dx, 
                axes_handle[2], 'intensity_x \n $\u03B4$ = %.3f um   FWHM = %.3f um', 
                "Intensity_x", 'x (%s)', 'Intensity (a. u.)'
                )
            
        elif mode == 'section':
            gaussianfit_and_plot(
                fit, fit_mode, self.optic.xtick * 1e6, self.intensity_cuty, 
                axes_handle[1], 'intensity_y \n $\u03B4$ = %.3f um  FWHM = %.3f um', 
                "Intensity_y", 'y (%s)', 'Intensity (a. u.)'
                )
            gaussianfit_and_plot(
                fit, fit_mode, self.optic.ytick * 1e6, self.intensity_cutx, 
                axes_handle[2], 'intensity_x \n $\u03B4$ = %.3f um  FWHM = %.3f um', 
                "Intensity_x", 'x (%s)', 'Intensity (a. u.)'
                )
        figure.tight_layout()
        
    #---------------------------------------------------
    # plot the intensity, 1d

    def i1d(self):
        
        """
        Plot 1D intensity data along the x and y axes.
        
        Notes:
            This function creates a 4x4 figure and plots the 1D intensity data along the x and y axes.
            The x and y axis labels are set accordingly, 
            and the plot represents the intensity in arbitrary units.
            
        """

        plt.figure(figsize = (4, 4))
        
        plt.plot(self.optic.xtick * 1e6, self.intensity_1dy)
        plt.plot(self.optic.ytick * 1e6, self.intensity_1dx)
        
        plt.xlabel('x / y %s' % (micrometer_label_unit), fontsize = 12)
        plt.ylabel('Intensity (a. u.)', fontsize = 12)
        
        plt.tight_layout()
        
    #---------------------------------------------------
    # plot the coherent modes

    def cmode(self, count = (3, 3)):
        
        """
        Plot coherent mode intensity and phase.
        
        Parameters:
            - count: Tuple, optional, default value is (3, 3), 
                     a tuple specifying the number of rows and columns for subplots.
        
        Returns:
        None
        
        Notes:
        This function creates subplots to display the coherent mode intensity and phase.
        It calculates and displays the intensity and phase of each coherent mode in the specified 
        subplot arrangement.
        The x and y axis labels are set accordingly.
        """

        #---------------------------------------------------
        # coherent mode intensity
        
        figure, axes_handle = plt.subplots(
            int(count[0]), int(count[1]), figsize = (4 * int(count[1]), 4 * int(count[0]))
            )
        cmode_index = 0

        for idx_0 in range(int(count[0])):
            for idx_1 in range(int(count[1])):

                if int(count[0]) == 1: ax_handle_i = axes_handle[idx_1]
                else: ax_handle_i = axes_handle[idx_0, idx_1]
                
                intensity_cmode = (
                    np.abs(self.optic.cmode[cmode_index])**2 / 
                    np.max(np.abs(self.optic.cmode[cmode_index])**2)
                    )
                cmode_i = ax_handle_i.imshow(intensity_cmode, extent = self.extent, cmap = 'coolwarm')
                
                ax_handle_i.set_title("index: %d" % (cmode_index))
                ax_handle_i.set_xlabel('x (%s)' % (micrometer_label_unit), fontsize = 12)
                ax_handle_i.set_ylabel('y (%s)' % (micrometer_label_unit), fontsize = 12)
                
                cmode_index += 1
        
        figure.tight_layout()
        
        #---------------------------------------------------
        # plot coherent mode phase
        
        figure, axes_handle = plt.subplots(
            int(count[0]), int(count[1]), figsize = (4 * int(count[1]), 4 * int(count[0]))
            )
        cmode_index = 0

        for idx_0 in range(int(count[0])):
            for idx_1 in range(int(count[1])):

                if int(count[0]) == 1: ax_handle_i = axes_handle[idx_1]
                else: ax_handle_i = axes_handle[idx_0, idx_1]
                
                intensity_cmode = (
                    np.angle(self.optic.cmode[cmode_index])**2 / 
                    np.max(np.abs(self.optic.cmode[cmode_index])**2)
                    )
                cmode_i = ax_handle_i.imshow(intensity_cmode, extent = self.extent)
                
                ax_handle_i.set_title("index: %d" % (cmode_index))
                ax_handle_i.set_xlabel('x (%s)' % (micrometer_label_unit), fontsize = 12)
                ax_handle_i.set_ylabel('y (%s)' % (micrometer_label_unit), fontsize = 12)
                
                cmode_index += 1
        
        figure.tight_layout()
    
    #---------------------------------------------------
    # plot the ratio
    
    def ratio(self, n = None):
        
        """
        Plot the normalized coherent mode ratio.
        
        Parameters:
        - n: int, optional, default value is None, 
             the number of coherent modes to consider for the ratio calculation.
        
        Returns:
        None
        
        Notes:
        This function calculates and plots the normalized coherent mode ratio.
        It considers the specified number of coherent modes (or all if not specified), 
        and displays the ratio in a scatter plot.
        The x axis represents the coherent mode number, and the y axis represents the normalized ratio.
        
        """

        count = int(self.optic.n) if n == None else int(n)
        
        normalized_coherent_mode_ratio = (
            np.array(self.optic.ratio[0 : count])**2 / np.sum(np.array(self.optic.ratio[0 : count])**2)
            )
            
        plt.figure(figsize = (4, 4))        
        plt.scatter(range(count), normalized_coherent_mode_ratio)
        plt.plot(range(count), normalized_coherent_mode_ratio)
        print(normalized_coherent_mode_ratio, flush = True)
        
        # write label and title
        
        plt.title("coherent mode ratio")
        plt.xlabel('coherent mode number (n)', fontsize = 12)
        plt.ylabel('ratio (normalized)', fontsize = 12)
        
        plt.tight_layout()
        
    def global_doc(self, n = None):
        
        """
        Plot the normalized coherent mode global cohernet ratio.
        
        Parameters:
        - n: int, optional, default value is None, 
             the number of coherent modes to consider for the ratio calculation.
        
        Returns:
        None
        
        Notes:
        This function calculates and plots the normalized coherent mode ratio.
        It considers the specified number of coherent modes (or all if not specified), 
        and displays the ratio in a scatter plot.
        The x axis represents the coherent mode number, and the y axis represents the normalized ratio.
        
        """

        count = int(self.optic.n) if n == None else int(n)
        
        normalized_coherent_mode_ratio = (
            np.array(self.optic.ratio[0 : count])**2 / 
            np.sum(np.array(self.optic.ratio[0 : count])**2)
            )
        global_ratio = np.sum(normalized_coherent_mode_ratio**2) / np.sum(
            normalized_coherent_mode_ratio
            )**2
            
        self.global_doc = global_ratio
        print(self.global_doc, flush = True)
        
    #---------------------------------------------------
    # plot the cumulated occupation
    
    def occupation(self, n = None):
        
        """
        Plot the normalized coherent mode occupation.
        
        Parameters:
        - n: int, optional, default value is 100, 
             the number of coherent modes to consider for the occupation calculation.
        
        Returns:
        None
        
        Notes:
        This function calculates and plots the normalized coherent mode occupation.
        It considers the specified number of coherent modes, and displays the occupation in a scatter plot.
        The x axis represents the coherent mode number, and the y axis represents the normalized occupation.
        
        """

        count = int(self.optic.n) if n == None else int(n)
        
        plt.figure(figsize = (4, 4))  
        
        occup = list()
        for idx in range(int(count)):
            value = (
                np.sum(np.array(self.optic.ratio[0 : int(idx + 1)])**2) /
                np.sum(np.array(self.optic.ratio)**2)
                )
            occup.append(value)
        
        print(occup, flush = True)
        
        plt.scatter(range(count), occup)
        plt.plot(range(count), occup)
        
        # write label and title
        
        plt.title("coherent mode occupation")
        plt.xlabel('coherent mode number (n)', fontsize = 12)
        plt.ylabel('Occupation', fontsize = 12)
        
        plt.tight_layout()
    
    #---------------------------------------------------
    # plot certain cmode
    
    def cmode_idx(self, i = 0):
        
        figure, axes_handle = plt.subplots(1, 2, figsize = (9.5, 4))
        
        # plot intensity
        
        cmode_inten = axes_handle[0].imshow(
            np.abs(self.optic.cmode[i])**2 / np.max(np.abs(self.optic.cmode[i])**2),
            extent = self.extent
            )
        figure.colorbar(cmode_inten, ax = axes_handle[0])
        
        axes_handle[0].set_title("Intensity @index: %d" % (i))
        axes_handle[0].set_xlabel('x (%s)' % (micrometer_label_unit), fontsize = 12)
        axes_handle[0].set_ylabel('y (%s)' % (micrometer_label_unit), fontsize = 12)
        
        # plot phase
        
        cmode_phase = axes_handle[1].imshow(
            np.angle(self.optic.cmode[i]), extent = self.extent
            )
        figure.colorbar(cmode_phase, ax = axes_handle[1])
        
        axes_handle[1].set_title("Phase @index: %d" % (i))
        axes_handle[1].set_xlabel('x (%s)' % (micrometer_label_unit), fontsize = 12)
        axes_handle[1].set_ylabel('y (%s)' % (micrometer_label_unit), fontsize = 12)
        
        figure.tight_layout()
        
    #---------------------------------------------------
    # plot the cross spectral density

    def csd(self, method = "csd"):

        self.optic.cal_csd(direction = "x")
        self.optic.cal_csd(direction = "y")
        
        extent_x = [
            np.min(self.optic.xtick * 1e6), np.max(self.optic.xtick * 1e6),
            np.min(self.optic.xtick * 1e6), np.max(self.optic.xtick * 1e6)
            ]
        extent_y = [
            np.min(self.optic.ytick * 1e6), np.max(self.optic.ytick * 1e6),
            np.min(self.optic.ytick * 1e6), np.max(self.optic.ytick * 1e6)
            ]
        
        figure, axes_handle = plt.subplots(2, 2, figsize = (12, 10))
        
        #---------------------------------------------------
        # plot 2d csd
        
        def internal_plot_csd2d(
                csd2d_data, extent, ax_handle, csd2d_xlabel, csd2d_ylabel, title
                ):
            
            csd2d = ax_handle.imshow(np.abs(csd2d_data), extent = extent, cmap = 'coolwarm')
            figure.colorbar(csd2d, ax = ax_handle)
            
            ax_handle.set_title(title)
            ax_handle.set_xlabel(csd2d_xlabel, fontsize = 12)
            ax_handle.set_ylabel(csd2d_ylabel, fontsize = 12)
        
        internal_plot_csd2d(
             getattr(self.optic, method + "2x"), extent_x, 
             axes_handle[0, 0], 'x1 - x2 (%s)' % (micrometer_label_unit), 
            'x1 + x2 (%s)' % (micrometer_label_unit), 'csd2x')
        internal_plot_csd2d(
            getattr(self.optic, method + "2y"), extent_y, 
            axes_handle[1, 0], 'y1 - y2 (%s)' % (micrometer_label_unit), 
            'y1 + y2 (%s)' % (micrometer_label_unit), 
            'csd2y'
            )
        
        def internal_plot_csd1d(csd1d_data, xtick, ax_handle, csd2d_xlabel, csd2d_ylabel, title):
            
            ax_handle.plot(xtick, np.abs(csd1d_data))
            ax_handle.set_title(title)
            ax_handle.set_xlabel(csd2d_xlabel, fontsize = 12)
            ax_handle.set_ylabel(csd2d_ylabel, fontsize = 12)
        
        internal_plot_csd1d(
            getattr(self.optic, method + "1x")[int(self.optic.xcount / 2) :], 
            self.optic.xtick[int(self.optic.xcount / 2) :] * 1e6, axes_handle[0, 1], 
            'x1 - x2 (%s)' % (micrometer_label_unit), 'intensity (a. u.)', 
            'csd2x'
            )
        internal_plot_csd1d(
            getattr(self.optic, method + "1y")[int(self.optic.ycount / 2) :], 
            self.optic.ytick[int(self.optic.xcount / 2) :] * 1e6, axes_handle[1, 1], 
            'y1 - y2 (%s)' % (micrometer_label_unit), 'intensity (a. u.)', 
            'csd2y'
            )
    
    #---------------------------------------------------
    # plot the depth
    
    def depth(self, positions):
        
        depth_x = np.rot90(np.sum(np.abs(np.array(self.optic.cmode))**2, 1))
        depth_y = np.rot90(np.sum(np.abs(np.array(self.optic.cmode))**2, 2))
        
        figure, axes_handle = plt.subplots(1, 2, figsize = (12, 5))
        
        axes_handle[0].imshow(
            depth_x, aspect = "auto", extent = [
                np.min(positions), np.max(positions), 
                np.min(self.optic.ytick * 1e6), np.max(self.optic.ytick * 1e6)
                ])
        axes_handle[0].set_title("depth along x")
        axes_handle[0].set_xlabel(
            "depth (%s)" % (micrometer_label_unit), fontsize = 12
            )
        axes_handle[0].set_ylabel(
            "y (%s)" % (micrometer_label_unit), fontsize = 12
            )
        
        axes_handle[1].imshow(
            depth_y, aspect = "auto", extent = [
                np.min(positions), np.max(positions), 
                np.min(self.optic.xtick * 1e6), np.max(self.optic.xtick * 1e6)
                ])
        axes_handle[1].set_title("depth along y")
        axes_handle[1].set_xlabel(
            "depth (%s)" % (micrometer_label_unit), fontsize = 12
            )
        axes_handle[1].set_ylabel(
            "x (%s)" % (micrometer_label_unit), fontsize = 12
            )
        
        plt.tight_layout()
    
    def export_gif(self, gif_file_path, kind = "intensity", fps = 25):
        
        from matplotlib.animation import FuncAnimation, PillowWriter 
        
        figure, axis_handle = plt.subplots(figsize = (4, 4))
        counts = np.arange(len(self.optic.cmode))
        
        if kind == "intensity": 
            axis_handle.set_title("Intensity")
            image_object = plt.imshow(np.abs(self.optic.cmode[0])**2, extent = self.extent)
        elif kind == "phase": 
            axis_handle.set_title("Phase")
            image_object = plt.imshow(np.angle(self.optic.cmode[0]), extent = self.extent)
        
        axis_handle.set_xlabel('x (%s)' % (micrometer_label_unit), fontsize = 12)
        axis_handle.set_ylabel('y (%s)' % (micrometer_label_unit), fontsize = 12)
        
        def init_function():
            
            if kind == "intensity": 
                image_object.set_data(np.abs(self.optic.cmode[0])**2)
            elif kind == "phase": 
                image_object.set_data(np.angle(self.optic.cmode[0]))

            return image_object
            
        def update_function(idx):
            
            if kind == "intensity": 
                image_object.set_data(np.abs(self.optic.cmode[idx])**2)
            elif kind == "phase": 
                image_object.set_data(np.angle(self.optic.cmode[idx]))
                
            return image_object
            
        animation = FuncAnimation(
            figure, update_function, counts, init_func = init_function
            )  
        plt.show()
        plt.tight_layout()
        writer = PillowWriter(fps = fps)  
        animation.save(gif_file_path, writer = writer)  
        
    #---------------------------------------------------
    # save the figure

    def save(self, fig_name):

        plt.savefig(fig_name + '.png', dpi = 800)
        
#-----------------------------------------------------------------------------#
# main