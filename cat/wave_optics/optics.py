#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Thu Mar 21 11:04:14 2024"
__email__    = "xuhan@ihep.ac.cn"
__version__  = "beta-0.6"

"""
Description
"""

#-----------------------------------------------------------------------------#
# modules

import sys
import os
from copy import deepcopy

import numpy as np
import h5py as h5

from cat.wave_optics._optic_plane import _locate, _geometry
from cat.wave_optics._optic_plane import _op

#-----------------------------------------------------------------------------#
# parameters

#-----------------------------------------------------------------------------#
# functions

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

#-----------------------------------------------------------------------------#
# classes

#------------------------------------------------------
# source class

class source_optic(_op):
    
    """
    Initialize the source optic object.

    Parameters:
        source_file_name (str): The path to the source file.
        optic_name (str): The name of the optic (default is "source").
        n_vector (int): The number of vectors.
        i_vector (int): The index of the vector.
        position (int): The position.

    Raises:
        ValueError: If the source_file_name is unsupported or does not exist.
    """
    
    def __init__(
        self, source_file_name = None, optic_name = "source", n_vector = 0, 
        i_vector = None, position = 0
        ):
        
        if not os.path.isfile(source_file_name) or source_file_name is None: 
            raise ValueError("Unsupported source_file_name: {}".format(source_file_name))

        with h5.File(source_file_name, 'a') as f:
            
            #------------------------------------------------------
            # load geometry parameters from source file
            
            geometry_parameters = ["xstart", "xfin", "nx", "ystart", "yfin", "ny", "screen"]
            class_parameters = ["xstart", "xend", "xcount", "ystart", "yend", "ycount", "position"]
            for idx in range(7):
                setattr(
                    self, class_parameters[idx], 
                    np.array(f["description/%s" % (geometry_parameters[idx])])
                    )
            if position != 0: self.position = position
            self.n_row = np.copy(self.ycount)
            self.n_column = np.copy(self.xcount)
            self.n = n_vector

            # calcualte geometry parameters
            
            self.xpixel, self.xcoor, self.xtick = _geometry(self.xstart, self.xend, self.xcount)
            self.ypixel, self.ycoor, self.ytick = _geometry(self.ystart, self.yend, self.ycount)
            
            self.xgrid, self.ygrid = np.meshgrid(self.xtick, self.ytick)
            self.xcount = int(self.xcount)
            self.ycount = int(self.ycount)
            
            self.n_column = np.copy(self.xcount)
            self.n_row = np.copy(self.ycount)
            
            #------------------------------------------------------
            # the undulator parameters of source
            
            undulator_parameters = [
                "sigma_x0", "sigma_y0", "sigma_xd", "sigma_yd", "energy_spread", "current", 
                "hormonic_energy", "n_electron"
                ]
            for idx in range(8):
                setattr(
                    self, undulator_parameters[idx], 
                    np.array(f["description/%s" % (undulator_parameters[idx])])
                    )
            
            #------------------------------------------------------
            # the coherence properites of source.
            
            self.cmode = list()
            self.wavelength = np.array(f["description/wavelength"])
            self.ratio = np.array(f["coherence/eig_value"])
            
            if i_vector != None:
                cmode = np.reshape(
                    np.array(f["coherence/eig_vector"][:, i_vector]), (self.n_row, self.n_column)
                    )
                self.cmode.append(cmode)
                
            else:
                self.n = n_vector
                for i in range(n_vector):
                    cmode = np.reshape(
                        np.array(f["coherence/eig_vector"][:, i]), (self.n_row, self.n_column)
                        )
                    self.cmode.append(cmode)
                    
            self.name = optic_name
        
            #------------------------------------------------------
            # calculate the 1d slice
            
            if ("coherence/eig_vector_x" in f) and ("coherence/eig_vector_y" in f):
                
                self.cmode_x, self.cmode_y = [list(), list()]
                self.cmode_index = np.array(f["coherence/eig_vector_index"])
                
                if i_vector != None:
                    cmode_x = np.array(f["coherence/eig_vector_x"][:, i_vector])
                    self.cmode_x.append(cmode_x)
                    cmode_y = np.array(f["coherence/eig_vector_y"][:, i_vector])
                    self.cmode_y.append(cmode_y)
                    
                else:
                    self.n = n_vector
                    for idx_vector in range(n_vector):
                        cmode_x = np.array(f["coherence/eig_vector_x"][:, idx_vector])
                        self.cmode_x.append(cmode_x)
                        cmode_y = np.array(f["coherence/eig_vector_y"][:, idx_vector])
                        self.cmode_y.append(cmode_y)

#------------------------------------------------------
# screen class

class screen(_op):
    
    def __init__(
        self, optic = None, optic_file = None, name = "screen", n_vector = 0, 
        i_vector = None, position = 0, dim = 2
        ):
        
        super().__init__(
            optic = optic, optic_file = optic_file, 
            name = name, n_vector = n_vector, i_vector = i_vector, position = position,
            dim = dim
            )

#------------------------------------------------------
# ideal lens

class ideal_lens(_op):
    
    def __init__(
        self, 
        optic = None, optic_file = None, name = "ideal_lens", 
        n_vector = 0, i_vector = None, position = 0, xfocus = 0, yfocus = 0,
        dim = 2
        ):
        
        super().__init__(
            optic = optic, optic_file = optic_file, 
            name = name,  n_vector = n_vector, i_vector = i_vector,
            position = position, dim = dim
            )
        
        #---------------------------------------------------
        # the lens_phase of lens
        
        self.focus_x = xfocus
        self.focus_y = yfocus
        
        k_vector = 2 * np.pi / self.wavelength
        
        if self.dim == 2:
        
            phase_map = (
                np.sqrt(self.focus_x**2 + self.xgrid**2) - np.abs(self.focus_x) +
                np.sqrt(self.focus_y**2 + self.ygrid**2) - np.abs(self.focus_y)
                )
            self.lens_phase = np.exp(1j * k_vector * phase_map)
                
        elif self.dim == 1:
            
            phase_x = np.sqrt(self.focus_x**2 + self.xtick**2) - np.abs(self.focus_x)
            phase_y = np.sqrt(self.focus_y**2 + self.ytick**2) - np.abs(self.focus_y)
            
            self.lens_phase_x = np.exp(1j * k_vector * phase_x)
            self.lens_phase_y = np.exp(1j * k_vector * phase_y)
        
        if self.dim == 2:
            
            for idx in range(int(self.n)): 
                self.cmode[idx] *= self.lens_phase
        
        elif self.dim == 1:
            
            for idx in range(int(self.n)):
                self.cmode_x[idx] *= self.lens_phase_x
                self.cmode_y[idx] *= self.lens_phase_y
            
#------------------------------------------------------
# crl class

class crl(_op):
    
    def __init__(
        self, 
        optic = None, optic_file = None, name = "crl", 
        n_vector = 0, i_vector = None, position = 0, nlens = 0, 
        delta = 2.216e-6, rx = 0, ry = 0, dim = 2
        ):
    
        
        super().__init__(
            optic = optic, optic_file = optic_file, name = name, 
            n_vector = n_vector, i_vector = i_vector, 
            position = position, dim = dim
            )
        
        #---------------------------------------------------
        # the lens_phase of lens
        
        self.focus_x = rx / (2 * nlens * delta) if rx != 0 else 1e20
        self.focus_y = ry / (2 * nlens * delta) if ry != 0 else 1e20
        
        k_vector = 2 * np.pi / self.wavelength
        
        self.lens_phase = np.exp(
            1j * k_vector *
            (self.xgrid**2 / (2 * self.focus_x) + self.ygrid**2 / (2 * self.focus_y))
            )
        self.lens_phase_x = np.exp(1j * k_vector * (self.xtick**2 / (2 * self.focus_x)))
        self.lens_phase_y = np.exp(1j * k_vector * (self.ytick**2 / (2 * self.focus_y)))
        
        if self.dim == 2:
            
            for idx in range(int(self.n)): 
                self.cmode[idx] *= self.lens_phase
        
        elif self.dim == 1:
            
            for idx in range(int(self.n)):
                self.cmode_x[idx] *= self.lens_phase_x
                self.cmode_y[idx] *= self.lens_phase_y

#------------------------------------------------------
# kb class

class kb(_op):

    def __init__(
        self, 
        optic = None, optic_file = None, name = "kb_mirror", 
        direction = 'v', n_vector = 0, i_vector = None, position = 0,
        pfocus = 0, qfocus = 0, dim = 2
        ):
    
        super().__init__(
            optic = optic, optic_file = optic_file, 
            name = name, n_vector = n_vector, i_vector = i_vector, 
            position = position, dim = dim
            )
        
        k_vector = 2 * np.pi / self.wavelength
        
        if self.dim == 2:
            
            if direction == 'h':
                
                self.lens_phase = np.exp(
                    1j * (2*np.pi / self.wavelength) *
                    (np.sqrt(self.xgrid**2 + pfocus**2) + np.sqrt(self.xgrid**2 + qfocus**2))
                    )
                
            elif direction == 'v':
                
                self.lens_phase = np.exp(
                    1j * (2*np.pi / self.wavelength) *
                    (np.sqrt(self.ygrid**2 + pfocus**2) + np.sqrt(self.ygrid**2 + qfocus**2))
                    )

        elif self.dim == 1:
            
            if direction == 'h':
                
                self.lens_phase_x = np.exp(1j * k_vector * (
                    np.sqrt(self.xtick**2 + pfocus**2) + np.sqrt(self.xtick**2 + qfocus**2)
                    ))
                self.lens_phase_y = 1
                
            elif direction == 'v':
                
                self.lens_phase_x = 1
                self.lens_phase_y = np.exp(1j * k_vector * (
                    np.sqrt(self.ytick**2 + pfocus**2) + np.sqrt(self.ytick**2 + qfocus**2)
                    ))
            
        if self.dim == 2:
            
            for idx in range(int(self.n)): 
                self.cmode[idx] *= self.lens_phase
        
        elif self.dim == 1:
            
            for idx in range(int(self.n)):
                self.cmode_x[idx] *= self.lens_phase_x
                self.cmode_y[idx] *= self.lens_phase_y

#------------------------------------------------------
# akb class


class akb(_op):

    def __init__(
        self, 
        optic = None, optic_file = None, name = "akb_mirror", 
        direction = 'v', kind = 'ep', n_vector = 0, i_vector = None, position = 0,
        pfocus = 0, qfocus = 0, afocus = 0, bfocus = 0, dim = 2
        ):
    
        super().__init__(
            optic = optic, optic_file = optic_file, 
            name = name, n_vector = n_vector, i_vector = i_vector, 
            position = position, dim = dim
            )
            
        #---------------------------------------------------
        # lens phase
        
        k_vector = 2 * np.pi / self.wavelength
        
        if direction == 'h':
            
            if kind == 'ep':
                
                self.lens_phase = np.exp(
                    1j * (2*np.pi / self.wavelength) *
                    (np.sqrt(self.xgrid**2 + pfocus**2) + np.sqrt(self.xgrid**2 + qfocus**2))
                    )
                self.lens_phase_x = np.exp(1j * k_vector * (
                    np.sqrt(self.xtick**2 + pfocus**2) + np.sqrt(self.xtick**2 + qfocus**2)
                    ))
                self.lens_phase_y = 1
                
            elif kind == 'hb':
                
                self.lens_phase = np.exp(
                    1j * (2*np.pi / self.wavelength) *
                    (np.sqrt(self.xgrid**2 + afocus**2) - np.sqrt(self.xgrid**2 + bfocus**2))
                    )
                self.lens_phase_x = np.exp(1j * k_vector * (
                    np.sqrt(self.xtick**2 + afocus**2) - np.sqrt(self.xtick**2 + bfocus**2)
                    ))
                self.lens_phase_y = 1
        
        elif direction == 'v':
            
            if kind == 'ep':
            
                self.lens_phase = np.exp(
                    1j * (2*np.pi / self.wavelength) *
                    (np.sqrt(self.ygrid**2 + pfocus**2) + np.sqrt(self.ygrid**2 + qfocus**2))
                    )
                self.lens_phase_x = 1
                self.lens_phase_y = np.exp(1j * k_vector * (
                    np.sqrt(self.ytick**2 + pfocus**2) + np.sqrt(self.ytick**2 + qfocus**2)
                    ))
            
            elif kind == 'hb':
            
                self.lens_phase = np.exp(
                    1j * (2*np.pi / self.wavelength) *
                    (np.sqrt(self.ygrid**2 + afocus**2) - np.sqrt(self.ygrid**2 + bfocus**2))
                    )
                self.lens_phase_x = 1
                self.lens_phase_y = np.exp(1j * k_vector * (
                    np.sqrt(self.ytick**2 + afocus**2) - np.sqrt(self.ytick**2 + bfocus**2)
                    ))
        
        if self.dim == 2:
            
            for idx in range(int(self.n)): 
                self.cmode[idx] *= self.lens_phase
        
        elif self.dim == 1:
            
            for idx in range(int(self.n)):
                self.cmode_x[idx] *= self.lens_phase_x
                self.cmode_y[idx] *= self.lens_phase_y

#-----------------------------------------------------------------------------#
# main