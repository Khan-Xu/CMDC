#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Sun Feb 18 14:43:00 2024"
__email__    = "xuhan@ihep.ac.cn"


"""
Description
"""

#-----------------------------------------------------------------------------#
# modules

import os
import sys

import numpy as np
import h5py as h5

from scipy.linalg import eig as acc_eig
from scipy.sparse import linalg as sparse_linalg

from cat.source import _constant
from cat.source import _file_utils

#-----------------------------------------------------------------------------#
# parameters

#-----------------------------------------------------------------------------#
# functions

#---------------------------------------------------
# the calculation of hermite mode

#---------------------------------------------------
# the calculation of cross sepctral density


# TODO: Phase calculation is WRONG!!!!! Use hermite gsm mode calculation instead

def _calculate_gsm_csd(
        x, k_vector, beam_size, coherence_length, magnify_factor, beam_radius
        ):
    
    """
    ---------------------------------------------------------------------------
    Calculate the cross-spectral density (CSD) of a Gaussian Schell-model beam.

    Args:
        x (array_like): 1-D array of spatial coordinates
        k_vector (float): wave vector
        beam_size (float): size of the beam
        coherence_length (float): coherence length of the beam
        magnify_factor (float): magnification factor
        beam_radius (float): radius of the beam
    
    Returns:
        ndarray: Cross-spectral density (CSD) of the Gaussian Schell-model beam evaluated 
        on the grid defined by the input spatial coordinates x.
    ---------------------------------------------------------------------------
    """

    x1, x2 = np.meshgrid(x, x)
    
    a = 1 /magnify_factor
    intensity_part = -(x1**2 + x2**2) / (4 * beam_size**2)
    coherence_part = -(x1 - x2)**2 / (2 * coherence_length**2)
    phase = 1j * k_vector * (x2**2 - x1**2) / (2 * beam_radius)
    
    csd_x1x2 = a * np.exp(intensity_part + coherence_part + phase)
    
    return csd_x1x2

#-----------------------------------------------------------------------------#
# classes

#---------------------------------------------------
# the calculation of parameters of synchrotrn gaussian schell source

class _gaussian_source_parameters(object):
    
    """
    ---------------------------------------------------------------------------
    Class for calculating parameters of a Gaussian source.
    
    Attributes:
        light_eV (float): Energy of light in electron-volts
        wave_length (float): Wavelength of the source
        k_vector (float): Wave vector of the source
        se_size (float): Single electron size of the source
        se_divergence (float): Single electron divergence of the source
    
    Methods:
        _single_electron_source(undulator_length, undulator_hormonic_energy): Calculate 
        parameters for a single electron source based on the undulator properties.
        
        _gaussian_source(se_size, se_divergence, electron_beam_size, electron_beam_divergence, k_vector): 
        Calculate parameters for a Gaussian source based on single electron and electron beam properties.
        
        propagate_parameters(distance, effective_distance): Calculate magnification and beam radius
        based on the propagation distance and effective distance.
    ---------------------------------------------------------------------------
    """

    def __init__(self):
        
        """
        -----------------------------------------------------------------------
        Initialize the Gaussian source parameters class.
        Sets the light energy in electron-volts based on a constant.
        -----------------------------------------------------------------------
        """

        self.light_eV = _constant._Light_eV_mu
    
    def _single_electron_source(
            self, undulator_length, undulator_hormonic_energy
            ):
        
        """
        -----------------------------------------------------------------------
        Calculate parameters for a single electron source.
        
        Args:
            undulator_length (float): Length of the undulator
            undulator_hormonic_energy (float): Harmonic energy of the undulator
        
        Updates the wave length, wave vector, single electron size, and single electron
        divergence attributes based on the input undulator properties.
        -----------------------------------------------------------------------
        """

        self.wave_length = self.light_eV * 1e-6 / undulator_hormonic_energy
        self.k_vector = 2 * np.pi / self.wave_length
        self.se_size = (2 * undulator_length * self.wave_length)**0.5 / (4 * np.pi)
        self.se_divergence = (self.wave_length / (2 * undulator_length))**0.5
    
    def _gaussian_source(
            self, se_size, se_divergence, electron_beam_size, 
            electron_beam_divergence, k_vector
            ):
        
        """
        -----------------------------------------------------------------------
        Calculate parameters for a Gaussian source.
        
        Args:
            se_size (float): Single electron size of the source
            se_divergence (float): Single electron divergence of the source
            electron_beam_size (float): Size of the electron beam
            electron_beam_divergence (float): Divergence of the electron beam
            k_vector (float): Wave vector of the source
        
        Returns:
            dict: Dictionary containing various source parameters such as size, divergence, emittance,
            degree of coherence, coherence length, and effective distance based on the input parameters.
        -----------------------------------------------------------------------
        """

        self.se_emittance = se_size * se_divergence
        
        gs_dict = dict()
        gs_dict["source_size"] = np.sqrt(se_size**2 + electron_beam_size**2)
        gs_dict["source_divergence"] = np.sqrt(se_divergence**2 + electron_beam_divergence**2)
        gs_dict["source_eimittance"] = gs_dict["source_size"] * gs_dict["source_divergence"]
        
        # the calculation of coherence properites
        gs_dict["degree_of_coherence"] = self.se_emittance / gs_dict["source_eimittance"]
        gs_dict["coherence_length"] = (
            2 * gs_dict["source_size"] * gs_dict["degree_of_coherence"] / 
            np.sqrt(1 - gs_dict["degree_of_coherence"]**2)
            )
        gs_dict["effective_distance"] = (
            2 * k_vector * gs_dict["source_size"]**2 * gs_dict["degree_of_coherence"]
            )
        
        return gs_dict
        
    def propagate_parameters(self, distance, effective_distance):
        
        """
        -----------------------------------------------------------------------
        Calculate magnification and beam radius based on the propagation distance and effective distance.
        
        Args:
            distance (float): Propagation distance
            effective_distance (float): Effective distance of the source
        
        Returns:
            tuple: A tuple containing the magnification and beam radius calculated based on the inputs.
        -----------------------------------------------------------------------
        """

        magnification = np.sqrt(1 + (distance / effective_distance)**2)
        beam_radius = distance * (1 + (effective_distance / distance)**2)
        
        return magnification, beam_radius

#---------------------------------------------------
# the calculation of coherent modes of synchrotrn gaussian schell source

class _gaussian_schell_mode(_gaussian_source_parameters):
    
    def __init__(self, undulator, electron_beam, screen, n = 500):
        
        super().__init__()
        self.undulator = undulator
        self.electron_beam = electron_beam
        self.screen = screen
        self.n = n
        
        # geometry
        self.distance = self.screen["screen"]
        self.x_range = np.linspace(self.screen["xstart"], self.screen["xfin"], int(self.screen["nx"]))
        self.y_range = np.linspace(self.screen["ystart"], self.screen["yfin"], int(self.screen["ny"]))
        
        # the source of the single electron
        self._single_electron_source(
            self.undulator["period_length"] * self.undulator["period_number"], 
            self.undulator["hormonic_energy"]
            )
        
        # the source parameters calculation in x direction
        self.gs_source_x = self._gaussian_source(
            self.se_size, self.se_divergence, 
            self.electron_beam["sigma_x0"], self.electron_beam["sigma_xd"], self.k_vector
            )
        self.magnify_x, self.beam_radius_x = self.propagate_parameters(
            screen["screen"], self.gs_source_x["effective_distance"]
            )
        self.bs_x_distance = self.magnify_x * self.gs_source_x["source_size"]
        
        # the source parameters calculation in y direction
        self.gs_source_y = self._gaussian_source(
            self.se_size, self.se_divergence, 
            self.electron_beam["sigma_y0"], self.electron_beam["sigma_yd"], self.k_vector
            )
        self.magnify_y, self.beam_radius_y = self.propagate_parameters(
            screen["screen"], self.gs_source_y["effective_distance"]
            )
        self.bs_y_distance = self.magnify_y * self.gs_source_y["source_size"]
    
    #---------------------------------------------------
    # the calculation of coherent modes and ratio 1d
    
    def ratio_1d(self, n = 5000, direction = "x", norm = True):
        
        if direction == "x": 
            doc = self.gs_source_x["degree_of_coherence"]
            kai = (1 - doc) / (1 + doc)
            beta_0 = (2 * np.pi)**0.5 * self.gs_source_x["source_size"] * 2 * doc / (1 + doc)
            beta = [kai**idx * beta_0 for idx in range(int(n))]
            
        elif direction == "y": 
            doc = self.gs_source_y["degree_of_coherence"]
            kai = (1 - doc) / (1 + doc)
            beta_0 = (2 * np.pi)**0.5 * self.gs_source_y["source_size"] * 2 * doc / (1 + doc)
            beta = [kai**idx * beta_0 for idx in range(int(n))]
        
        eigenvalues = np.array(beta) / np.sum(beta) if norm else beta
        
        return eigenvalues
    
    def hermite_mode_1d(self, index, direction = "x"):
        
        def _calculate_hermite_mode(
                bs_at_distance, doc_1d, idx, x_range, k_vector, distance, 
                effective_distance, beam_radius
                ):
            
            from scipy import special
            
            amplitude_constant = (
                (np.pi * bs_at_distance**2 * doc_1d)**-0.25 * (2**idx * special.factorial(idx))**-0.5
                )
            hermite = special.hermite(int(idx))(x_range / (bs_at_distance * (2 * doc_1d)**0.5))
            decay = np.exp(-x_range**2 / (4 * bs_at_distance**2 * doc_1d))
            phase = (
                k_vector * distance - 
                (idx + 1) * np.arctan(distance / effective_distance) + 
                k_vector * x_range**2 / (2 * beam_radius)
                )
            mode_idx = amplitude_constant * hermite * decay * np.exp(1j * phase) 
            
            return mode_idx
        
        if direction == "x":
            return _calculate_hermite_mode(
                self.bs_x_distance, self.gs_source_x["degree_of_coherence"], index, 
                self.x_range, self.k_vector, self.distance, self.gs_source_x["effective_distance"], 
                self.beam_radius_x
                )
        
        elif direction == "y":
            return _calculate_hermite_mode(
                self.bs_y_distance, self.gs_source_y["degree_of_coherence"], index, 
                self.y_range, self.k_vector, self.distance, self.gs_source_y["effective_distance"], 
                self.beam_radius_y
                )
        
    #---------------------------------------------------
    # the calculation of coherent modes and ratio 2d
    
    def ratio_2d(self):
        
        # eigenvalues calculation and index determination
        
        eighenvalues = np.array([
            eighenvalue_ix * eighenvalue_iy 
            for eighenvalue_iy in self.ratio_1d(n = 1000, direction = "y", norm = False) 
            for eighenvalue_ix in self.ratio_1d(n = 1000, direction = "x", norm = False)
            ])
        sort_index = np.flipud(np.argsort(eighenvalues))
        eighenvalues = eighenvalues[sort_index] / np.sum(eighenvalues)
        
        self.eighenvalues = eighenvalues
        
    def hermite_mode_2d(self):
        
        # eigenvalues calculation and index determination
        
        eighenvalues = np.array([
            eighenvalue_ix * eighenvalue_iy 
            for eighenvalue_ix in self.ratio_1d(n = 1000, direction = "x", norm = False) 
            for eighenvalue_iy in self.ratio_1d(n = 1000, direction = "y", norm = False)
            ])
        sort_index = np.flipud(np.argsort(eighenvalues))
        eighenvalues = eighenvalues[sort_index] / np.sum(eighenvalues)
        mode_index = np.array(
            [[index_x, index_y] for index_x in range(1000) for index_y in range(1000)]
            )[sort_index, :]
        
        # calculate all the hermite modes based on the eigenvalues
        
        hermite_mode = list()
        for index_x, index_y in mode_index[0 : int(self.n), :]:
            mode_index_x = np.matrix(self.hermite_mode_1d(int(index_x), direction = "x"))
            mode_index_y = np.matrix(self.hermite_mode_1d(int(index_y), direction = "y"))
            hermite_mode.append(np.dot(mode_index_y.T, mode_index_x))
        
        self.eighenvalues = np.sqrt(eighenvalues[0 : int(self.n)]) 
        self.mode = np.array(hermite_mode)
        self.mode_index = mode_index
    
    #---------------------------------------------------
    # the calculation of cross spectral density 1d
    
    # TODO: Phase calculation is WRONG!!!!! Use hermite gsm mode calculation instead
    
    def gaussian_csd_1d(self):
        
        # gsm csd, ratio, and modes, 1d calculation 
        
        # x direction
        self.csd_1dx = _calculate_gsm_csd(
            self.x_range, self.k_vector, self.bs_x_distance, 
            self.magnify_x * self.gs_source_x["coherence_length"], self.magnify_x,
            self.beam_radius_x
            )
        ratio_x, gsm_mode_x = acc_eig(self.csd_1dx)
        self.ratio_1dx = ratio_x[0 : int(self.n)].real
        self.gsm_mode_x = gsm_mode_x[:, 0 : int(self.n)]
        
        # y direction
        self.csd_1dy = _calculate_gsm_csd(
            self.y_range, self.k_vector, self.bs_y_distance, 
            self.magnify_y * self.gs_source_y["coherence_length"], self.magnify_y,
            self.beam_radius_y
            )
        ratio_y, gsm_mode_y = acc_eig(self.csd_1dy)
        self.ratio_1dy = ratio_y[0 : int(self.n)].real
        self.gsm_mode_y = gsm_mode_y[:, 0 : int(self.n)]
    
    def gaussian_mode_2d(self):
        
        self.gaussian_csd_1d()
        
        # the calculation of gaussian mode
        
        eighenvalues = np.array([
            eighenvalue_ix * eighenvalue_iy 
            for eighenvalue_ix in self.ratio_1dx for eighenvalue_iy in self.ratio_1dy
            ])
        sort_index = np.flipud(np.argsort(eighenvalues))
        eighenvalues = eighenvalues[sort_index] / np.sum(eighenvalues)
        mode_index = np.array(
            [[idx, idy] for idx in range(int(self.n)) for idy in range(int(self.n))]
            )[sort_index, :]
        
        # calculate all the hermite modes based on the eigenvalues
        
        hermite_mode = [
            np.dot(np.matrix(self.gsm_mode_y[:, idy]).T, np.matrix(self.gsm_mode_x[:, idx])) 
            for idx, idy in mode_index[0 : int(self.n), :]
            ]
        
        self.eighenvalues = np.sqrt(eighenvalues[0 : int(self.n)]) 
        self.mode = np.array(hermite_mode)
        self.mode_index = mode_index
        
    #---------------------------------------------------
    # export the coherent mode to h5 file
    
    def save_h5(self, file_name, cal_mode = "cmd"):
        
        # TODO: Phase calculation is DIFFERENT!!!!! Use CSD 1d decompostion 
        
        if cal_mode == "hermite": self.hermite_mode_2d()
        elif cal_mode == "cmd": self.gaussian_mode_2d()
        
        _file_utils._construct_source_file(
            file_name, self.electron_beam, self.undulator, self.screen, self.wave_length
            )
        hermite_mode = np.reshape(
            self.mode, (int(self.n), (self.screen["nx"] * self.screen["ny"]))
            ).T
        hermite_mode = np.real(hermite_mode) - 1j * np.imag(hermite_mode)
        
        with h5.File(file_name, "a") as f:
            coherence_group = f.create_group("coherence")
            coherence_group.create_dataset("eig_value", data = self.eighenvalues[0 : int(self.n)])
            coherence_group.create_dataset("eig_vector", data = hermite_mode) 
            coherence_group.create_dataset(
                "eig_vector_x", 
                data = self.gsm_mode_x[:, 0 : int(self.n)].real - 1j * self.gsm_mode_x[:, 0 : int(self.n)].imag
                )
            coherence_group.create_dataset(
                "eig_vector_y", 
                data = self.gsm_mode_y[:, 0 : int(self.n)].real - 1j * self.gsm_mode_y[:, 0 : int(self.n)].imag
                )
            coherence_group.create_dataset("eig_vector_index", data = self.mode_index) 
    
#-----------------------------------------------------------------------------#
# main