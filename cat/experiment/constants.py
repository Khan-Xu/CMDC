#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Mon Apr 17 14:22:42 2023"
__email__    = "xuhan@ihep.ac.cn"


"""
Description: load atomic form factors, atomic mass, atomic scattering factors
"""

#-----------------------------------------------------------------------------#
# modules

import os
import sys

import numpy as np
import h5py as h5

#-----------------------------------------------------------------------------#
# parameters

aff_file = "atomic_form_factor.h5"
am_file = "atomic_mass.h5"
asf_file = "atomic_scattering_factor.h5"

aff_keys = ['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4', 'c']

#-----------------------------------------------------------------------------#
# functions

#-------------------------------------------------
# load basic parameters 

# return atomic form factor of a certain ions

def atomic_form_factor(file_path = None, ions = None):
    
    """Return a dictionary of atomic form factors for the given ions.

    The atomic form factors are read from a HDF5 file specified by the file path.
    The file must contain a group named 'atomic_form_factor' with datasets for each ion.
    The function raises an exception if the file does not exist or the ions are not found in the file.

    Args:
        file_path (str): The path to the HDF5 file that contains the atomic form factors.
        ions (list, optional): A list of ions to get the atomic form factors for. Defaults to None.

    Returns:
        dict: A dictionary of numpy arrays with the keys being the ion names and the values being the atomic form factors.

    Raises:
        FileExistsError: If the file does not exist or is not a regular file.
        KeyError: If the ions do not exist or are not included in the file.
    """
    
    if file_path is None:
        aff_filepath = os.path.join(
            os.path.abspath(os.getcwd()), r"cat\experiment\atomic_form_factor.h5"
            )
    else:
        aff_filepath = os.path.join(file_path, aff_file)
    
    if not os.path.exists(aff_filepath):
        
        raise FileExistsError("The atomic form factor file dose not exist")
    else:
        with h5.File(aff_filepath, 'a') as aff:
            try: 
                return np.array(aff['atomic_form_factor'][ions])
            except:
                raise KeyError("The ions dose not exist or not included")

# return atomic mass of a certain element
    
def atomic_mass(file_path = '', element = None):
    
    """Returns the atomic mass of an element from a file.

    Args:
        file_path (str): The path to the file that contains the atomic mass data.
        element (str): The name of the element to look up. If None, returns None.
    
    Returns:
        float: The atomic mass of the element in atomic mass units (amu), or None if the element is not found.
    
    Raises:
        FileExistsError: If the file_path does not exist.
        KeyError: If the element is not in the file or is None.
    """

    am_filepath = os.path.join(file_path, am_file)
    
    if not os.path.exists(am_filepath):
        raise FileExistsError("The atomic mass file dose not exist")
    else:
        with h5.File(am_filepath, 'a') as am:
            try: return np.array(am['atomic_mass'][element])
            except:
                raise KeyError("The element dose not exist or not included")      

def atomic_scattering_factor(file_path = '', element = None, energy = 'all'):
    
    """Returns the atomic scattering factor of an element from a file.

    Args:
        file_path (str): The path to the file that contains the atomic scattering factor data.
        element (str): The name of the element to look up. If None, returns None.
        energy (float or int or str): The energy value to look up. If 'all', returns all the energy values and atomic scattering factors. If None, returns None.
    
    Returns:
        dict or list or None: A dictionary with keys 'energy', 'imag_f' and 'real_f' and values as arrays of energy values and atomic scattering factors, or a list with three elements [energy, energy_imag_f, energy_real_f], or None if the element or energy is not found.
    
    Raises:
        FileExistsError: If the file_path does not exist.
        KeyError: If the element is not in the file or is None.
    """

    asf_filepath = os.path.join(file_path, asf_file)
    
    if not os.path.exists(asf_filepath):
        raise FileExistsError("The atomic scattering factor file dose not exist")
    else:
        with h5.File(asf_filepath, 'a') as asf:
            try: 
                energys = np.array(asf['atomic_scattering_factor'][element + '_energy'])
                imag_f = np.array(asf['atomic_scattering_factor'][element + '_imag_f'])
                real_f = np.array(asf['atomic_scattering_factor'][element + '_real_f'])
            except:
                raise KeyError("The element dose not exist or not included") 
    
    if energy == 'all':
        return {'energy': energy, 'imag_f': imag_f, 'real_f': real_f}
    elif isinstance(energy, (float, int)):
        energy_imag_f = np.interp(energy, energys, imag_f)
        energy_real_f = np.interp(energy, energys, real_f)
        return [energy, energy_imag_f, energy_real_f]
        
#-------------------------------------------------
# calculate basic parameters of the cell structure

def density(elements_mass, counts, lattice):
     
    """Calculate the density of a crystal.
    
    Args:
        elements_mass (list): A list of atomic masses of elements in the unit cell (in g/mol).
        counts (list): A list of numbers of atoms of each element in the unit cell.
        lattice (list): A list of lattice parameters of the unit cell (in Å), in the order of a, b, c, α, β, γ.
    
    Returns:
        float: The density of the crystal (in g/cm³).
    """

    # Extract the lattice parameters from the lattice list 
    lattice_a, lattice_b, lattice_c = lattice[0 : 3]
    alpha, beta, gamma = lattice[3 :]
    
    # Calculate the volume of the unit cell using trigonometric formula
    lattice_volume = (
        lattice_a * lattice_b * lattice_c *
        np.sqrt(
            1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) -
            np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
            )
        )
    mass_cell = np.array(elements_mass) * np.array(counts)
    
    # Calculate the density by dividing the mass by the volume, 
    # Avogadro's number and a unit conversion factor
    return np.sum(mass_cell) / (lattice_volume * 6.022e23 * 0.1)
 
def attenuation_coefficient(elements_mass, counts, asf2_list, energy):
    
    """Calculate the attenuation coefficient of a crystal.
    Args:
        elements_mass (list): A list of atomic masses of elements in the unit cell (in g/mol).
        counts (list): A list of numbers of atoms of each element in the unit cell.
        asf2_list (list): A list of squared atomic scattering factors of each element in the unit cell.
        energy (float): The energy of the X-ray beam (in keV).
    
    Returns:
        float: The attenuation coefficient of the crystal (in cm⁻¹).
    
    """
    
    # Calculate the wavelength of the X-ray beam using Planck's constant
    wave_length = 12.38 / energy # unit Angs
    
    # Define some constants, such as Avogadro's number and classical electron radius
    na = 6.022e23 / 1e23
    r0 = 2.82e-5
    
    # Calculate the attenuation coefficient by summing up the contributions from each element
    
    return np.sum(
        2 * r0 * na * wave_length * 
        np.array(elements_mass) * np.array(counts) * np.array(asf2_list) 
        )
               
#-----------------------------------------------------------------------------#
# classes

#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    pass

    
    
    
    
    

    