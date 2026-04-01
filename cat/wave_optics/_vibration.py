#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Thu Sep 23 14:48:26 2021"
__email__    = "xuhan@ihep.ac.cn"

"""
Description: The generation of vibration.
"""

#-----------------------------------------------------------------------------#
# library

import os
import numpy as np

from copy import deepcopy

#-----------------------------------------------------------------------------#
# constant

#-----------------------------------------------------------------------------#
# function

# the function of multi sinc vibration

def sinc_vibration(amplitude, freq, phase, time):
    
    if not (isinstance(freq, list) and isinstance(phase, list) and 
        isinstance(amplitude, list)): 
        
        raise TypeError(
            "The data type of amplitude, freq and phase should be list"
            )
    
    if len(amplitude) == len(freq) == len(phase):
        
        for index, i_freq in enumerate(freq):
            
            if index == 0:
                vibration = (amplitude[0] * np.cos(2*np.pi * time * i_freq + phase[0]))
            else:
                vibration += (
                    amplitude[index] * np.cos(2*np.pi * time * i_freq + phase[index])
                    )
        
        return vibration
    
    else:
        
        raise ValueError(
            "The length of amplitude, freq and phase should be the same"
            )

# the function of monte carlo vibration

def monte_carlo_vibration(amplitude, freq, time):
    
    if not (isinstance(freq, list) and isinstance(amplitude, list)): 
        
        raise TypeError(
            "The data type of amplitude, freq and phase should be list"
            )
    
    if len(amplitude) == len(freq):
        
        for index, i_freq in enumerate(freq):
            
            phase = 2*np.pi * np.random.rand(np.shape(time)[0])
            if index == 0:
                vibration = (amplitude[0] * np.sin(2*np.pi * time * i_freq + phase))
            else:
                vibration += (
                    amplitude[index] * np.sin(2*np.pi * time * i_freq + phase)
                    )
        
        return vibration
    
    else:
        
        raise ValueError(
            "The length of amplitude, freq and phase should be the same"
            )

# # gaussian random vibration

# def gaussian_vibration(freq, sigma, amplitude):
    
#     freqency_spectrum = np.exp(1j)

#-----------------------------------------------------------------------------#
# class

#--------------------------------------------------------------------------
# the class of source
