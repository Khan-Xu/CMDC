#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
configure: The process of source construction.

Functions: None.
           
Classes  : None.
"""

#-----------------------------------------------------------------------------#
# library

from cat.utils import configure

#-----------------------------------------------------------------------------#
# parameters

undulator = {
        "period_length":        0.0186,
        "period_number":        215.0,
        "n_hormonic":           1,
        "hormonic_energy":      8000,
        "direction":            "v",
        "symmetry_v":           -1,
        "symmetry_h":           0
        }

electron_beam = {
        "n_electron":           50000,
        "current":              0.2,
        "energy":               6.0,
        "energy_spread":        1.06e-03,
        # "energy_spread":        0.0,
        "sigma_x0":             9.334e-06,
        "sigma_xd":             3.331e-06,
        "sigma_y0":             2.438e-06,
        "sigma_yd":             1.275e-06
        }

screen = {
        "xstart":               -0.0003,
        "xfin":                 0.0003,
        "nx":                   200,
        "ystart":               -0.0003,
        "yfin":                 0.0003,
        "ny":                   200,
        "screen":               20.0,
        "n_vector":             200
        }

source_file_name = "mci_8000eV.h5"
wfr_calculation_method = 'vib'

#-----------------------------------------------------------------------------#

configure.multi_electron_source(
    undulator, electron_beam, screen, source_file_name, wfr_calculation_method
    )

