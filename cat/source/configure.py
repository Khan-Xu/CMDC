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

from cat.source import _source
from cat.source import _decomposition
from cat.source import _multi

#-----------------------------------------------------------------------------#
# constant

_N_SVD_TOP = 4000
_N_SVD_OPT = 2500
_N_SVD_TOL = 1500

c_rank = _multi._get_rank()

#-----------------------------------------------------------------------------#
# function

def multi_electron_source(
    undulator, electron_beam, screen, source_file_name, wfr_calculation_method
    ):

    # wfrs calculation

    _source._cal_wfrs(
        undulator, electron_beam, screen, 
        file_name = source_file_name, method = wfr_calculation_method
        )

    n_rank = _multi._get_size()
    n_electron = electron_beam['n_electron']
    
    #---------------------------------------------------
    # standard multi-layer SVD method
    
    if int(n_electron/n_rank) > _N_SVD_TOL and int(n_electron/n_rank) < _N_SVD_TOP:
        
        _func = _decomposition._multi_layer_svd_exceed_0

        if c_rank == 0:
            print(
                "coherent mode decomposition start. (multi_layer_SVD method)", 
                flush = True
                )
    
    #---------------------------------------------------
    # less electron multi-layer SVD method
    
    elif int(n_electron/n_rank) < _N_SVD_TOL:
        
        _func = _decomposition._multi_layer_svd_exceed_1

        if c_rank == 0:
            print(
                "coherent mode decomposition start. (multi_layer_SVD method)", 
                flush = True
                )
    
    #---------------------------------------------------
    # calculated CSD and decomposition
    
    else:
        
        _func = _decomposition._CSD_eigsh

        if c_rank == 0:
            print(
                "coherent mode decomposition start. (CSD calcuation and decomposition)", 
                flush = True
                ) 

    _func(
        electron_beam['n_electron'], 
        int(screen['nx']), int(screen['ny']), int(screen['n_vector']),
        file_name = source_file_name
        )

def single_electron(undulator, electron_beam, screen, mc_para = [0, 0, 0, 0, 0]):
    
    return _source._cal_single(
        undulator, electron_beam, screen, mc_para = [0, 0, 0, 0, 0]
        )