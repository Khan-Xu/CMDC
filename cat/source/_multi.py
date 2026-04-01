#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.12.2021"
__version__  = "beta-0.3"
__email__    = "xuhan@ihep.ac.cn"


"""
_multi : The multiprocess tools base on mpi4py

Functions: _send     - block communicate send
           _recv     - block communicate recv
           _get_rank - get the rank number of current process
           _get_size - get the size of the rank
           
Classes  : None 
"""

#-----------------------------------------------------------------------------#
# library

import mpi4py.MPI as mpi
import numpy as np

#-----------------------------------------------------------------------------#
# constant

a = mpi.ANY_SOURCE  # recv from any source
i = mpi.INT         # int var
f = mpi.FLOAT       # float var
c = mpi.COMPLEX     # complex var

#-----------------------------------------------------------------------------#
# function

def _send(data, dtype = f, dest = 0, tag = 0): 

    """
    ---------------------------------------------------------------------------
    This function is a blocking communication operation that sends a numpy array
    to a specified process within the MPI communicator.

    Args:
        data (np.ndarray): The data array to send.
        dtype (MPI_Datatype, optional): The data type of the elements to be sent.Defaults to f (float32).
        dest (int, optional): The rank of the destination process. Defaults to 0.
        tag (int, optional): The message tag to identify the communication. Defaults to 0.

    Raises:
        ValueError: If the data provided is not a numpy array.

    Returns:
        None
    ---------------------------------------------------------------------------
    """
    
    # numpy array is used in this function
    
    if isinstance(data, np.ndarray):   
        mpi.COMM_WORLD.Send([data, dtype], dest = dest, tag = tag)
    else:
        raise ValueError("data is not numpy array")

        
def _recv(size, np_dtype = np.float32, dtype = f, source = 0, 
          tag = 0, mode = 'auto'):
    
    """
    ---------------------------------------------------------------------------
    This function is a blocking communication operation that receives a numpy array
    from a specified process within the MPI communicator.

    Args:
        size (int or tuple of ints): The size of the numpy array to be received.
        np_dtype (data-type, optional): The desired data-type for the array. Defaults to np.float.
        dtype (MPI_Datatype, optional): The data type of the elements to be received. Defaults to f (float32).
        source (int, optional): The rank of the source process. Defaults to 0.
        tag (int, optional): The message tag to identify the communication. Defaults to 0.
        mode (str, optional): The mode of receiving data. Defaults to 'auto'.

    Raises:
        ValueError: If the data received is not a numpy array.

    Returns:
        np.ndarray: The data array that was received.
    ---------------------------------------------------------------------------
    """
    
    if isinstance(size, (list, tuple)):
        data = np.zeros(size, dtype = np_dtype)
    else:
        data = np.zeros(int(size), dtype = np_dtype)

    if isinstance(data, np.ndarray):
        mpi.COMM_WORLD.Recv([data, dtype], source = source, tag = tag)
    else:
        raise ValueError("data is not numpy array")
        
    return data

def _get_rank():
    
    """
    ---------------------------------------------------------------------------
    Get the rank number of current process.
    
    Args: None.
    
    Return: the rank number of current process.  
    ---------------------------------------------------------------------------
    """

    return mpi.COMM_WORLD.Get_rank()


def _get_size():
    
    """
    ---------------------------------------------------------------------------
    Get the process number.
    
    Args: None.
    
    Return: the process number.  
    ---------------------------------------------------------------------------
    """
    
    return mpi.COMM_WORLD.Get_size()


#-----------------------------------------------------------------------------#        

        
    