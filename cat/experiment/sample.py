#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Tue Mar 26 15:17:20 2024"
__email__    = "xuhan@ihep.ac.cn"


"""
Description
"""

#-----------------------------------------------------------------------------#
# modules

import sys
import os

from copy import deepcopy

import numpy as np
import scipy as sp

#-----------------------------------------------------------------------------#
# parameters

#-----------------------------------------------------------------------------#
# functions

#------------------------------------------------------
# support functions

def spiral_archimedes(spiral_speed, count):
    
    """
    Generate points on an Archimedean spiral.
    
    Parameters:
        - spiral_speed (float): The speed factor of the spiral.
        - count (int): The number of points to generate.
    
    Returns:
        - pos_x (numpy.ndarray): Array of x-coordinates of the generated points.
        - pos_y (numpy.ndarray): Array of y-coordinates of the generated points.
    """

    pos_x, pos_y = list(), list()
    spiral_step = np.pi
    
    for idx in range(count):
        
        spiral_theta = deepcopy(spiral_step)
        spiral_radius = deepcopy(spiral_speed * spiral_step / (2*np.pi))
        
        pos_x.append(spiral_radius * np.cos(spiral_theta))
        pos_y.append(spiral_radius * np.sin(spiral_theta))
        spiral_step = spiral_step + 2*np.pi / (1 + spiral_step**2)**0.5
    
    return np.array(pos_x), np.array(pos_y)

def defects_construct(shape, distance = 0):
    
    """
    Generate a binary mask with randomly distributed defects.
    
    Parameters:
        - shape (int): Size of the output mask (both width and height).
        - distance (float): Minimum distance between defects. 
          If 0, it's automatically set to shape / 50.
    
    Returns:
        - defect (numpy.ndarray): Binary mask with randomly distributed defects.
    """
    
    # construct the random geometry parameters

    tick = [np.arange(shape), np.arange(shape)]
    defects_2d = np.ones((shape, shape), dtype = float)
    if distance == 0: distance = shape / 50
    
    # construct the random defects
    
    for idx in range(int(shape // int(distance))**2):
    
        center = [np.random.rand() * shape, np.random.rand() * shape]
        sigma = [np.random.rand() * distance * 0.3, np.random.rand() * distance * 0.3]
        gauss_x, gauss_y = [
            np.exp(-1 * (tick[i] - center[i])**2 / sigma[i]**2) 
            for i in range(2)
            ]
        defects_2d += np.dot(gauss_x[:, np.newaxis], gauss_y[np.newaxis, :])
    
    defects_2d /= defects_2d.max()
    defect = np.ones((shape, shape), dtype = int)
    defect[defects_2d > 0.5] = 0
    
    return defect

#------------------------------------------------------
# standard sample

def siemens_star(shape = 2048, nb_rays = 72, nb_rings = 12, defect = False):
    
    """
    Construct Siemens star.
    
    Parameters:
        - shape (int): Size of the output image (both width and height).
        - nb_rays (int): Number of rays in the star.
        - nb_rings (int): Number of rings in the star.
        - defect (bool): Apply defects or not
    
    Returns:
        - star (numpy.ndarray): Binary array representing the Siemens star.
    """
    
    # construct siemen stars
    
    xgrid, ygrid = np.meshgrid(
        np.arange(-shape // 2, shape // 2, dtype=np.float32),
        np.arange(-shape // 2, shape // 2, dtype=np.float32)
        )
    radius_distance = np.sqrt(xgrid**2 + ygrid**2)
    
    star_construct = (np.arctan2(ygrid, xgrid) % (2*np.pi / nb_rays)) < (2*np.pi / nb_rays / 2)
    star_radius_limit = radius_distance < (shape * 0.75 // 2)
    star_rays = (
        (radius_distance % (shape * np.sqrt(2) / 2 / nb_rings)) < 
        ((shape * np.sqrt(2) / 2 / nb_rings) * 0.9)
        )
    star = star_construct * star_radius_limit * star_rays
    star[radius_distance < 2 * (shape * np.sqrt(2) / 2 / nb_rings)] = 0
    if defect: star = star * defects_construct(shape)
    
    return np.array(star, dtype = np.complex128) 

def random_mask(shape, element_size = 20, defect = False):
    
    """
    Generate a random binary mask.
    
    Parameters:
        - shape (int): Size of the output mask (both width and height).
        - element_size (int): Size of each element in the mask.
        - defect (bool): Apply defects or not
    
    Returns:
        - mask_interp (numpy.ndarray): Binary mask.
    """
    
    mask_shape = [int(shape / element_size), int(shape / element_size)]
    mask = np.random.randint(0, 10, size = mask_shape)
    mask[mask <= 5] = 0
    mask[mask >= 5] = 1
    
    from scipy.interpolate import RegularGridInterpolator
    func = RegularGridInterpolator(
        (np.linspace(0, mask_shape[0], mask_shape[0]), np.linspace(0, mask_shape[0], mask_shape[1])), 
        mask, bounds_error = False, fill_value = 0
        )
    interp_x, interp_y = np.meshgrid(
        np.linspace(0, mask_shape[0], shape), np.linspace(0, mask_shape[1], shape)
        )
    mask_interp = func((interp_x.flatten(), interp_y.flatten()))
    mask_interp[mask_interp >= 0.1] = 1
    mask_interp[mask_interp <= 0.1] = 0
    
    mask2d = mask_interp.reshape((shape, shape))
    if defect: 
        mask2d *= defects_construct(shape)
    
    return np.array(mask2d, dtype = np.complex128)

#------------------------------------------------------
# standard image

def standard_image(shape, over_sample_ratio = 4, name = "redpanda"):
    
    """
    Generate a standard image from a dataset with optional oversampling.
    
    Parameters:
        - shape (tuple): Size of the output image (rows, columns).
        - over_sample_ratio (int): Ratio of oversampling. If 0, no oversampling is performed.
        - name (str): Name of the image dataset.
    
    Returns:
        - image (numpy.ndarray): Standard image.
    """
    
    # parameter check and load the images
    
    dataset_path = "sample_dataset/%s.npy" % (name)
    
    if os.path.exists(dataset_path): 
        dataset = np.load(dataset_path)
    else:
        raise ValueError("Unsupported image name: {%s}" % (name))
    
    # oversampleing 
    
    xtick, ytick = [np.arange(count) - 0.5 * count for count in dataset.shape]
    xstart, xend = [-dataset.shape[0] / 2, dataset.shape[0] / 2]
    ystart, yend = [-dataset.shape[1] / 2, dataset.shape[1] / 2]
    
    from scipy.interpolate import RegularGridInterpolator
    func = RegularGridInterpolator(
        (xtick, ytick), dataset, bounds_error = False, fill_value = 0
        )
    
    if over_sample_ratio == 0:
        interp_x, interp_y = np.meshgrid(
            np.linspace(xstart, xend, shape[0]), np.linspace(ystart, yend, shape[1])
            )

    elif over_sample_ratio != 0:
        x_os_ratio = over_sample_ratio**0.5
        interp_x, interp_y = np.meshgrid(
            np.linspace(xstart * x_os_ratio, xend * x_os_ratio, shape[0]), 
            np.linspace(ystart * x_os_ratio, yend * x_os_ratio, shape[1])
            )
        
    image = func((interp_x.flatten(), interp_y.flatten()))
    
    return np.flipud(np.rot90(np.reshape(image, shape)))

#-----------------------------------------------------------------------------#
# classes

#-----------------------------------------------------------------------------#
# main