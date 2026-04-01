#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Tue Mar 26 17:43:19 2024"
__email__    = "xuhan@ihep.ac.cn"


"""
Description
"""

#-----------------------------------------------------------------------------#
# modules

import sys
import os

import numpy as np

from scipy import ndimage

#-----------------------------------------------------------------------------#
# parameters

# an example for a geometry dict
geometry_dict = {
    '1': [200, [0, 0, -1]], '2': [200, [0, 0, -1]], '3': [200, [0, -1, 0]], 
    '4': [200, [0, -1, 0]], '5': [200, [-1, 0, 0]], '6': [200, [-1, 0, 0]]
    }

#-----------------------------------------------------------------------------#
# functions

def euler_rotate(yaw = 0, pitch = 0, roll = 0, dimension = 3):
    
    """Returns the rotation matrix for a given yaw, pitch and roll angles.
    
    Args:
        yaw (float): The yaw angle in radians.
        pitch (float): The pitch angle in radians.
        roll (float): The roll angle in radians.
    
    Returns:
        numpy.ndarray: A 3x3 numpy array representing the rotation matrix.
    """
    
    # Define the sine and cosine of the angles
    
    sny, csy = [np.sin(yaw), np.cos(yaw)]
    snp, csp = [np.sin(pitch), np.cos(pitch)]
    snr, csr = [np.sin(roll), np.cos(roll)]
    
    # Calculate the rotation matrix using the Euler angles formula
    
    if dimension == 3:
        euler_matrix = np.array([
            [csy * csp, csy * snp * snr - sny * csr, csy * snp * csr + sny * snr],
            [sny * csp, sny * snp * snr + csy * csr, sny * snp * csr - csy * snr],
            [-snp, csp * snr, csp * csr]
            ])
    
    elif dimension == 2:
        euler_matrix = np.array([[csy, -sny], [sny, csy]])
    
    else:
        raise ValueError("Unsupported dimension: {%s}" % (str(dimension)))
    
    return euler_matrix

def matrix_point_index(matrix_shape):
    
    # parameter checking
    if isinstance(matrix_shape, (list, tuple)):
        if len(matrix_shape) == 3 or len(matrix_shape) == 2: pass
        else: raise ValueError("The matrix shape list should contain 2/3 elements")
    else: raise ValueError("The matrix shape should be a 2/3 elements list")
    
    # Calculate number of points and reciprocal space grid
    counts = np.prod(matrix_shape)
    index_range = [np.array(range(int(count))) for count in matrix_shape]
    
    # Reshape and stack reciprocal space grid
    func = lambda x: np.reshape(x, counts)
    
    if len(matrix_shape) == 3:
        index_x, index_y, index_z = np.meshgrid(
            index_range[0], index_range[1], index_range[2]
            )
        return np.vstack([func(index_x), func(index_y), func(index_z)])
    
    elif len(matrix_shape) == 2:
        index_x, index_y = np.meshgrid(index_range[0], index_range[1])
        return np.vstack([func(index_x), func(index_y)])

def matrix_point_distance(matrix_shape, center = None, mode = "point"):
    
    # parameter checking
    if isinstance(matrix_shape, (list, tuple)):
        if len(matrix_shape) == 3 or len(matrix_shape) == 2: pass
        else: raise ValueError("The matrix shape list should contain 2/3 elements")
    else: raise ValueError("The matrix shape should be a 2/3 elements list")
    
    # Calculate number of points and reciprocal space grid
    if center is None:
        index_range = [np.array(range(int(count))) - int(count/2) + 0.5 for count in matrix_shape]
    else:
        if isinstance(center, (list, tuple)):
            if len(matrix_shape) == 3 or len(matrix_shape) == 2: pass
            else: raise ValueError("The center list should contain 2/3 elements")
        else: raise ValueError("The center list should be a 2/3 element list")
        
        index_range = [
            np.array(range(int(matrix_shape[i]))) - center[i] 
            for i in range(len(matrix_shape))
            ]
    
    if len(matrix_shape) == 3:
        index_x, index_y, index_z = np.meshgrid(index_range[0], index_range[1], index_range[2])
        distance_matrix = np.sqrt(index_x**2 + index_y**2 + index_z**2)
        indexes = [index_x, index_y, index_z]
        
    elif len(matrix_shape) == 2:
        index_x, index_y = np.meshgrid(index_range[0], index_range[1])
        distance_matrix = np.sqrt(index_x**2 + index_y**2)
        indexes = [index_x, index_y]
        
        if mode == "point": return distance_matrix
        elif mode == "matrix": return indexes

#-----------------------------------------------------------------------------#
# classes

#---------------------------------------------------
# create a strain_field

class displacement(object):
    
    def __init__(self, shape, center):
        
        # Check the parameter of shape
        if isinstance(shape, (list, tuple)):
            shape_value_error = "The shape should be a 2/3 element list (3d or 2d size)"
            if len(shape) == 3 or len(shape) == 2: self.shape = shape
            else: raise ValueError(shape_value_error)
        else: raise ValueError(shape_value_error)
        
        self.distance_matrix = matrix_point_distance(shape, center = center)
        self.strain_field = np.ones(shape, dtype = float)
        
    def linear(self, strain = 1e-2):
        
        self.strain_field = self.distance_matrix * strain
        
#---------------------------------------------------
# create a polyhedron

class polyhedron(displacement):
    
    def __init__(
            self, shape, real_vectors = None, center = None, unit = 4, 
            yaw = 0, pitch = 0, roll = 0, strain = 0
            ):
        
        super().__init__(shape, center)
        
        # Check the parameter of shape
        
        shape_value_error = "The shape should be a 2/3 element list (3d or 2d size)"
        
        if isinstance(shape, (list, tuple)):
        
            self.dimension = len(shape)
            if self.dimension == 3 or self.dimension == 2: 
                self.shape = shape
                if self.dimension == 3: int(shape[0] * shape[1] * shape[2])
                elif self.dimension == 2: int(shape[0] * shape[1])
                
            else: raise ValueError(shape_value_error)
            
        else: raise ValueError(shape_value_error)
        
        # Direction matrix
        
        self.rotation_matrix = euler_rotate(yaw, pitch, roll, dimension = self.dimension)
        
        if real_vectors is None:
            if self.dimension == 3:
                self.real_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            elif self.dimension == 2:
                self.real_vectors = np.array([[1, 0], [0, 1]])
        else:
            self.real_vectors = real_vectors
        
        self.ub_matrix = np.matrix(np.eye(self.dimension))
        self.profile_matrix = np.ones(shape, dtype = int) 
        self.center = (np.array(self.shape) / 2).astype(int)
        self.matrix_index = matrix_point_index(self.shape)
        self.matrix_bool = np.ones(np.prod(self.shape), dtype = int)
        
        # The unit of the pixel
        self.unit = 1 if unit == None else int(unit)
        
        # # apply strain field
        # self.linear(strain = strain)
    
    #---------------------------------------------------
    # tools to create polygons
    
    def cut_plane(self, n, distance):
        
        # Shift the center from corner to center 
        
        recenter_matrix_index = np.array(
            [self.matrix_index[i, :] - self.shape[i] / 2 + 0.5 for i in range(self.dimension)], 
            dtype = int
            )
        
        # Bool the index accroding to the distance to the plane

        n = np.squeeze(np.asarray(np.matmul(
            self.ub_matrix, np.matmul(self.rotation_matrix, np.matrix(n).T)
            )))
        
        point2plane = np.squeeze(np.asarray(np.matmul(
            np.matrix(n), np.matrix(recenter_matrix_index))
            ))
        if self.dimension == 3:
            index_bool = np.zeros(self.shape[0] * self.shape[1] * self.shape[2])
        elif self.dimension == 2:
            index_bool = np.zeros(self.shape[0] * self.shape[1])
        index_bool[point2plane <= distance] = 1
        
        # Cut the profile matrix based the index bool
        
        profile_bool = tuple(np.array(
            [self.matrix_index[i, :] * index_bool for i in range(self.dimension)], 
            dtype = int).tolist())
        profile_matrix = np.zeros(self.shape, dtype = int)
        profile_matrix[profile_bool] = 1
        
        self.profile_matrix *= profile_matrix
        self.matrix_bool *= np.array(index_bool, dtype = int)
    
    def create_geometry(self, geometry_dict):
        
        # parameter check of geometry_dict
        if max([ivalue[0] for ivalue in geometry_dict.values()]) > min(self.shape):
            raise ValueError("The size of the goemtry should be smaller than primary shape")
        
        # create the geometry
        for icut in geometry_dict.values():
            distance, direction = icut
            direction = np.sum(np.array(
                [direction[i] * self.real_vectors[i] for i in range(self.dimension)]
                ), 0)
            n_direction = np.array(direction) / np.sum(np.abs(direction)**2)**0.5
            self.cut_plane(n_direction, distance)
        
    def cube(self, radius = None):
        
        if radius != None: 
            
            if self.dimension == 3:
                geometry_dict = {
                    '1': [int(radius), [ 0,  0,  1]], '2': [int(radius), [ 0,  0, -1]], 
                    '3': [int(radius), [ 0,  1,  0]], '4': [int(radius), [ 0, -1,  0]], 
                    '5': [int(radius), [ 1,  0,  0]], '6': [int(radius), [-1,  0,  0]]
                    }
            elif self.dimension == 2:
                geometry_dict = {
                    '1': [int(radius), [ 0,  1]], '2': [int(radius), [ 0, -1]], 
                    '3': [int(radius), [ 1,  0]], '4': [int(radius), [-1,  0]]
                    }
                
            # parameter check of geometry_dict
            if max([ivalue[0] for ivalue in geometry_dict.values()]) > min(self.shape):
                raise ValueError("The size of the goemtry should be smaller than primary shape")
            self.create_geometry(geometry_dict)
            
        else: pass
        
        # return self.profile_matrix
    
    def random_polygon(self, radius = False):
        
        # calcute r and fundmental cube
        
        radius = min(self.shape) // 4 if not radius else radius
        self.cube(radius)
        
        # construct the random geometry dict
        
        geometry_dict = dict()
        nb_cut = np.random.randint(1, 100)

        for idx in range(nb_cut):
            
            radius_n = np.random.randint(1, 3)
            i_radius = np.random.randint(
                min(self.shape) // int(8 * radius_n), min(self.shape) // int(4 * radius_n)
                )
            direction_x = np.random.randint(-20, 20)
            direction_y = np.random.randint(-20, 20)
            direction_z = np.random.randint(-20, 20)
            
            if self.dimension == 3:
                geometry_dict[str(idx)] = [i_radius, [direction_x, direction_y, direction_z]]
            elif self.dimension == 2:
                geometry_dict[str(idx)] = [i_radius, [direction_x, direction_y]]
        
        self.create_geometry(geometry_dict)
        centers = ndimage.measurements.center_of_mass(self.profile_matrix)
        
        for idx in range(self.dimension):
            self.profile_matrix = np.roll(
                self.profile_matrix, 
                int(self.shape[idx] // 2 - centers[idx]), axis = idx
                )
        
    #---------------------------------------------------
    # tools to create strains
    
    def random_strain(self, distance = 0, strain = 2 * np.pi):
        
        # construct the random geometry dict
        
        if distance == 0: distance = min(self.shape) / 50
        
        counts = (min(self.shape) // int(2 * distance))**2
        tick = [np.arange(self.shape[i]) for i in range(len(self.shape))]
        strain_nd = np.ones(self.shape, dtype = float)
        
        # construct the random geometry dict
        
        for idx in range(counts):
        
            center = [idx // 8 + int(np.random.rand() * idx // 2) for idx in self.shape]
            sigma = [
                np.random.randint(25, 60) / 60 * distance * 3 * np.random.randint(20, 30) / 30, 
                np.random.randint(25, 60) / 60 * distance * 3 * np.random.randint(20, 30) / 30,
                np.random.randint(25, 60) / 60 * distance * 3 * np.random.randint(20, 30) / 30
                ]
            
            if self.dimension == 3:
                gauss_x, gauss_y, gauss_z = [
                    np.exp(-1 * (tick[i] - center[i])**2 / sigma[i]**2) 
                    for i in range(3)
                    ]
                gauss_3d = gauss_x[:, None, None] * gauss_y[None, :, None] * gauss_z[None, None, :]
                strain_nd += np.random.rand() * gauss_3d
                
            elif self.dimension == 2:
                gauss_x, gauss_y = [
                    np.exp(-1 * (tick[i] - center[i])**2 / sigma[i]**2) 
                    for i in range(2)
                    ]
                gauss_2d = gauss_x[:, None] * gauss_y[None, :]
                strain_nd += np.random.rand() * gauss_2d
        
        if strain == 0:
            strain = 0.5 * np.pi * np.random.randint(20, 100) / 100
        
        strain_nd = strain_nd / np.max(strain_nd)
        strain_nd = strain_nd - np.mean(strain_nd)
        
        self.strain = strain_nd * 4 * np.pi
            
#---------------------------------------------------
# create a strain field within the nanoparticle

        
#-----------------------------------------------------------------------------#
# main