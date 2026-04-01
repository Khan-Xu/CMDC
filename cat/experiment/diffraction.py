# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Fri Apr 26 14:59:32 2024"
__email__    = "xuhan@ihep.ac.cn"


"""
Description: diffraction and unit_cell geometry
"""

#-----------------------------------------------------------------------------#
# modules

import os
import sys
import re

import numpy as np
import h5py as h5

from cat.experiment import constants

#-----------------------------------------------------------------------------#
# parameters

na = 6.022e23 / 1e23
r0 = 2.82e-5

# an example of element_dict 
# element_dict = {
#     'Sr2+': [[1/8, [0.0, 0.0, 0.0]], [1/8, [1.0, 0.0, 0.0]],
#              [1/8, [0.0, 1.0, 0.0]], [1/8, [1.0, 1.0, 0.0]],
#              [1/8, [0.0, 0.0, 1.0]], [1/8, [1.0, 0.0, 1.0]],
#              [1/8, [0.0, 1.0, 1.0]], [1/8, [1.0, 1.0, 1.0]]],
#     'Ti4+': [[1/1, [0.5, 0.5, 0.5]]], 
#     'O2-' : [[1/2, [0.5, 0.5, 0.0]], [1/2, [0.5, 0.5, 1.0]],
#              [1/2, [0.5, 0.0, 0.5]], [1/2, [0.0, 0.5, 0.5]],
#              [1/2, [0.5, 1.0, 0.5]], [1/2, [1.0, 0.5, 0.5]]]
#     }

# an example of geometry_dict 
# geometry_dict = {'n': [200, [0, 0, 1]], 'x': [200, [1, 0, 0]], 'y': [200, [0, 1, 0]]}
   
#-----------------------------------------------------------------------------#
# functions

def lattice_vectors(lattice = None, mode = 'r'):
    
    """Calculates the real or reciprocal lattice vectors for a given lattice.
    
    Args:
        lattice (list): A list of  lattice parameters [a, b, c, α, β, γ].
        mode (str): A string indicating whether to return the real ('r') or reciprocal ('q') lattice vectors.
    
    Returns:
        list: A list of lattice vectors reciprocal lattice vector.
    
    Raises:
        ValueError: If the input is not a list of six numbers or if the mode is not 'r' or 'q'.
    """

    # Check if the input is valid
    if len(lattice) != 6 or not isinstance(lattice, list):
        raise ValueError("Input should be a list contain [a, b, c, α, β, γ]")
        
    else:
        # Extract the lattice parameters
        a, b, c, alpha, beta, gamma = lattice
        
        # Calculate the real space vectors using trigonometry
        vector_a = a * np.array([1, 0, 0])
        vector_b = b * np.array([np.cos(gamma), np.sin(gamma), 0])
        vector_c = c / np.sin(gamma) * np.array([
            np.cos(beta) * np.sin(gamma), np.cos(alpha) - np.cos(beta) * np.cos(gamma),
            np.sqrt(np.sin(gamma)**2 - np.cos(alpha)**2 - np.cos(beta)**2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
            ]) 
        
        # calcuate the reciprocal space vectors using cross product and volume formula
        volume = a * b * c * np.sqrt(
            1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) -
            np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
            )
        q_vector_a = np.cross(vector_b, vector_c) / volume
        q_vector_b = np.cross(vector_c, vector_a) / volume
        q_vector_c = np.cross(vector_a, vector_b) / volume
    
    # Check if the mode is valid and return the corresponding vectors
    if mode == 'r': 
        return [vector_a, vector_b, vector_c]
    elif mode == 'q': 
        return [q_vector_a, q_vector_b, q_vector_c]

def euler_rotate(yaw = 0, pitch = 0, roll = 0):
    
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
    euler_matrix = np.array([
        [csy * csp, csy * snp * snr - sny * csr, csy * snp * csr + sny * snr],
        [sny * csp, sny * snp * snr + csy * csr, sny * snp * csr - csy * snr],
        [-snp, csp * snr, csp * csr]
        ])
    
    return euler_matrix

def calculate_ub_matrix(lattice_parameter, new_axes):
    
    trans_miller1_orth = lattice_vectors(lattice_parameter)
    trans_miller2_orth = [
        sum([new_axes[k][i] * trans_miller1_orth[i] for i in range(3)]) for k in range(3)
        ]
    
    # normalize the direction vectors
    f_norm = lambda x: np.array(x) / np.sum(np.abs(np.array(x))**2)**0.5
    ub_miller1_orth = np.matrix([f_norm(trans_miller1_orth[i]) for i in range(3)])
    ub_miller2_orth = np.matrix([f_norm(trans_miller2_orth[i]) for i in range(3)])
    
    ub_matrix = np.matmul(np.linalg.inv(ub_miller2_orth), ub_miller1_orth)

    return ub_matrix

def define_reciprocal_space(
        q_range = [-0.5, 0.5, -0.5, 0.5, 1.8, 2.2], points = [300, 300, 300],
        scale = [3.905, 3.905, 3.905]
        ):
    
    """
    Defines the reciprocal space of a crystal as a grid of points in three dimensions.
    
    Args:
        q_range: A list of six values that define the minimum and maximum
            values of the three reciprocal space coordinates (qx, qy, qz). 
        points: A list of three integers that define the number of points in each
            direction of the reciprocal space. Default is [300, 300, 300].
    
    Returns:
        np.ndarray: A two-dimensional array of shape (n_points, 3), where n_points is the total
        number of points in the reciprocal space grid. Each row of the array represents a point
        in the reciprocal space, with the three columns corresponding to the qx, qy, and qz
        coordinates, respectively.
    
    Raises:
        ValueError: If q_range is not a list of six values, or if points is not a
        list of three integers.
    """

    # check the input variables
    if not isinstance(q_range, list):
        raise ValueError("q range should be None or a list")
    else:
        if len(q_range) != 6:
            raise ValueError("q range should be [x_min, x_max, y_min, y_max, z_min, z_max]")
            
    if not isinstance(points, list): 
        raise ValueError("points should be a list")
    else:
        if len(points) != 3: raise ValueError("points should be [nx, ny, nz]")
    
    # Calculate number of points and reciprocal space grid
    counts = points[0] * points[1] * points[2]
    q_array = [np.linspace(q_range[2 * i], q_range[1 + i * 2], points[i]) for i in range(3)]
    
    # Reshape and stack reciprocal space grid
    f_reshape = lambda x: 2 * np.pi * np.reshape(x, counts)
    q_mesh = np.meshgrid(q_array[0], q_array[1], q_array[2])
    q_space = np.vstack([f_reshape(q_mesh[i]) / scale[i] for i in range(3)])
    
    return q_space 
        
#-----------------------------------------------------------------------------#
# classes

#---------------------------------------------------
# the baisc funcation of unit cell

class unit_cell(object):
    
    def __init__(
            self, element_dict, lattice = [3.905, 3.905, 3905, np.pi/2, np.pi/2, np.pi],
            axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
            yaw = 0, pitch = 0, roll = 0, energy = 12.398
            ):

        # the elements in the unit cell
        self.element_dict = element_dict
        self.element_list = list()
        self.element_counts = list()
        
        template_ion = re.compile("([a-zA-Z]+)([0-9]+)")
        template_atom = re.compile("([a-zA-Z]+)")
        
        for element in self.element_dict.keys():
        
            match_ion = template_ion.match(element)
            match_atom = template_atom.match(element)
            if match_ion is None: self.element_list.append(match_atom.groups()[0])
            else: self.element_list.append(match_ion.groups()[0])
            self.element_counts.append(sum([icount[0] for icount in element_dict[element]]))
        
        # the real and reciprocal vectors
        self.lattice = lattice
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = lattice
        self.vectors = lattice_vectors(lattice, mode = 'r')
        self.q_vectors = lattice_vectors(lattice, mode = 'q')
        
        # x-ray unit Å (real space) or 1/Å (reciprocal space)
        self.energy = energy
        self.wavelength = 12.398 / self.energy 
        self.k = 2 * np.pi / self.wavelength # unit 1/Å
        
        # the reciprocal space rotation
        # first, check the axes of the unit cell
        if np.array(axes).shape != (3, 3):
            raise ValueError("axes: a list contains three orthogonal directions (also list)")
        else:
            self.axes = np.array(axes)
            axis_x, axis_y, axis_z = axes
            if (np.prod(np.array(axis_x) * np.array(axis_y)) != 0 or 
                np.prod(np.array(axis_y) * np.array(axis_z)) != 0 or
                np.prod(np.array(axis_x) * np.array(axis_z)) != 0):
                raise ValueError("The three axes should be orthogonal")
                
            else: self.ub_matrix = calculate_ub_matrix(lattice, axes)

        self.rotation = euler_rotate(yaw, pitch, roll)
        
    def q_position(self, yaw = 0, pitch = 0, roll = 0):
        
        """Calculates the position of the bragg peaks after applying a rotation.
        
        Args:
            yaw (float): The yaw angle in radians.
            pitch (float): The pitch angle in radians.
            roll (float): The roll angle in radians.
        
        Returns:
            None: Updates the position attribute of the self object.
        
        Attributes:
            peak_index (str or list): The index of the peak to calculate the position for. 
            q_vectors (list): [q_vector_a, q_vector_b, q_vector_c].
            position (numpy.ndarray): A the position of the q vectors after rotation.
        """

        rotation = euler_rotate(yaw, pitch, roll)
        
        if self.peak_index == 'all':
            self.position = np.zeros([int(7**3), 3])
            self.position = np.array([
                [ix * self.q_vectors[0], iy * self.q_vectors[1], iz * self.q_vectors[2]] 
                for ix in range(7) for iy in range(7) for iz in range(7)
                ])
        elif isinstance(self.peak_index, list) and len(self.peak_index) == 3:
            self.position = np.array(
                [self.peak_index[i] * self.q_vectors[i]] for i in range(3)
                )
        
        self.position = np.mat(np.matrix(self.position).T, rotation)
        
        return self.position
    
    def diffraction_angle(self, out_of_plane = [0, 0, 1], bragg_index = [0, 0, 2]):
        
        """
        Calculates the diffraction angle for a given crystal structure.
        
        Parameters:
        out_of_plane (list): A list containing the out of plane vector components.
        bragg_index (list): A list containing the Bragg index vector components.
        
        Returns:
        list: A list containing two angles theta and delta.
        """

        g_vector_bragg = np.squeeze(np.asarray(
            np.matmul(np.matrix(bragg_index), np.matrix(self.q_vectors))
            ))
        g_vector_direction = np.squeeze(np.asarray(
            np.matmul(np.matrix(out_of_plane), np.matrix(self.q_vectors))
            ))
        
        self.lattice_distance = 1 / np.sum(np.abs(g_vector_bragg)**2)**0.5
        theta_angle = np.arcsin(self.wavelength / (2 * self.lattice_distance))
        intersect_angle = np.arccos(
            np.sum(g_vector_bragg * g_vector_direction) / 
            (np.sum(np.abs(g_vector_bragg)**2)**0.5 * np.sum(np.abs(g_vector_direction)**2)**0.5)
            )
        
        self.theta = [theta_angle - intersect_angle, theta_angle + intersect_angle]
        self.delta = theta_angle * 2 + intersect_angle        
        
        return [self.theta, self.delta]
        
    def structure_factor(self, q = None):
        
        """Calculates the structure factor of the unit cell for a given wave vector.

        Args:
            q (list): The wave vector in reciprocal space.
        
        Returns:
            complex: The structure factor as a complex number.
        """
        
        # Calcualte the absolute q_space
        q_space = np.matmul(self.ub_matrix, np.matmul(self.rotation, np.matrix(q)))
        
        # The absolute q for atomic form factor calculate
        abs_q_space = np.sqrt(np.sum(np.abs(np.array(q_space)**2), axis = 0))
        f_atomic_form_f = lambda a, b, abs_q: a * np.exp(-b * abs_q**2 / (4 * np.pi)**2)
        uc_form_factor = 0
        
        for element in self.element_dict.keys():
            
            # Get the atomic form factor coefficients for the element
            
            atomic_form_f = constants.atomic_form_factor(ions = element)
            atomic_form_f_q_space = sum(
                [f_atomic_form_f(atomic_form_f[i], atomic_form_f[i + 4], abs_q_space) for i in range(4)]
                ) + atomic_form_f[8]
            
            # Calculate the phase factor for the element position
            for position in self.element_dict[element]:
                
                ratio, position = position
                vector = np.sum(np.array([position[i] * self.vectors[i] for i in range(3)]), 0)
                phase = np.squeeze(np.asarray(np.matmul(np.matrix(vector), q_space)))
                uc_form_factor += ratio * atomic_form_f_q_space * np.exp(1j * phase)
        
        return uc_form_factor
    
    def crystal_factor(self, q = None, size = 2e6):
        
        """Calculates the structure factor of the unit cell for a given wave vector.

        Args:
            q (list): The wave vector in reciprocal space.
        
        Returns:
            complex: The structure factor as a complex number.
        """
        
        # q_space calculation
        q_space = np.matmul(self.ub_matrix, np.matmul(self.rotation, np.matrix(q)))
           
        calculate_crystal_factor = (
            lambda d: (1 - np.exp(1j * size * np.matmul(np.matrix(d), q_space)) + 1e-6) / 
            (1 - np.exp(1j * np.matmul(np.matrix(d), q_space)) + 1e-6)
            )
        crystal_structure_factor = (
            np.squeeze(np.asarray(calculate_crystal_factor(self.vectors[0]))) * 
            np.squeeze(np.asarray(calculate_crystal_factor(self.vectors[1]))) *
            np.squeeze(np.asarray(calculate_crystal_factor(self.vectors[2])))
            )
        uc_structure_factor = self.structure_factor(q)
        
        return crystal_structure_factor * uc_structure_factor
    
    def absorption(self, distance):
        
        element_mass = list()
        asf2_list = list()
        
        for el in self.element_list:
            element_mass.append(constants.atomic_mass(element = el))
            asf2_list.append(constants.atomic_scattering_factor(element = el, energy = self.energy)[2])
            
        attunuation = constants.attenuation_coefficient(
            element_mass, self.element_counts, asf2_list, self.energy
            )
        
        # return attunuation
        return np.exp(distance * 1e-7 / attunuation)
    
#---------------------------------------------------
# the layer model, for CTR calculation

class layer(unit_cell):
    
    """Calculates the structure factor of a single layer or a truncation surface.

    This class calculates the structure factor of a single layer or a truncation surface.
    It inherits from the 'unit_cell' class.
    
    Attributes:
        element_dict (dict): A dictionary contains atomic symbol and fractional coordinates.
        lattice (list): A list contains the lattice parameters and angles.
        direction (str): A string indicates the out-of-plane direction.
        n_layer (int): The number of layers or truncation surface.
        energy (float): The energy of the incident beam in keV.
    
    Methods:
        layer_structure_factor(q = None):
        Calculates the structure factor of a single layer or a truncation surface.
    """

    def __init__(
            self, element_dict, lattice = [3.905, 3.905, 3,905, np.pi/2, np.pi/2, np.pi/2], 
            direction = [0, 0, 1], n_layer = None, energy = 12.398
            ):
                
        super().__init__(
            element_dict, lattice = lattice, axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
            energy = energy
            )
        self.n_layer = n_layer
    
        # Check the parameter of direction
        direction_value_error = "The direction should be a three-element list (miller index)"
        if isinstance(direction, list):
            if len(direction) == 3: self.direction = direction
            else: raise ValueError(direction_value_error)
        else: raise ValueError(direction_value_error)
    
    def layer_structure_factor(self, q = None):
        
        """Calculates the structure factor of a single layer or a truncation surface.

        Args: q (ndarray): 
            A 3-element ndarray contains the q-values in the x, y, and z directions.
        Returns: 
            complex: The calculated structure factor of a single layer or a truncation surface.
        """
        
        # Calcualte distance base on the miller index of out-of-plane
        g_vector = np.squeeze(
            np.asarray(np.matmul(np.matrix(self.direction), np.matrix(self.q_vectors)))
            )
        distance = 1 / np.sum(np.abs(g_vector)**2)**0.5
        absorption_factor = self.absorption(distance)
        
        # Calcualte q space after oritentation setup
        q_space = np.matmul(self.ub_matrix, np.matmul(self.rotation, np.matrix(q)))
        self.phase = np.matmul(np.matrix([0, 0, distance]), q_space)
        
        # n_layer is None, represenet a truncation surface
        if self.n_layer == None:
            ctr_factor = 1 / (1 - absorption_factor * np.exp(1j * self.phase) + 1e-6)
        
        # thin layers
        elif isinstance(self.n_layer, (int, float)):
            ctr_factor = (
                (1 - absorption_factor**int(self.n_layer) * np.exp(1j * int(self.n_layer) * self.phase)) /
                (1 - absorption_factor * np.exp(1j * self.phase))
                ) + 1e-6
    
        else: raise ValueError("the number of layer should be an int")
        
        return np.squeeze(np.asarray(ctr_factor))  * self.structure_factor(q)
            
#---------------------------------------------------
# the particle model

# An accurate model for calculating the form factor of a particle, quite slow...., for small particle
class particle_accurate(unit_cell):
    
    def __init__(
            self, element_dict, lattice = [3.905, 3.905, 3.905, np.pi/2, np.pi/2, np.pi/2],
            axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], yaw = 0, pitch = 0, roll = 0, energy = 12.398
            ):
        
        super().__init__(
                element_dict, lattice = lattice, axes = axes, 
                yaw = yaw, pitch = pitch, roll = roll, energy = energy
                )

    def domain_structure_factor(self, geometry_class, q = [0, 0, 0]):
        
        # Create the real space and q space
        domain_factor = np.zeros(q.shape[0], dtype = complex)
        q_space = np.matmul(self.ub_matrix, np.matmul(self.rotation, np.matrix(q)))
        
        # The slice plane for the iteration
        from tqdm import tqdm
        counts = np.prod(self.size)
        if counts <= 50:
            
            for ipoint in tqdm(range(int(counts))):
                phase = np.squeeze(np.asarray(np.matmul(np.matmul(
                    np.matrix(geometry_class.matrix_index).T, np.matrix(self.vectors)
                    ), q[ipoint])))
                domain_factor[ipoint] = np.sum(geometry_class.matrix_bool * np.exp(1j * phase))
        else:
            
            n_iteration = counts // 50 + int(bool(counts % 100))
            
            for ipoint in tqdm(range(int(n_iteration))):
                if ipoint != int(n_iteration - 1): start, end = [int(50 * ipoint), int(50 * ipoint + 50)]
                elif ipoint == int(n_iteration - 1): start, end = [int(50 * ipoint), -1]
                    
                phase = np.squeeze(np.asarray(np.matmul(np.matmul(
                    np.matrix(geometry_class.matrix_index).T, np.matrix(self.vectors).T
                    ), np.matrix(q_space[start : end, :]))))
                domain_factor[start : end] = np.sum(
                    geometry_class.matrix_bool[:, np.newaxis] * np.exp(1j * phase), 0
                    )
                
        return domain_factor * self.structure_factor(q)
    
# This model calculate the structure factor of a particle with FFT
class particle(unit_cell):
    
    def __init__(
            self, element_dict, lattice = [3.905, 3.905, 3.905, np.pi/2, np.pi/2, np.pi/2],
            axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
            yaw = 0, pitch = 0, roll = 0, energy = 12.398
            ):
        
        super().__init__(
                element_dict, lattice = lattice, axes = axes, 
                yaw = yaw, pitch = pitch, roll = roll, energy = energy
                )
        
    def domain_structure_factor(self, geometry_class, bragg_index = [0, 0, 2]):
        
        # Calcuate the structure factor of the particle
        q_index = np.sum(
            np.array([bragg_index[i] * self.q_vectors[i] for i in range(3)]), axis = 0
            )
        q_index_u = np.matmul(self.ub_matrix, np.matmul(self.rotation, np.matrix(q_index).T))
        real_vector = np.matmul(
            np.matrix(geometry_class.matrix_index).T, 
            np.matmul(geometry_class.ub_matrix, np.matrix(self.vectors).T)
            )
        
        crystal_phase = np.squeeze(np.asarray(np.matmul(real_vector, q_index_u)))
        structure_factor = self.structure_factor(np.matrix(q_index).T)
        crystal_factor = np.sum(np.exp(1j * crystal_phase)) * structure_factor
        
        # Calculate the profile factor
        # Warning! Why I multiply np.exp(1j * 1e-6) to the fft result? Because the small value of 
        #          phase will transform to pi or -pi after np.angle
        profile_factor = np.fft.ifftshift(
            np.fft.fftn(np.fft.fftshift(geometry_class.profile_matrix))
            ) * np.exp(1j * 1e-6)
        domain_factor = profile_factor * crystal_factor
        
        # Calcuate the corresponding q range
        
        q_start = [-0.5 / lattice / geometry_class.ratio for lattice in self.lattice[0 : 3]]
        q_spacing = [q_start[i] * -2 / geometry_class.shape[i] for i in range(3)]
        q_start = np.array(q_start) + q_index_u
        
        specified_ub_matrix = np.array(
            [np.abs(np.array(self.axes[i])) / np.sum(np.array(self.axes[i])**2)**0.5 
             for i in range(3)]
            )
    
        return [q_start, q_spacing, domain_factor]

#---------------------------------------------------
# the diffraction geometry class

class diffraction_geometry_4c(object):
    
    def __init__(self, diffraction_geometry_dict):
        
        # load the parameters from diffraction geometry dict
        
        self.diffraction_geometry_dict = diffraction_geometry_dict
        
        for key in diffraction_geometry_dict.keys():
            setattr(self, key, diffraction_geometry_dict[key])
        
        # calculate parameters 
        
        self.pixel_radius = self.pixel_size / self.distance
        self.frame_number = self.theta_scan.shape[0]
        self.scan_step = (self.theta_scan[-1] - self.theta_scan[0]) / self.frame_number
        self.step_constant = self.scan_step / self.pixel_radius
        self.intersect = self.delta - self.theta
        self.k_vector = 2 * np.pi / self.wave_length
        
        # calculate rotation matrix
        
        self.delta_matrix = np.array([
            [np.cos(self.delta), 0, np.sin(self.delta)], [0, 1, 0], 
            [-1 * np.sin(self.delta), 0, np.cos(self.delta)]
            ])
        
        # the scale between detector pixel size and scan
        
        self.rebin_factor = np.abs(
            2.0 * np.sin(self.delta / 2.0) * self.step_constant
            )
        self.rsm_unit = self.k_vector * self.pixel_radius
        
    def calculate_q_position(self):
       
        # a function transform the position of a vector from detector to q
       
        def cal_q_vector(scan_index, detector_x, detector_y):
            
            q_vector = (
                np.dot(self.delta_matrix, np.array([detector_y, detector_x, 1/self.pixel_radius])) - 
                np.array([0, 0, 1/self.pixel_radius])
                )
            theta_scan = self.theta_scan[scan_index]
            theta_matrix = np.array([
                [np.cos(theta_scan), 0, -np.sin(theta_scan)], 
                [0, 1, 0], [np.sin(theta_scan), 0, np.cos(theta_scan)]
                ])
            
            return np.dot(theta_matrix, q_vector)
       
        # the q position of the eight corners calculation
        
        detector_boundary_x = [
            self.detector_size[0]/2, self.detector_size[0]/2, 
            -self.detector_size[0]/2, -self.detector_size[0]/2
            ]
        detector_boundary_y = [
            self.detector_size[0]/2, -self.detector_size[0]/2, 
            self.detector_size[0]/2, -self.detector_size[0]/2
            ]
        q_vector_boundary = list()
        
        for bound_idx in range(4):
            for scan_idx in [0, -1]:
                q_vector_boundary.append(cal_q_vector(
                    scan_idx, detector_boundary_x[bound_idx], detector_boundary_y[bound_idx]
                    ))
        self.q_vector_boundary = np.array(q_vector_boundary)
        self.q_vector_limit = np.array([
            [self.q_vector_boundary[:, idx].max(), self.q_vector_boundary[:, idx].min()] 
            for idx in range(3)]) * self.rsm_unit
        
        return self.q_vector_limit
    
    def calculate_transform_matrix(self):
        
        flip_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        
        # the direction along theta scan
        
        scan_x = (np.cos(self.theta) - np.cos(self.intersect)) * self.step_constant
        scan_z = (np.sin(self.theta) + np.sin(self.intersect)) * self.step_constant 
        direction_theta_scan = [scan_x, 0, scan_z]
        
        # the direction along detector
        
        direction_detector_out_of_plane = [np.cos(self.intersect), 0, -np.sin(self.intersect)]
        direction_detector_in_plane = [0, 1, 0]
        
        # the omega rocking matrix
        
        self.rocking_matrix = np.array([
            direction_theta_scan, 
            direction_detector_out_of_plane, direction_detector_in_plane]
            ).T
        self.affine_matrix = np.dot(self.rocking_matrix, flip_matrix)
        
        return self.affine_matrix
    
    def affine_transform_factor(self):
        
        # calculate the transform matrix and q positions
        
        self.calculate_transform_matrix()
        self.calculate_q_position()
        
        # parameters for affine_transform
        
        self.rsm_shape = np.array(
            np.ptp(self.q_vector_limit, axis = 1) / self.rsm_unit / self.rebin_factor, dtype = int
            )
        self.inv_rocking_matrix = np.linalg.inv(self.rocking_matrix)
        self.affine_offset = (
            np.array(self.detector_size + [self.frame_number]) / 2.0 - 
            np.dot(self.inv_rocking_matrix, self.rsm_shape.astype(float) / 2.0)
            )

#---------------------------------------------------
# bragg cdi experiment

#-----------------------------------------------------------------------------#
# main

