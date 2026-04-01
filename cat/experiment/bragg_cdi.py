# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Mon May  6 16:58:11 2024"
__email__    = "xuhan@ihep.ac.cn"


"""
Description
"""

#-----------------------------------------------------------------------------#
# modules

import sys
import os

import numpy as np
import scipy as sp

from skimage.restoration import unwrap_phase
from scipy import interpolate
from cat.experiment import diffraction, polygon

#-----------------------------------------------------------------------------#
# parameters

#---------------------------------------------------
# An example of the parameters for the bragg cdi experiment

#------------------------------------------------------------------------
# material parameters

au_element_dict = {
    'Au': [[1/8, [0.0, 0.0, 0.0]], [1/8, [1.0, 0.0, 0.0]],
           [1/8, [0.0, 1.0, 0.0]], [1/8, [1.0, 1.0, 0.0]],
           [1/8, [0.0, 0.0, 1.0]], [1/8, [1.0, 0.0, 1.0]],
           [1/8, [0.0, 1.0, 1.0]], [1/8, [1.0, 1.0, 1.0]],
           [1/2, [0.5, 0.5, 0.0]], [1/2, [0.5, 0.5, 1.0]],
           [1/2, [0.5, 0.0, 0.5]], [1/2, [0.0, 0.5, 0.5]],
           [1/2, [0.5, 1.0, 0.5]], [1/2, [1.0, 0.5, 0.5]]
           ]
    }
au_lattice = [4.078, 4.078, 4.078, np.pi/2, np.pi/2, np.pi/2]
au = diffraction.unit_cell(au_element_dict, lattice = au_lattice, energy = 12.398)
bragg_index = [0, 0, 2]

# bfo_element_dict = {
#     'Bi3+': [[1/1, [ 0.5,  0.5,  0.5]]],
#     'Fe3+': [[1/1, [ 0.0,  0.0,  0.0]]],
#     'O2-' : [[1/1, [ 0.0,  0.5,  0.5]], [1/1, [ 0.5,  0.5,  0.0]], [1/1, [ 0.5,  0.0,  0.5]]]
#     }
# bfo_lattice = [3.960, 3.960, 3.960, np.pi/2, np.pi/2, np.pi/2]
# bfo = diffraction.unit_cell(au_element_dict, lattice = au_lattice, energy = 12.398)
# bragg_index = [0, 0, 2]

# geometry parameters

geometry_dict = {
    '1':  [30, [ 0,  0, -1]], '2':  [30, [ 0,  0,  1]], '3':  [30, [ 1,  1,  0]], 
    '4':  [30, [-1, -1,  0]], '5':  [30, [ 0,  1,  0]], '6':  [30, [ 0, -1,  0]],
    '7':  [30, [ 1,  0,  0]], '8':  [30, [-1,  0,  0]], '9':  [30, [ 1, -1,  0]], 
    '10': [30, [-1,  1,  0]], '11': [30, [ 1,  1,  1]], '12': [30, [-1, -1,  1]],
    '13': [30, [-1,  1,  1]], '14': [30, [ 1, -1,  1]], '15': [30, [ 1,  1, -1]], 
    '16': [30, [-1, -1, -1]], '17': [30, [-1,  1, -1]], '18': [30, [ 1, -1, -1]]
    }
geometry_shape = [256, 256, 256]
geometry_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
crystal_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# experiment parameters

diffraction_geometry_dict = {
    "wave_length": 1.0, "theta_range": 0.6, "frame_number": 256,
    "bragg_index": [0, 0, 2], "pixel_size": 75e-6,
    "distance": 1.83, "detector_size": [256, 256]
    }
sample_geometry_dict = {
    "lattice_plane": geometry_dict, "sample_size": [256, 256, 256],
    "center": [128, 128, 128], "oversampling": 5,
    "rotation": [0, 0, 0] # yaw, pitch, roll
    }
sample_material_dict = {
    "sample_element": au_element_dict, "lattice": au_lattice, 
    "energy": 12.398 # keV
    }

#------------------------------------------------------------------------

#-----------------------------------------------------------------------------#
# functions

def _locate(ticks, value):
    
    """
    This function searches through an array of 'ticks' and returns the index of the entry
    that is closest to the specified 'value'. If the 'value' is outside the range of 'ticks',
    a ValueError is raised.
    
    Args:
        ticks (np.ndarray): An array of numerical values.
        value (float): The value to locate within 'ticks'.
    
    Raises:
        ValueError: If 'value' is outside the range of 'ticks'.
    
    Returns:
        int: The index of the closest entry in 'ticks' to 'value'.
    """

    if value > np.max(ticks) or value < np.min(ticks):
        raise ValueError("Unsupported value range: {}".format(value))
    else:
        return np.argmin(np.abs(ticks - value)) 

def interpolate_sample_dataset(sample_dataset, ratio):
    
    """
    Interpolates a given sample dataset to increase its resolution by a specified ratio.
    
    Parameters:
    - sample_dataset (ndarray): The input sample dataset to be interpolated.
    - ratio (float): The interpolation ratio by which to increase the resolution.
    
    Returns:
    - ndarray: The interpolated sample dataset with increased resolution.
    
    Notes:
    - This function performs linear interpolation using RegularGridInterpolator from scipy.interpolate.
    - The input sample_dataset is assumed to be a 3D numpy array.
    - Interpolation is performed along each dimension independently.
    - The phase of the input sample dataset is unwrapped before interpolation to avoid phase wrapping issues.
    - The interpolated sample dataset is computed as the product of interpolated magnitudes and
      original unwrapped phase values.
    
    Example:
    >>> import numpy as np
    >>> from scipy.interpolate import interpolate
    >>> sample_dataset = np.random.rand(10, 10, 10)  # Example 3D sample dataset
    >>> interpolated_dataset = interpolate_sample_dataset(sample_dataset, 2.0)
    """

    sample_shape = sample_dataset.shape
    new_shape = np.array(np.array(sample_shape) * ratio, dtype = int)
    counts = new_shape[0] * new_shape[1] * new_shape[2]
    
    xtick0, ytick0, ztick0 = [np.arange(sample_shape[idx]) for idx in range(3)]
    xtick1, ytick1, ztick1 = [
        np.linspace(0, sample_shape[idx], int(sample_shape[idx] * ratio)) 
        for idx in range(3)
        ]
    
    points = np.meshgrid(xtick1, ytick1, ztick1, indexing = 'xy')
    points = (
        np.reshape(points[0], (1, counts)), np.reshape(points[1], (1, counts)), 
        np.reshape(points[2], (1, counts))
        )
    unwraped_phase = unwrap_phase(np.angle(sample_dataset))
    func_abs = interpolate.RegularGridInterpolator(
        (xtick0, ytick0, ztick0), np.abs(sample_dataset), method = "nearest",
        bounds_error = False, fill_value = 0
        )
    func_angle = interpolate.RegularGridInterpolator(
        (xtick0, ytick0, ztick0), unwraped_phase, method = "nearest", 
        bounds_error = False, fill_value = 0
        )
    
    interpolated_sample_dataset = np.reshape(
        func_abs(points) * np.exp(1j * func_angle(points)), new_shape
        )
    
    return interpolated_sample_dataset
            
#-----------------------------------------------------------------------------#
# classes

class bragg_cdi_experiment(object):
    
    def __init__(self, sample_material, sample_geometry, diffraction_geometry):
        
        # load the paramters of sample material, nanoisland geometry, and diffraction geometry
        
        self.material_dict = sample_material
        self.geometry_dict = sample_geometry
        self.diffraction_dict = diffraction_geometry
        
        # the construction of sample material
        
        self.material_class = diffraction.unit_cell(
            self.material_dict["sample_element"], lattice = self.material_dict["lattice"], 
            energy = self.material_dict["energy"]
            )
        
        # the construction of nanoisland geometry
        
        self.geometry_class = polygon.polyhedron(
            self.geometry_dict["sample_size"], self.material_class.vectors, 
            center = self.geometry_dict["center"], unit = self.geometry_dict["oversampling"], 
            yaw = self.geometry_dict["rotation"][0], pitch = self.geometry_dict["rotation"][1], 
            roll = self.geometry_dict["rotation"][2], strain = 0
            )
        # self.geometry_class.create_geometry(self.geometry_dict["lattice_plane"])
        
        # the construction of diffraction geometry
        
        theta, delta = self.material_class.diffraction_angle(
            bragg_index = self.diffraction_dict["bragg_index"]
            )
        self.diffraction_dict["theta"] = theta[0]
        self.diffraction_dict["delta"] = delta
        self.diffraction_dict["theta_scan"] = np.linspace(
            theta[0] - np.deg2rad(self.diffraction_dict["theta_range"]/2),
            theta[0] + np.deg2rad(self.diffraction_dict["theta_range"]/2),
            int(self.diffraction_dict["frame_number"])
            )
        self.diffraction_class = diffraction.diffraction_geometry_4c(
            self.diffraction_dict
            )
    #---------------------------------------------------
    # the baisc funcation of experiment
    
    def calculate_q_space(self):
        
        """
        Calculate the q space covered by the experiment considering oversampling.
        
        This function calculates the q space covered by the experiment, taking into account oversampling.
        It first computes the q space covered by the experiment based on the diffraction class.
        Then, it determines the q space covered by oversampling, adjusting the oversampling ratio if necessary.
        The oversampling ratio is adjusted iteratively until the scan q space covers the oversampled q space.
        Finally, it sets the q index, start, and end values for the oversampled q space.
        
        Returns:
            None
        
        Raises:
            None
        """

        # the q space covered by the experiment
        
        scan_q_space = np.flipud(np.rot90(self.diffraction_class.calculate_q_position(), -1))
        
        # the q space covered by oversampling
        
        def oversampling_q_space(unit):
            
            q_start = np.array([-0.5 / self.material_class.lattice[idx] / unit for idx in range(3)])
            bragg_q_index = np.sum(np.array([
                self.diffraction_dict["bragg_index"][idx] * 2 * np.pi * 
                self.material_class.q_vectors[idx] for idx in range(3)]), 
                axis = 0)
            oversample_q_space = np.array([np.array(q_start), -1 * np.array(q_start)] + bragg_q_index)
            
            return oversample_q_space, bragg_q_index

        # compare and reset the oversampling ratio
        
        oversample_q_space, q_index = oversampling_q_space(self.geometry_class.unit)
        flag = (
            all(scan_q_space[0, :] > oversample_q_space[0, :]) and 
            all(scan_q_space[1, :] < oversample_q_space[1, :])
            )
        while not flag:
            self.geometry_class.unit -= 0.1
            oversample_q_space, q_index = oversampling_q_space(self.geometry_class.unit)
            flag = (
                all(scan_q_space[0, :] > oversample_q_space[0, :]) and 
                all(scan_q_space[1, :] < oversample_q_space[1, :])
                )
        
        self.q_index = q_index
        self.q_start = oversample_q_space[0, :]
        self.q_end = oversample_q_space[1, :]
        self.scan_q_space = scan_q_space
        
    def real_space_sample(self, strain = 0, rotate_angle = 0, axis = [1, 0]):
        
        """
        Construct the real space sample profile with optional strain and rotation.
        
        This function constructs the real space sample profile based on the specified geometry and lattice plane.
        It applies optional strain to the profile and rotates the sample by a specified angle around a given axis.
        
        Args:
            strain (float, optional): Strain applied to the sample profile. Default is 0.
            rotate_angle (float, optional): Angle of rotation in degrees. Default is 0.
            axis (list, optional): Axis of rotation as a 2-element list. Default is [1, 0].
        
        Returns:
            numpy.ndarray: The rotated and optionally strained real space sample profile.
        
        Raises:
            None
        """

        # construct the real space sample profile
        
        self.geometry_class.create_geometry(self.geometry_dict["lattice_plane"])
        
        if strain == 0: strain = 1
        nanoisland = self.geometry_class.profile_matrix * strain
        
        # rotate the sample accroding to the rotate_angle
        
        rotated_nanoisland = sp.ndimage.rotate(nanoisland, rotate_angle, axis = axis)
        
        return rotated_nanoisland
    
    def structure_factor(self, strain = 0):
        
        """
        Calculate the structure factor considering crystal and profile factors.
        
        This function calculates the structure factor taking into account crystal and profile factors.
        It computes the real vector and q index based on the geometry and material classes.
        Then, it calculates the crystal phase and structure factor using the q index.
        Next, it computes the crystal factor as the sum of exponential terms.
        Afterward, it calculates the profile factor using Fourier transforms.
        Note: a small phase value adjustment is applied to avoid phase wrapping issues.
        
        Returns:
            None
        
        Raises:
            None
        """

        # crystal factor calculation
        
        self.calculate_q_space()
        self.geometry_class.create_geometry(self.geometry_dict["lattice_plane"])
        
        real_vector = np.matmul(
            np.matrix(self.geometry_class.matrix_index).T, 
            np.matmul(self.geometry_class.ub_matrix, np.matrix(self.material_class.vectors).T)
            )
        q_index_u = np.matmul(self.geometry_class.ub_matrix, np.matrix(self.q_index).T)

        crystal_phase = np.squeeze(np.asarray(np.matmul(real_vector, q_index_u)))
        structure_factor = self.material_class.structure_factor(np.matrix(self.q_index).T)
        crystal_factor = np.sum(np.exp(1j * crystal_phase)) * structure_factor
        
        # Calculate the profile factor
        # Warning! Why I multiply np.exp(1j * 1e-6) to the fft result? Because the small value of 
        #          phase will transform to pi or -pi after np.angle
        
        if strain == "random":
            self.geometry_class.random_strain()
            strain = self.geometry_class.profile_matrix * self.geometry_class.strain
            
        profile_factor = np.fft.ifftshift(np.fft.fftn(
            np.fft.fftshift(self.geometry_class.profile_matrix * np.exp(1j * strain))
            )) * np.exp(1j * 1e-6)
        self.nanoisland_factor = profile_factor * crystal_factor
    
    #---------------------------------------------------
    # import sample model from external source
    
    def import_sample(self, sample_dataset, voxel = None):
        
        #---------------------------------------------------
        # check the input parameters

        if not isinstance(sample_dataset, np.ndarray):
            raise ValueError("{sample_dataset} should be 3d numpy.ndarray")
        else:
            if not len(sample_dataset.shape) == 3:
                raise ValueError("{sample_dataset} should be 3d numpy.ndarray")
        
        if voxel is None:
            raise ValueError("Parameter {voxel} should be provided")
        
        #---------------------------------------------------
        # adjust the oversampling of the sample. 
        
        scan_q_space = np.flipud(np.rot90(self.diffraction_class.calculate_q_position(), -1))
        
        def oversampling_q_space(voxel_size):
            
            q_range = 0.1 * 2 * np.pi / voxel_size
            bragg_q_index = np.sum(np.array([
                self.diffraction_dict["bragg_index"][idx] * 2 * np.pi * self.material_class.q_vectors[idx] 
                for idx in range(3)]), axis = 0)
            
            q_start = np.array([-1 * q_range / 2 for i in range(3)])
            oversample_q_space = np.array([q_start, -1 * q_start]) + bragg_q_index
            
            return oversample_q_space, bragg_q_index
        
        # compare and reset the oversampling ratio
        
        voxel_size = float(voxel)
        oversample_q_space, q_index = oversampling_q_space(voxel_size)
        flag = (
            all(scan_q_space[0, :] > oversample_q_space[0, :]) and 
            all(scan_q_space[1, :] < oversample_q_space[1, :])
            )
        while not flag:
            voxel_size -= 0.1
            oversample_q_space, q_index = oversampling_q_space(voxel_size)
            flag = (
                all(scan_q_space[0, :] > oversample_q_space[0, :]) and 
                all(scan_q_space[1, :] < oversample_q_space[1, :])
                )
        
        # self.oversample_q_space = oversample_q_space
        # self.voxel_size = voxel_size
        self.q_index = q_index
        self.q_start = oversample_q_space[0, :]
        self.q_end = oversample_q_space[1, :]
        self.scan_q_space = scan_q_space
        
        #---------------------------------------------------
        # calculate the structure_factor
        
        sample_dataset = interpolate_sample_dataset(sample_dataset, voxel / voxel_size)
        self.nanoisland_factor = np.fft.fftshift(
            np.fft.fftn(np.fft.fftshift(sample_dataset))
            )
        
    #---------------------------------------------------
    # the operation of the experiment
    
    def reciprocal_to_angle(
            self, mode = "external", export_gif = False, gif_file_path = "theta_scan.gif", 
            sample_dataset = None, voxel = None, strain = 0
            ):
        
        """
        Convert reciprocal space map (RSM) to angle space representation.
        
        Parameters:
        - export_gif (bool): Whether to export the animation as a GIF. Default is False.
        - gif_file_path (str): File path to save the exported GIF. Default is "theta_scan.gif".
        
        Returns:
        - angle_space_factor (ndarray): Array representing the angle space factor.
        
        Notes:
        - This function requires prior calculation of q space and structure factor.
        - Affine transformation factors are calculated based on the diffraction class.
        - The RSM is transformed to angle space using an affine transformation.
        - The dataset can be exported as a GIF animation showing the evolution of angle space.
        """

        #---------------------------------------------------
        # adjust the q_space to the q_space determinted by theta scan
            
        if mode == "external":
            self.import_sample(sample_dataset, voxel = voxel)
            
        elif mode == "internal":
            self.structure_factor(strain = strain)
        
        oversample_q_range = [
            np.linspace(self.q_start[idx], self.q_end[idx], self.geometry_class.shape[idx]) 
            for idx in range(3)
            ]
        index = [[
            _locate(oversample_q_range[idx], self.scan_q_space[0, idx]), 
            _locate(oversample_q_range[idx], self.scan_q_space[1, idx])
            ] for idx in range(3)]
        nanoisland_factor = self.nanoisland_factor[
            index[2][0] : index[2][1], index[1][0] : index[1][1], index[0][0] : index[0][1]
            ]
        
        #---------------------------------------------------
        # calculate the affine transform factors
        
        self.diffraction_class.affine_transform_factor()
        
        rsm_shape = nanoisland_factor.shape
        scan_shape = self.diffraction_class.detector_size + [self.diffraction_class.frame_number]
        affine_offset = (
            np.array(rsm_shape) / 2.0 - np.dot(
                self.diffraction_class.rocking_matrix, 
                np.array(scan_shape).astype(float) / 2.0
            ))
        
        from scipy.ndimage import affine_transform
        from skimage.restoration import unwrap_phase
        
        # Warning! The prefilter must be True, or there will be interpolate probelm.
        
        angle_space_abs = affine_transform(
            np.abs(nanoisland_factor), self.diffraction_class.rocking_matrix, 
            offset = affine_offset, output_shape = tuple(np.array(scan_shape).astype(int)), order = 3, 
            mode = 'constant', cval = 0, output = float, prefilter = True
            )
        angle_space_image = affine_transform(
            unwrap_phase(np.angle(nanoisland_factor)), self.diffraction_class.rocking_matrix, 
            offset = affine_offset, output_shape = tuple(np.array(scan_shape).astype(int)), order = 3, 
            mode = 'constant', cval = 0, output = float, prefilter = True
            )
        
        self.angle_space_factor = angle_space_abs * np.exp(1j * angle_space_image)
        
        #---------------------------------------------------
        # export the dataset to gif
        
        if export_gif:
                
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, PillowWriter 
            
            figure, axis_handle = plt.subplots(figsize = (4, 4))
            angles = np.rad2deg(self.diffraction_dict["theta_scan"])
            
            axis_handle.set_title("theta angle @%.2f degree" % (angles[0]))
            image_object = plt.imshow(
                np.flipud(np.abs(self.angle_space_factor[0, :, :])**2), 
                vmax = np.abs(self.angle_space_factor).max()**2 / 10000
                )
            
            axis_handle.set_xlabel('x (pixel)', fontsize = 12)
            axis_handle.set_ylabel('y (pixel)', fontsize = 12)
            
            def init_function():
                axis_handle.set_title("theta angle @%.2f degree" % (angles[0]))
                image_object.set_data(np.flipud(np.abs(self.angle_space_factor[0, :, :])**2))
                return image_object
                
            def update_function(idx):
                axis_handle.set_title("theta angle @%.2f degree" % (angles[idx]))
                image_object.set_data(
                    np.flipud(np.abs(self.angle_space_factor[idx, :, :])**2)
                    )
                return image_object
                
            animation = FuncAnimation(
                figure, update_function, self.diffraction_class.frame_number, 
                init_func = init_function
                )  
            
            plt.show()
            plt.tight_layout()
            writer = PillowWriter(fps = 25)  
            animation.save(gif_file_path, writer = writer)  
        
        return self.angle_space_factor
    
#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    pass

