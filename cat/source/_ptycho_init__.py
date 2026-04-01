# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Sat Oct 11 22:41:30 2025"
__email__    = "xuhan@ihep.ac.cn"


"""
Description
"""

#-----------------------------------------------------------------------------#
# modules

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional


#-----------------------------------------------------------------------------#
# parameters

#-----------------------------------------------------------------------------#
# functions

#-----------------------------------------------------------------------------#
# classes

class CXI_DataLoader:
    """
    CXI格式数据加载器
    专门用于处理CXI格式的ptychography实验数据
    """
    
    def __init__(self):
        self.data_loaded = False
        self.params_initialized = False
        
        # 数据存储
        self.scan_positions = None  # 扫描点位置 [N, 2]
        self.diffraction_patterns = None  # 衍射图案 [N, ny, nx]
        self.probe_initial = None  # 初始探针估计 [ny, nx]
        
        # 几何参数
        self.geometry_params = {
            'wavelength': None,           # 波长 (m)
            'detector_distance': None,    # 探测器到样品距离 (m)
            'detector_pixelsize': None,   # 探测器像素大小 (m)
            'focus_angle': None,          # 聚焦光学元件接收角 (rad)
            'focal_length': None,         # 焦距 (m)
            'defocus_distance': None,     # 离焦距离 (m)
            'energy': None               # 能量 (eV)，可由波长计算
        }
        
        # 重构参数
        self.reconstruction_params = {
            'n_cycles': 100,              # 重构循环次数
            'beta': 1.0,                  # 探针更新正则化参数
            'alpha': 0.05,                # 样本更新正则化参数  
            'gamma_o': 0.2,               # 样本更新步长
            'gamma_p': 1.0,               # 探针更新步长
            'probe_support_radius': None  # 探针支撑半径
        }
        
        # 数据参数
        self.data_info = {
            'n_positions': 0,             # 扫描点数量
            'pattern_shape': None,       # 衍射图案形状
            'scan_range': None,          # 扫描范围
            'data_path': None            # 数据路径
        }
    
    def load_cxi_data(self, data_path: str) -> bool:
        """
        加载CXI格式的实验数据
        
        Args:
            data_path: CXI数据文件路径
            
        Returns:
            bool: 数据加载是否成功
        """
        try:
            self.data_info['data_path'] = data_path
            
            if not os.path.exists(data_path):
                print(f"文件不存在: {data_path}")
                return False
                
            with h5.File(data_path, 'r') as f:
                # 读取衍射数据
                if '/entry_1/data_1/data' in f:
                    self.diffraction_patterns = f['/entry_1/data_1/data'][:]
                    print(f"加载衍射数据，形状: {self.diffraction_patterns.shape}")
                else:
                    print("未找到衍射数据路径: /entry_1/data_1/data")
                    return False
                
                # 读取扫描位置
                if '/entry_1/data_1/translation' in f:
                    positions = f['/entry_1/data_1/translation'][:]
                    self.scan_positions = positions.astype(int)
                    print(f"加载扫描位置，形状: {self.scan_positions.shape}")
                else:
                    print("未找到扫描位置路径: /entry_1/data_1/translation")
                    return False
                
                # 尝试从CXI文件读取几何参数
                self._read_cxi_geometry_params(f)
                
            self._analyze_loaded_data()
            self.data_loaded = True
            print("CXI数据加载成功!")
            return True
            
        except Exception as e:
            print(f"CXI数据加载错误: {e}")
            return False
    
    def _read_cxi_geometry_params(self, f: h5.File):
        """
        从CXI文件中读取几何参数
        
        Args:
            f: 打开的HDF5文件对象
        """
        # 读取探测器距离
        if '/entry_1/instrument_1/detector_1/distance' in f:
            self.geometry_params['detector_distance'] = f['/entry_1/instrument_1/detector_1/distance'][()]
            print(f"探测器距离: {self.geometry_params['detector_distance']} m")
        
        # 读取像素大小
        if '/entry_1/instrument_1/detector_1/x_pixel_size' in f:
            x_pixel_size = f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]
            y_pixel_size = f['/entry_1/instrument_1/detector_1/y_pixel_size'][()]
            self.geometry_params['detector_pixelsize'] = (x_pixel_size + y_pixel_size) / 2
            print(f"探测器像素大小: {self.geometry_params['detector_pixelsize']} m")
        
        # 读取能量
        if '/entry_1/instrument_1/source_1/energy' in f:
            energy = f['/entry_1/instrument_1/source_1/energy'][()]
            self.geometry_params['energy'] = energy
            print(f"光子能量: {energy} eV")
            
            # 从能量计算波长
            h = 4.135667662e-15  # Planck常数 (eV·s)
            c = 299792458        # 光速 (m/s)
            wavelength = h * c / energy
            self.geometry_params['wavelength'] = wavelength
            print(f"计算波长: {wavelength} m")
    
    def _analyze_loaded_data(self):
        """分析加载的数据"""
        if self.diffraction_patterns is not None:
            self.data_info['n_positions'] = len(self.diffraction_patterns)
            self.data_info['pattern_shape'] = self.diffraction_patterns[0].shape
            
        if self.scan_positions is not None:
            # 计算扫描范围
            x_min, x_max = np.min(self.scan_positions[:, 0]), np.max(self.scan_positions[:, 0])
            y_min, y_max = np.min(self.scan_positions[:, 1]), np.max(self.scan_positions[:, 1])
            self.data_info['scan_range'] = {
                'x': (x_min, x_max),
                'y': (y_min, y_max),
                'extent': (x_max - x_min, y_max - y_min)
            }
    
    def set_geometry_parameters(self, 
                              wavelength: Optional[float] = None,
                              detector_distance: Optional[float] = None,
                              detector_pixelsize: Optional[float] = None,
                              focus_angle: Optional[float] = None,
                              focal_length: Optional[float] = None, 
                              defocus_distance: Optional[float] = None,
                              energy: Optional[float] = None):
        """
        设置几何参数
        
        Args:
            wavelength: 波长 (m)
            detector_distance: 探测器到样品距离 (m)
            detector_pixelsize: 探测器像素大小 (m)
            focus_angle: 聚焦光学元件接收角 (rad)
            focal_length: 焦距 (m)
            defocus_distance: 离焦距离 (m)
            energy: 能量 (eV)，如果提供则覆盖波长
        """
        if energy is not None:
            # 从能量计算波长: λ = hc/E
            h = 4.135667662e-15  # Planck常数 (eV·s)
            c = 299792458        # 光速 (m/s)
            wavelength = h * c / energy
        
        # 只更新提供的参数
        params = {
            'wavelength': wavelength,
            'detector_distance': detector_distance,
            'detector_pixelsize': detector_pixelsize,
            'focus_angle': focus_angle,
            'focal_length': focal_length,
            'defocus_distance': defocus_distance,
            'energy': energy
        }
        
        for key, value in params.items():
            if value is not None:
                self.geometry_params[key] = value
    
    def set_reconstruction_parameters(self, 
                                    n_cycles: int = 100,
                                    beta: float = 1.0,
                                    alpha: float = 0.05,
                                    gamma_o: float = 0.2,
                                    gamma_p: float = 1.0,
                                    probe_support_radius: Optional[float] = None):
        """
        设置重构参数
        """
        self.reconstruction_params.update({
            'n_cycles': n_cycles,
            'beta': beta,
            'alpha': alpha,
            'gamma_o': gamma_o,
            'gamma_p': gamma_p,
            'probe_support_radius': probe_support_radius
        })
    
    def initialize_probe_estimate(self, probe_shape: Tuple[int, int], 
                                probe_type: str = 'gaussian',
                                **kwargs) -> np.ndarray:
        """
        初始化探针估计
        
        Args:
            probe_shape: 探针形状 (ny, nx)
            probe_type: 探针类型 ('gaussian', 'aperture', 'focused')
            **kwargs: 探针特定参数
            
        Returns:
            初始探针估计
        """
        ny, nx = probe_shape
        
        if probe_type == 'gaussian':
            # 高斯探针
            sigma_x = kwargs.get('sigma_x', nx // 6)
            sigma_y = kwargs.get('sigma_y', ny // 6)
            x = np.linspace(-nx//2, nx//2, nx)
            y = np.linspace(-ny//2, ny//2, ny)
            X, Y = np.meshgrid(x, y)
            probe = np.exp(-(X**2/(2*sigma_x**2) + Y**2/(2*sigma_y**2)))
            
        elif probe_type == 'aperture':
            # 圆形孔径探针
            radius = kwargs.get('radius', min(nx, ny) // 4)
            y, x = np.ogrid[-ny//2:ny//2, -nx//2:nx//2]
            mask = x*x + y*y <= radius*radius
            probe = np.zeros((ny, nx))
            probe[mask] = 1.0
            
        elif probe_type == 'focused':
            # 聚焦探针（考虑离焦）
            if self.geometry_params['defocus_distance'] is None:
                raise ValueError("聚焦探针需要defocus_distance参数")
                
            # 使用角谱方法计算聚焦探针
            probe = self._calculate_focused_probe(probe_shape, **kwargs)
            
        else:
            raise ValueError(f"不支持的探针类型: {probe_type}")
        
        # 添加随机相位扰动
        phase_noise = kwargs.get('phase_noise', 0.1)
        if phase_noise > 0:
            phase = np.random.uniform(-phase_noise, phase_noise, (ny, nx))
            probe = probe * np.exp(1j * phase)
        
        self.probe_initial = probe.astype(np.complex128)
        return self.probe_initial
    
    def _calculate_focused_probe(self, probe_shape: Tuple[int, int], **kwargs) -> np.ndarray:
        """
        计算聚焦探针（考虑离焦效应）
        """
        ny, nx = probe_shape
        wavelength = self.geometry_params['wavelength']
        defocus = self.geometry_params['defocus_distance']
        focal_length = self.geometry_params['focal_length']
        
        # 计算波数
        k = 2 * np.pi / wavelength
        
        # 创建坐标网格
        x = np.linspace(-nx//2, nx//2, nx) * self.geometry_params['detector_pixelsize']
        y = np.linspace(-ny//2, ny//2, ny) * self.geometry_params['detector_pixelsize']
        X, Y = np.meshgrid(x, y)
        
        # 计算波前曲率（离焦效应）
        if defocus != 0:
            # 离焦引起的相位变化
            r_squared = X**2 + Y**2
            phase_defocus = k * defocus * (1 - np.sqrt(1 - r_squared / (defocus**2)))
            probe = np.exp(1j * phase_defocus)
        else:
            # 完美聚焦（平面波前）
            probe = np.ones((ny, nx))
        
        # 应用孔径限制（如果提供接收角）
        if self.geometry_params['focus_angle'] is not None:
            max_radius = focal_length * np.tan(self.geometry_params['focus_angle'])
            aperture_mask = (X**2 + Y**2) <= max_radius**2
            probe = probe * aperture_mask
        
        return probe
    
    def initialize_sample_estimate(self, sample_shape: Tuple[int, int],
                                 sample_type: str = 'random',
                                 **kwargs) -> np.ndarray:
        """
        初始化样本估计
        
        Args:
            sample_shape: 样本形状 (ny, nx)
            sample_type: 样本类型 ('random', 'constant', 'support')
            
        Returns:
            初始样本估计
        """
        ny, nx = sample_shape
        
        if sample_type == 'random':
            # 随机相位样本
            amplitude = kwargs.get('amplitude', 1.0)
            phase_range = kwargs.get('phase_range', (0, 2*np.pi))
            phase = np.random.uniform(phase_range[0], phase_range[1], (ny, nx))
            sample = amplitude * np.exp(1j * phase)
            
        elif sample_type == 'constant':
            # 常数样本
            amplitude = kwargs.get('amplitude', 1.0)
            phase = kwargs.get('phase', 0.0)
            sample = amplitude * np.exp(1j * phase) * np.ones((ny, nx))
            
        elif sample_type == 'support':
            # 基于支撑约束的样本
            support_radius = kwargs.get('support_radius', min(nx, ny) // 3)
            y, x = np.ogrid[-ny//2:ny//2, -nx//2:nx//2]
            support = x*x + y*y <= support_radius*support_radius
            sample = support.astype(np.complex128)
            
        else:
            raise ValueError(f"不支持的样本类型: {sample_type}")
        
        return sample
    
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """
        验证所有参数是否合理
        
        Returns:
            (是否有效, 错误消息列表)
        """
        errors = []
        
        # 检查数据加载
        if not self.data_loaded:
            errors.append("未加载实验数据")
        
        # 检查几何参数
        required_geometry = ['wavelength', 'detector_distance', 'detector_pixelsize']
        for param in required_geometry:
            if self.geometry_params[param] is None:
                errors.append(f"缺少必要的几何参数: {param}")
        
        # 检查数据一致性
        if self.diffraction_patterns is not None and self.scan_positions is not None:
            if len(self.diffraction_patterns) != len(self.scan_positions):
                errors.append(f"衍射图案数量({len(self.diffraction_patterns)})与扫描位置数量({len(self.scan_positions)})不匹配")
        
        # 检查重构参数
        if self.reconstruction_params['n_cycles'] <= 0:
            errors.append("重构循环次数必须为正数")
        
        return len(errors) == 0, errors
    
    def get_summary(self) -> Dict:
        """获取参数摘要"""
        summary = {
            'data_loaded': self.data_loaded,
            'data_info': self.data_info,
            'geometry_params': self.geometry_params,
            'reconstruction_params': self.reconstruction_params,
            'probe_initialized': self.probe_initial is not None
        }
        return summary
    
    def save_parameters(self, filepath: str):
        """保存参数到JSON文件"""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def plot_initial_data(self, save_path: Optional[str] = None):
        """绘制初始数据概览"""
        if not self.data_loaded:
            print("未加载数据，无法绘图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 扫描位置分布
        if self.scan_positions is not None:
            axes[0, 0].scatter(self.scan_positions[:, 0], self.scan_positions[:, 1], 
                              alpha=0.5, s=10)
            axes[0, 0].set_title('扫描位置分布')
            axes[0, 0].set_xlabel('X位置 (像素)')
            axes[0, 0].set_ylabel('Y位置 (像素)')
            axes[0, 0].set_aspect('equal')
        
        # 2. 示例衍射图案
        if self.diffraction_patterns is not None:
            sample_pattern = self.diffraction_patterns[0]
            im = axes[0, 1].imshow(np.log10(sample_pattern + 1), cmap='viridis')
            axes[0, 1].set_title('示例衍射图案 (log尺度)')
            plt.colorbar(im, ax=axes[0, 1])
        
        # 3. 初始探针（如果已初始化）
        if self.probe_initial is not None:
            axes[1, 0].imshow(np.abs(self.probe_initial), cmap='hot')
            axes[1, 0].set_title('初始探针振幅')
            axes[1, 1].imshow(np.angle(self.probe_initial), cmap='hsv')
            axes[1, 1].set_title('初始探针相位')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 使用示例
def test_cxi_loader():
    """测试CXI数据加载器"""
    # 创建数据加载器实例
    loader = CXI_DataLoader()
    
    # 设置几何参数（示例值，需要根据实验调整）
    loader.set_geometry_parameters(
        wavelength=1.5e-10,           # 1.5 Å波长
        detector_distance=2.0,        # 2米探测器距离
        detector_pixelsize=75e-6,     # 75μm像素大小
        focus_angle=0.01,             # 10mrad接收角
        focal_length=0.2,             # 20cm焦距
        defocus_distance=4e-3         # 4mm离焦距离
    )
    
    # 设置重构参数
    loader.set_reconstruction_parameters(
        n_cycles=100,
        beta=1.0,
        alpha=0.05,
        gamma_o=0.2,
        gamma_p=1.0
    )
    
    # 加载实验数据（需要替换为实际文件路径）
    data_file = "small_beam_2um_012.cxi"  # 替换为实际路径
    if os.path.exists(data_file):
        success = loader.load_cxi_data(data_file)
        if success:
            # 初始化探针估计
            probe_shape = (256, 256)  # 根据实际情况调整
            probe_initial = loader.initialize_probe_estimate(
                probe_shape, 
                probe_type='focused',
                phase_noise=0.1
            )
            
            # 验证参数
            is_valid, errors = loader.validate_parameters()
            if is_valid:
                print("所有参数验证通过!")
                
                # 显示参数摘要
                summary = loader.get_summary()
                print("\n参数摘要:")
                for key, value in summary.items():
                    print(f"{key}: {value}")
                
                # 绘制初始数据
                loader.plot_initial_data("initial_data_overview.png")
                
                # 保存参数
                loader.save_parameters("reconstruction_parameters.json")
                
            else:
                print("参数验证失败:")
                for error in errors:
                    print(f"  - {error}")
        else:
            print("数据加载失败")
    else:
        print(f"数据文件不存在: {data_file}")

if __name__ == "__main__":
    test_cxi_loader()


#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    pass

