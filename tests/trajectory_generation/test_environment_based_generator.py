"""
基于环境的轨迹生成器测试模块
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import rasterio.transform

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import TerrainLoader
from src.trajectory_generation import EnvironmentBasedGenerator

@pytest.fixture
def terrain_loader():
    """创建测试用TerrainLoader实例"""
    loader = TerrainLoader()
    # 创建简单的测试数据
    dem_data = np.array([
        [100, 110, 120],
        [110, 120, 130],
        [120, 130, 140]
    ], dtype=np.float32)
    landcover_data = np.array([
        [1, 1, 2],
        [1, 2, 2],
        [2, 2, 3]
    ], dtype=np.int32)
    
    # 设置地理变换矩阵
    # 使用北京附近区域作为测试区域 (116.0E, 39.0N)
    resolution = 30  # 30米分辨率
    transform = rasterio.transform.from_origin(
        west=116.0,     # 左边界经度
        north=39.0,     # 上边界纬度
        xsize=resolution,  # 经度方向分辨率
        ysize=resolution   # 纬度方向分辨率
    )
    
    loader.dem_data = dem_data
    loader.landcover_data = landcover_data
    loader.resolution = resolution
    loader.transform = transform
    return loader

def test_environment_based_generator_init(terrain_loader):
    """测试基于环境的轨迹生成器初始化"""
    generator = EnvironmentBasedGenerator(terrain_loader)
    
    # 验证基本属性
    assert generator.terrain_loader is terrain_loader
    assert generator.terrain_analyzer is not None
    assert generator.terrain_analyzer.dem_data is not None
    assert generator.terrain_analyzer.slope_magnitude is not None
    assert generator.terrain_analyzer.slope_aspect is not None
    
    # 验证参数设置
    assert 'min_speed' in generator.params
    assert 'max_speed' in generator.params
    assert 'max_acceleration' in generator.params
    assert 'max_turn_rate' in generator.params
    assert 'time_step' in generator.params
    
    # 验证环境参数
    assert 'slope_speed_factors' in generator.env_params
    assert 'landcover_speed_factors' in generator.env_params
    assert 'path_smoothness' in generator.env_params
    assert 'waypoint_spacing' in generator.env_params

def test_generate_trajectory(terrain_loader):
    """测试轨迹生成功能"""
    generator = EnvironmentBasedGenerator(terrain_loader)
    
    # 设置测试起终点（在测试区域内）
    start_point = (116.001, 39.001)  # 起点
    end_point = (116.002, 39.002)    # 终点
    
    # 生成轨迹
    trajectory = generator.generate_trajectory(start_point, end_point)
    
    # 验证基本数据结构
    assert isinstance(trajectory, pd.DataFrame)
    assert len(trajectory) > 0
    
    # 验证必要的列存在
    required_columns = [
        'timestamp_ms',
        'longitude', 'latitude', 'altitude_m',
        'velocity_north_ms', 'velocity_east_ms', 'velocity_down_ms',
        'acceleration_x_ms2', 'acceleration_y_ms2', 'acceleration_z_ms2',
        'angular_velocity_x_rads', 'angular_velocity_y_rads', 'angular_velocity_z_rads'
    ]
    for col in required_columns:
        assert col in trajectory.columns
    
    # 验证时间戳递增
    assert (np.diff(trajectory['timestamp_ms']) > 0).all()
    
    # 验证位置在合理范围内
    assert trajectory['longitude'].between(116.0, 116.1).all()
    assert trajectory['latitude'].between(39.0, 39.1).all()
    
    # 验证速度约束
    speed = np.sqrt(
        trajectory['velocity_north_ms']**2 + 
        trajectory['velocity_east_ms']**2
    )
    assert speed.between(
        generator.params['min_speed'],
        generator.params['max_speed']
    ).all()
    
    # 验证加速度约束
    acceleration = np.sqrt(
        trajectory['acceleration_x_ms2']**2 + 
        trajectory['acceleration_y_ms2']**2 + 
        trajectory['acceleration_z_ms2']**2
    )
    assert (acceleration <= generator.params['max_acceleration']).all()
    
    # 验证转向率约束
    turn_rate = np.sqrt(
        trajectory['angular_velocity_x_rads']**2 + 
        trajectory['angular_velocity_y_rads']**2 + 
        trajectory['angular_velocity_z_rads']**2
    )
    assert (turn_rate <= np.radians(generator.params['max_turn_rate'])).all()

def test_validate_trajectory(terrain_loader):
    """测试轨迹验证功能"""
    generator = EnvironmentBasedGenerator(terrain_loader)
    
    # 创建有效轨迹数据
    data = {
        'timestamp_ms': np.arange(0, 5000, 1000),
        'longitude': [116.001, 116.002, 116.003, 116.004, 116.005],
        'latitude': [39.001, 39.002, 39.003, 39.004, 39.005],
        'altitude_m': [100, 110, 120, 130, 140],
        'velocity_north_ms': [1.0, 1.5, 2.0, 1.5, 1.0],
        'velocity_east_ms': [0.5, 1.0, 1.5, 1.0, 0.5],
        'velocity_down_ms': [0.0, 0.1, 0.2, 0.1, 0.0],
        'acceleration_x_ms2': [0.1, 0.2, 0.1, 0.0, -0.1],
        'acceleration_y_ms2': [0.1, 0.1, 0.0, -0.1, -0.1],
        'acceleration_z_ms2': [0.0, 0.1, 0.0, -0.1, 0.0],
        'angular_velocity_x_rads': [0.01, 0.02, 0.01, 0.0, -0.01],
        'angular_velocity_y_rads': [0.01, 0.01, 0.0, -0.01, -0.01],
        'angular_velocity_z_rads': [0.1, 0.2, 0.1, 0.0, -0.1]
    }
    valid_trajectory = pd.DataFrame(data)
    
    # 验证有效轨迹
    assert generator.validate_trajectory(valid_trajectory)
    
    # 创建无效轨迹（速度过大）
    invalid_trajectory = valid_trajectory.copy()
    invalid_trajectory['velocity_north_ms'] *= 100
    assert not generator.validate_trajectory(invalid_trajectory)
    
    # 创建无效轨迹（加速度过大）
    invalid_trajectory = valid_trajectory.copy()
    invalid_trajectory['acceleration_x_ms2'] *= 100
    assert not generator.validate_trajectory(invalid_trajectory)
    
    # 创建无效轨迹（转向率过大）
    invalid_trajectory = valid_trajectory.copy()
    invalid_trajectory['angular_velocity_z_rads'] *= 100
    assert not generator.validate_trajectory(invalid_trajectory)

def test_invalid_points(terrain_loader):
    """测试无效点处理"""
    generator = EnvironmentBasedGenerator(terrain_loader)
    
    # 测试超出范围的点
    invalid_point = (150.0, 60.0)  # 超出中国范围
    assert not generator._validate_point(*invalid_point)
    
    # 测试超出地形数据范围的点
    invalid_point = (117.0, 40.0)  # 超出测试数据范围
    assert not generator._validate_point(*invalid_point)
    
    # 测试有效点
    valid_point = (116.001, 39.001)
    assert generator._validate_point(*valid_point)

def test_update_params(terrain_loader):
    """测试参数更新功能"""
    generator = EnvironmentBasedGenerator(terrain_loader)
    
    # 保存原始参数
    original_params = generator.params.copy()
    
    # 更新部分参数
    new_params = {
        'max_speed': 30.0,
        'max_acceleration': 3.0
    }
    generator.update_params(new_params)
    
    # 验证参数更新
    assert generator.params['max_speed'] == new_params['max_speed']
    assert generator.params['max_acceleration'] == new_params['max_acceleration']
    
    # 验证其他参数保持不变
    for key in original_params:
        if key not in new_params:
            assert generator.params[key] == original_params[key] 