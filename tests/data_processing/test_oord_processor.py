"""
OORD数据处理器测试模块
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

from src.data_processing import TerrainLoader, TerrainAnalyzer, OORDProcessor

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

@pytest.fixture
def test_trajectory_data(tmp_path):
    """创建测试用轨迹数据"""
    # 创建测试数据
    data = {
        'timestamp_ms': np.arange(0, 5000, 1000),  # 5个点，每秒一个
        'latitude': [39.0, 39.001, 39.002, 39.003, 39.004],
        'longitude': [116.0, 116.001, 116.002, 116.003, 116.004],
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
    df = pd.DataFrame(data)
    
    # 保存到临时文件
    test_file = tmp_path / "test_trajectory.csv"
    df.to_csv(test_file, index=False)
    return test_file

def test_oord_processor_init(terrain_loader):
    """测试OORDProcessor初始化"""
    processor = OORDProcessor(terrain_loader)
    assert processor.terrain_loader is terrain_loader
    assert isinstance(processor.trajectories, dict)
    assert isinstance(processor.processed_trajectories, dict)
    assert len(processor.trajectories) == 0
    assert len(processor.processed_trajectories) == 0

def test_load_trajectory(terrain_loader, test_trajectory_data):
    """测试轨迹加载功能"""
    processor = OORDProcessor(terrain_loader)
    df = processor.load_trajectory(test_trajectory_data)
    
    # 验证基本数据结构
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5  # 测试数据有5个点
    
    # 验证计算的特征
    assert 'speed' in df.columns
    assert 'heading' in df.columns
    assert 'turn_rate' in df.columns
    assert 'acceleration' in df.columns
    
    # 验证速度计算
    expected_speed = np.sqrt(df['velocity_north_ms']**2 + df['velocity_east_ms']**2)
    np.testing.assert_array_almost_equal(df['speed'], expected_speed)
    
    # 验证加速度计算
    expected_accel = np.sqrt(
        df['acceleration_x_ms2']**2 + 
        df['acceleration_y_ms2']**2 + 
        df['acceleration_z_ms2']**2
    )
    np.testing.assert_array_almost_equal(df['acceleration'], expected_accel)
    
    # 验证航向角计算
    expected_heading = np.degrees(np.arctan2(
        df['velocity_east_ms'],
        df['velocity_north_ms']
    )) % 360
    np.testing.assert_array_almost_equal(df['heading'], expected_heading)

def test_process_trajectory(terrain_loader, test_trajectory_data):
    """测试轨迹处理功能"""
    processor = OORDProcessor(terrain_loader)
    processor.load_trajectory(test_trajectory_data)
    trajectory_id = Path(test_trajectory_data).stem
    
    df_processed = processor.process_trajectory(
        trajectory_id,
        max_speed=50.0
    )
    
    # 验证基本处理结果
    assert isinstance(df_processed, pd.DataFrame)
    assert len(df_processed) > 0
    assert all(df_processed['speed'] <= 50.0)
    
    # 验证环境分组
    assert 'slope_group' in df_processed.columns
    assert 'group_label' in df_processed.columns
    assert df_processed['slope_group'].notna().all()
    assert df_processed['group_label'].notna().all()

def test_analyze_environment_interaction(terrain_loader, test_trajectory_data):
    """测试环境交互分析功能"""
    processor = OORDProcessor(terrain_loader)
    processor.load_trajectory(test_trajectory_data)
    trajectory_id = Path(test_trajectory_data).stem
    processor.process_trajectory(trajectory_id)
    
    stats = processor.analyze_environment_interaction()
    
    # 验证统计结果
    assert isinstance(stats, dict)
    assert len(stats) > 0
    
    # 验证统计指标
    for group_stats in stats.values():
        required_stats = [
            'speed_mean', 'speed_std', 'speed_median', 'speed_max',
            'acceleration_std', 'turn_rate_std', 'sample_size'
        ]
        for stat in required_stats:
            assert stat in group_stats
            assert isinstance(group_stats[stat], (int, float))
            assert not np.isnan(group_stats[stat])

def test_calculate_haversine_distance():
    """测试Haversine距离计算"""
    # 使用北京天安门（116.397, 39.916）和上海外滩（121.484, 31.233）的坐标
    lon1, lat1 = 116.397, 39.916  # 北京天安门
    lon2, lat2 = 121.484, 31.233  # 上海外滩
    
    # 计算两点之间的距离
    processor = OORDProcessor()  # 不需要地形数据
    distance = processor.calculate_haversine_distance(
        lon1=lon1, lat1=lat1,
        lon2=lon2, lat2=lat2
    )
    
    # 预期距离约为1067公里
    expected_distance = 1067.0  # 单位：公里
    assert abs(distance - expected_distance) < 5.0  # 允许5公里的误差

def test_calculate_heading():
    """测试航向角计算"""
    processor = OORDProcessor()  # 不需要地形数据
    
    # 测试正北方向
    heading = processor.calculate_heading(
        velocity_north=1.0,
        velocity_east=0.0
    )
    assert abs(heading - 0.0) < 1e-6
    
    # 测试正东方向
    heading = processor.calculate_heading(
        velocity_north=0.0,
        velocity_east=1.0
    )
    assert abs(heading - 90.0) < 1e-6
    
    # 测试正南方向
    heading = processor.calculate_heading(
        velocity_north=-1.0,
        velocity_east=0.0
    )
    assert abs(heading - 180.0) < 1e-6
    
    # 测试正西方向
    heading = processor.calculate_heading(
        velocity_north=0.0,
        velocity_east=-1.0
    )
    assert abs(heading - 270.0) < 1e-6
    
    # 测试东北方向（45度）
    heading = processor.calculate_heading(
        velocity_north=1.0,
        velocity_east=1.0
    )
    assert abs(heading - 45.0) < 1e-6 