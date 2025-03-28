"""
测试配置文件
定义测试环境和通用fixture
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# 测试数据目录
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_data"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture
def test_dem_data():
    """生成测试用DEM数据"""
    # 创建一个简单的10x10的DEM数据
    dem = np.array([
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        [100, 110, 110, 110, 110, 110, 110, 110, 110, 100],
        [100, 110, 120, 120, 120, 120, 120, 120, 110, 100],
        [100, 110, 120, 130, 130, 130, 130, 120, 110, 100],
        [100, 110, 120, 130, 140, 140, 130, 120, 110, 100],
        [100, 110, 120, 130, 140, 140, 130, 120, 110, 100],
        [100, 110, 120, 130, 130, 130, 130, 120, 110, 100],
        [100, 110, 120, 120, 120, 120, 120, 120, 110, 100],
        [100, 110, 110, 110, 110, 110, 110, 110, 110, 100],
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    ], dtype=np.float32)
    return dem

@pytest.fixture
def test_landcover_data():
    """生成测试用土地覆盖数据"""
    # 创建一个简单的10x10的土地覆盖数据
    landcover = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 3, 3, 3, 3, 3, 3, 2, 1],
        [1, 2, 3, 4, 4, 4, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 4, 4, 4, 3, 2, 1],
        [1, 2, 3, 3, 3, 3, 3, 3, 2, 1],
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.int32)
    return landcover

@pytest.fixture
def test_trajectory_data(tmp_path):
    """生成测试用轨迹数据"""
    # 创建一个简单的轨迹CSV文件
    import pandas as pd
    
    # 生成测试数据 - 使用正弦曲线生成更真实的轨迹
    n_points = 20
    timestamps = pd.date_range('2024-03-27', periods=n_points, freq='30s')
    t = np.linspace(0, 2*np.pi, n_points)
    
    # 基准点和振幅
    base_lon, base_lat = 116.0, 39.0
    amp_lon, amp_lat = 0.01, 0.01
    
    data = {
        'timestamp': timestamps,
        'longitude': base_lon + amp_lon * np.sin(t),
        'latitude': base_lat + amp_lat * np.cos(t)
    }
    df = pd.DataFrame(data)
    
    # 保存到临时文件
    test_file = tmp_path / "test_trajectory.csv"
    df.to_csv(test_file, index=False)
    return test_file 