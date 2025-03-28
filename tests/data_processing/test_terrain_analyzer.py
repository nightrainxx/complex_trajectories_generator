"""
TerrainAnalyzer模块的单元测试
"""

import pytest
import numpy as np
import rasterio
from pathlib import Path

from src.data_processing import TerrainAnalyzer

def test_terrain_analyzer_init():
    """测试TerrainAnalyzer的初始化"""
    analyzer = TerrainAnalyzer()
    assert analyzer.dem_data is None
    assert analyzer.slope_magnitude is None
    assert analyzer.slope_aspect is None
    assert analyzer.dzdx is None
    assert analyzer.dzdy is None
    assert analyzer.resolution is None

def test_load_dem(test_dem_data):
    """测试DEM数据加载功能"""
    analyzer = TerrainAnalyzer()
    resolution = (30.0, 30.0)  # 30米分辨率
    analyzer.load_dem(test_dem_data, resolution)
    
    assert analyzer.dem_data is not None
    assert np.array_equal(analyzer.dem_data, test_dem_data)
    assert analyzer.resolution == resolution

def test_calculate_slope_magnitude(test_dem_data):
    """测试坡度大小计算功能"""
    analyzer = TerrainAnalyzer()
    resolution = (30.0, 30.0)
    analyzer.load_dem(test_dem_data, resolution)
    
    slope_magnitude = analyzer.calculate_slope_magnitude()
    
    assert slope_magnitude is not None
    assert slope_magnitude.shape == test_dem_data.shape
    assert np.all(slope_magnitude >= 0)  # 坡度应该非负
    assert analyzer.slope_magnitude is not None
    
    # 检查平地和陡坡
    flat_mask = test_dem_data == test_dem_data[0, 0]  # 找出高度相同的点
    assert np.all(slope_magnitude[flat_mask] < 0.1)  # 平地坡度应接近0

def test_calculate_slope_aspect(test_dem_data):
    """测试坡向计算功能"""
    analyzer = TerrainAnalyzer()
    resolution = (30.0, 30.0)
    analyzer.load_dem(test_dem_data, resolution)
    
    slope_aspect = analyzer.calculate_slope_aspect()
    
    assert slope_aspect is not None
    assert slope_aspect.shape == test_dem_data.shape
    assert analyzer.slope_aspect is not None
    
    # 检查坡向范围
    valid_mask = slope_aspect != -1
    if np.any(valid_mask):
        assert np.all((slope_aspect[valid_mask] >= 0) & (slope_aspect[valid_mask] < 360))
    
    # 检查平地的坡向
    if analyzer.slope_magnitude is not None:
        flat_mask = analyzer.slope_magnitude < 0.1
        assert np.all(slope_aspect[flat_mask] == -1)

def test_calculate_gradients(test_dem_data):
    """测试梯度计算功能"""
    analyzer = TerrainAnalyzer()
    resolution = (30.0, 30.0)
    analyzer.load_dem(test_dem_data, resolution)
    
    dzdx, dzdy = analyzer.calculate_gradients()
    
    assert dzdx is not None
    assert dzdy is not None
    assert dzdx.shape == test_dem_data.shape
    assert dzdy.shape == test_dem_data.shape
    assert analyzer.dzdx is not None
    assert analyzer.dzdy is not None
    
    # 检查平地的梯度
    flat_mask = test_dem_data == test_dem_data[0, 0]
    assert np.all(np.abs(dzdx[flat_mask]) < 1e-6)
    assert np.all(np.abs(dzdy[flat_mask]) < 1e-6)

def test_save_results(test_dem_data, tmp_path):
    """测试结果保存功能"""
    analyzer = TerrainAnalyzer()
    resolution = (30.0, 30.0)
    analyzer.load_dem(test_dem_data, resolution)
    
    # 计算所有地形属性
    analyzer.calculate_slope_magnitude()
    analyzer.calculate_slope_aspect()
    analyzer.calculate_gradients()
    
    # 保存结果
    output_dir = tmp_path / "terrain_results"
    analyzer.save_results(output_dir)
    
    # 验证文件是否创建
    assert (output_dir / "slope_magnitude_30m_100km.tif").exists()
    assert (output_dir / "slope_aspect_30m_100km.tif").exists()
    assert (output_dir / "dzdx_30m_100km.tif").exists()
    assert (output_dir / "dzdy_30m_100km.tif").exists()
    
    # 验证保存的数据是否正确
    with rasterio.open(output_dir / "slope_magnitude_30m_100km.tif") as src:
        saved_slope = src.read(1)
        assert np.array_equal(saved_slope, analyzer.slope_magnitude)

def test_error_handling():
    """测试错误处理"""
    analyzer = TerrainAnalyzer()
    
    # 测试在未加载数据时的错误处理
    with pytest.raises(ValueError):
        analyzer.calculate_slope_magnitude()
    
    with pytest.raises(ValueError):
        analyzer.calculate_slope_aspect()
    
    with pytest.raises(ValueError):
        analyzer.calculate_gradients() 