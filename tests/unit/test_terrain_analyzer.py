"""地形分析器单元测试"""

import pytest
import numpy as np
import rasterio
from pathlib import Path

from src.generator.terrain_analyzer import TerrainAnalyzer
from config import *

@pytest.fixture
def test_dem_data():
    """创建测试用DEM数据"""
    # 创建一个简单的3x3 DEM数据
    return np.array([
        [100, 110, 120],
        [105, 115, 125],
        [110, 120, 130]
    ], dtype=np.float32)

@pytest.fixture
def test_dem_file(tmp_path, test_dem_data):
    """创建测试用DEM文件"""
    dem_path = tmp_path / "test_dem.tif"
    
    # 创建测试用GeoTIFF文件
    transform = rasterio.transform.from_origin(0, 0, 30, 30)
    with rasterio.open(
        dem_path,
        'w',
        driver='GTiff',
        height=3,
        width=3,
        count=1,
        dtype=np.float32,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(test_dem_data, 1)
    
    return dem_path

class TestTerrainAnalyzer:
    """地形分析器测试类"""
    
    def test_init(self, test_dem_file):
        """测试初始化"""
        analyzer = TerrainAnalyzer(test_dem_file)
        assert analyzer.dem_path == test_dem_file
        assert analyzer.dem_data is not None
        assert analyzer.transform is not None
    
    def test_calculate_slope_magnitude(self, test_dem_file):
        """测试坡度大小计算"""
        analyzer = TerrainAnalyzer(test_dem_file)
        slope_mag = analyzer.calculate_slope_magnitude()
        
        assert slope_mag is not None
        assert slope_mag.shape == (3, 3)
        assert np.all(slope_mag >= 0)  # 坡度应该非负
        assert np.all(slope_mag <= 90)  # 坡度应该小于90度
    
    def test_calculate_slope_aspect(self, test_dem_file):
        """测试坡向计算"""
        analyzer = TerrainAnalyzer(test_dem_file)
        aspect = analyzer.calculate_slope_aspect()
        
        assert aspect is not None
        assert aspect.shape == (3, 3)
        assert np.all((aspect >= -1) & (aspect <= 360))  # 坡向应该在-1到360度之间
    
    def test_calculate_gradients(self, test_dem_file):
        """测试梯度计算"""
        analyzer = TerrainAnalyzer(test_dem_file)
        dzdx, dzdy = analyzer.calculate_gradients()
        
        assert dzdx is not None and dzdy is not None
        assert dzdx.shape == (3, 3)
        assert dzdy.shape == (3, 3)
    
    def test_save_terrain_attributes(self, test_dem_file, tmp_path):
        """测试地形属性保存"""
        analyzer = TerrainAnalyzer(test_dem_file)
        
        # 计算地形属性
        slope_mag = analyzer.calculate_slope_magnitude()
        aspect = analyzer.calculate_slope_aspect()
        dzdx, dzdy = analyzer.calculate_gradients()
        
        # 保存文件
        slope_mag_path = tmp_path / "slope_mag.tif"
        aspect_path = tmp_path / "aspect.tif"
        dzdx_path = tmp_path / "dzdx.tif"
        dzdy_path = tmp_path / "dzdy.tif"
        
        analyzer.save_terrain_attributes(
            slope_mag_path,
            aspect_path,
            dzdx_path,
            dzdy_path
        )
        
        # 验证文件是否创建
        assert slope_mag_path.exists()
        assert aspect_path.exists()
        assert dzdx_path.exists()
        assert dzdy_path.exists()
        
        # 验证文件内容
        with rasterio.open(slope_mag_path) as src:
            saved_slope_mag = src.read(1)
            assert np.allclose(saved_slope_mag, slope_mag)
        
        with rasterio.open(aspect_path) as src:
            saved_aspect = src.read(1)
            assert np.allclose(saved_aspect, aspect)
    
    def test_invalid_dem_file(self):
        """测试无效DEM文件处理"""
        with pytest.raises(FileNotFoundError):
            TerrainAnalyzer("nonexistent.tif")
    
    def test_flat_dem(self, tmp_path):
        """测试平坦DEM的处理"""
        # 创建一个完全平坦的DEM
        flat_dem = np.full((3, 3), 100, dtype=np.float32)
        flat_dem_path = tmp_path / "flat_dem.tif"
        
        transform = rasterio.transform.from_origin(0, 0, 30, 30)
        with rasterio.open(
            flat_dem_path,
            'w',
            driver='GTiff',
            height=3,
            width=3,
            count=1,
            dtype=np.float32,
            crs='+proj=latlong',
            transform=transform
        ) as dst:
            dst.write(flat_dem, 1)
        
        analyzer = TerrainAnalyzer(flat_dem_path)
        slope_mag = analyzer.calculate_slope_magnitude()
        aspect = analyzer.calculate_slope_aspect()
        
        assert np.allclose(slope_mag, 0)  # 平坦地形的坡度应该为0
        assert np.all(aspect == -1)  # 平坦地形的坡向应该为-1 