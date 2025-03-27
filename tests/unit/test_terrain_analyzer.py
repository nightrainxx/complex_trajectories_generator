"""地形分析器单元测试"""

import pytest
import numpy as np
import rasterio
from pathlib import Path

from src.generator.terrain_analyzer import TerrainAnalyzer

@pytest.fixture
def test_dem_data():
    """创建测试用DEM数据"""
    # 创建3000x3000的测试DEM数据
    dem = np.zeros((3000, 3000), dtype=np.float32)
    
    # 添加一些地形特征
    # 1. 平坦区域(高度=0)
    # 2. 斜坡(线性变化)
    dem[1000:1100, 1000:1100] = np.linspace(0, 100, 100).reshape(-1, 1)
    # 3. 山峰(高斯分布)
    x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    mountain = 500 * np.exp(-(x**2 + y**2) / 8)
    dem[2000:2100, 2000:2100] = mountain
    
    return dem

@pytest.fixture
def test_dem_file(tmp_path, test_dem_data):
    """创建测试用DEM文件"""
    # 创建测试用GeoTIFF文件
    dem_path = tmp_path / "test_dem.tif"
    transform = rasterio.transform.from_origin(-6.0, 58.0, 0.001, 0.001)
    
    meta = {
        'driver': 'GTiff',
        'height': 3000,
        'width': 3000,
        'count': 1,
        'dtype': np.float32,
        'crs': '+proj=latlong',
        'transform': transform
    }
    
    with rasterio.open(dem_path, 'w', **meta) as dst:
        dst.write(test_dem_data, 1)
    
    return dem_path

class TestTerrainAnalyzer:
    """地形分析器测试类"""
    
    def test_init(self, test_dem_file):
        """测试初始化"""
        analyzer = TerrainAnalyzer(test_dem_file)
        assert analyzer.dem_data is not None
        assert analyzer.transform is not None
        assert analyzer.meta is not None
        assert analyzer.pixel_width > 0
        assert analyzer.pixel_height > 0
    
    def test_calculate_slope_magnitude(self, test_dem_file):
        """测试坡度大小计算"""
        analyzer = TerrainAnalyzer(test_dem_file)
        slope = analyzer.calculate_slope_magnitude()
        
        # 验证形状
        assert slope.shape == analyzer.dem_data.shape
        
        # 验证平坦区域坡度为0
        assert np.allclose(slope[0:100, 0:100], 0, atol=0.1)
        
        # 验证斜坡区域坡度
        slope_area = slope[1000:1100, 1000:1100]
        assert np.all(slope_area > 0)  # 坡度应该大于0
        assert np.all(slope_area < 90)  # 坡度应该小于90度
        
        # 验证山峰区域坡度变化
        mountain_slope = slope[2000:2100, 2000:2100]
        assert np.all(mountain_slope >= 0)  # 坡度应该非负
        assert np.all(mountain_slope <= 90)  # 坡度应该不超过90度
        # 山顶坡度应该接近0
        assert np.mean(mountain_slope[45:55, 45:55]) < 5
    
    def test_calculate_slope_aspect(self, test_dem_file):
        """测试坡向计算"""
        analyzer = TerrainAnalyzer(test_dem_file)
        aspect = analyzer.calculate_slope_aspect()
        
        # 验证形状
        assert aspect.shape == analyzer.dem_data.shape
        
        # 验证平坦区域坡向为-1
        assert np.all(aspect[0:100, 0:100] == -1)
        
        # 验证斜坡区域坡向
        slope_aspect = aspect[1000:1100, 1000:1100]
        valid_aspect = slope_aspect[slope_aspect != -1]
        assert np.all(valid_aspect >= 0)  # 坡向应该在0-360度之间
        assert np.all(valid_aspect < 360)
        
        # 验证山峰区域坡向变化
        mountain_aspect = aspect[2000:2100, 2000:2100]
        valid_mountain = mountain_aspect[mountain_aspect != -1]
        assert np.all(valid_mountain >= 0)
        assert np.all(valid_mountain < 360)
    
    def test_save_terrain_maps(self, test_dem_file, tmp_path):
        """测试地形属性地图保存"""
        analyzer = TerrainAnalyzer(test_dem_file)
        
        # 保存地图
        slope_path, aspect_path = analyzer.save_terrain_maps(str(tmp_path))
        
        # 验证文件是否存在
        assert Path(slope_path).exists()
        assert Path(aspect_path).exists()
        
        # 验证文件内容
        with rasterio.open(slope_path) as src:
            slope = src.read(1)
            assert slope.shape == analyzer.dem_data.shape
            assert np.all(slope >= 0)
            assert np.all(slope <= 90)
        
        with rasterio.open(aspect_path) as src:
            aspect = src.read(1)
            assert aspect.shape == analyzer.dem_data.shape
            valid_aspect = aspect[aspect != -1]
            assert np.all(valid_aspect >= 0)
            assert np.all(valid_aspect < 360)
    
    def test_invalid_file(self):
        """测试无效文件处理"""
        with pytest.raises(FileNotFoundError):
            TerrainAnalyzer("nonexistent_dem.tif") 