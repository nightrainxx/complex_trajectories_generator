"""
地形加载器单元测试
"""

import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.terrain import TerrainLoader
from src.utils.config import config

class TestTerrainLoader(unittest.TestCase):
    """地形加载器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.loader = TerrainLoader()
        
        # 创建测试数据
        self.test_dem = np.array([
            [100, 110, 120],
            [105, 115, 125],
            [110, 120, 130]
        ])
        self.test_landcover = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        self.test_slope = np.array([
            [5, 10, 15],
            [7, 12, 17],
            [9, 14, 19]
        ])
        self.test_aspect = np.array([
            [45, 90, 135],
            [60, 120, 180],
            [90, 150, 225]
        ])
        
        # 模拟rasterio.open
        self.rasterio_mock = MagicMock()
        self.rasterio_mock.transform = [30.0, 0.0, 0.0, 0.0, -30.0, 0.0]
        self.rasterio_mock.crs = 'EPSG:4326'
        
    def test_load_dem(self):
        """测试加载DEM数据"""
        self.rasterio_mock.read.return_value = self.test_dem
        
        with patch('rasterio.open', return_value=self.rasterio_mock):
            self.loader.load_dem(Path('test.tif'))
            
        np.testing.assert_array_equal(
            self.loader.dem_data,
            self.test_dem
        )
        self.assertEqual(self.loader.resolution, 30.0)
        self.assertEqual(self.loader.crs, 'EPSG:4326')
        
    def test_load_landcover(self):
        """测试加载土地覆盖数据"""
        self.rasterio_mock.read.return_value = self.test_landcover
        
        with patch('rasterio.open', return_value=self.rasterio_mock):
            self.loader.load_landcover(Path('test.tif'))
            
        np.testing.assert_array_equal(
            self.loader.landcover_data,
            self.test_landcover
        )
        
    def test_load_slope(self):
        """测试加载坡度数据"""
        self.rasterio_mock.read.return_value = self.test_slope
        
        with patch('rasterio.open', return_value=self.rasterio_mock):
            self.loader.load_slope(Path('test.tif'))
            
        np.testing.assert_array_equal(
            self.loader.slope_data,
            self.test_slope
        )
        
    def test_load_aspect(self):
        """测试加载坡向数据"""
        self.rasterio_mock.read.return_value = self.test_aspect
        
        with patch('rasterio.open', return_value=self.rasterio_mock):
            self.loader.load_aspect(Path('test.tif'))
            
        np.testing.assert_array_equal(
            self.loader.aspect_data,
            self.test_aspect
        )
        
    def test_get_terrain_attributes(self):
        """测试获取地形属性"""
        self.loader.dem_data = self.test_dem
        self.loader.landcover_data = self.test_landcover
        self.loader.slope_data = self.test_slope
        self.loader.aspect_data = self.test_aspect
        
        attrs = self.loader.get_terrain_attributes(1, 1)
        
        self.assertEqual(attrs['elevation'], 115.0)
        self.assertEqual(attrs['landcover'], 5)
        self.assertEqual(attrs['slope'], 12.0)
        self.assertEqual(attrs['aspect'], 120.0)
        
    def test_pixel_to_coord(self):
        """测试像素坐标转地理坐标"""
        self.loader.transform = self.rasterio_mock.transform
        
        x, y = self.loader.pixel_to_coord(1, 1)
        
        self.assertEqual(x, 30.0)
        self.assertEqual(y, -30.0)
        
    def test_coord_to_pixel(self):
        """测试地理坐标转像素坐标"""
        self.loader.transform = self.rasterio_mock.transform
        
        row, col = self.loader.coord_to_pixel(30.0, -30.0)
        
        self.assertEqual(row, 1)
        self.assertEqual(col, 1)
        
    def test_is_valid_pixel(self):
        """测试像素坐标有效性检查"""
        self.loader.dem_data = self.test_dem
        
        self.assertTrue(self.loader.is_valid_pixel(1, 1))
        self.assertFalse(self.loader.is_valid_pixel(3, 3))
        self.assertFalse(self.loader.is_valid_pixel(-1, -1))
        
    def test_is_passable(self):
        """测试可通行性检查"""
        self.loader.dem_data = self.test_dem
        self.loader.landcover_data = self.test_landcover
        self.loader.slope_data = self.test_slope
        
        # 设置不可通行的地物编码和最大坡度
        config.terrain.IMPASSABLE_LANDCOVER = [9]
        config.motion.MAX_SLOPE = 15.0
        
        self.assertTrue(self.loader.is_passable(1, 1))  # 可通行
        self.assertFalse(self.loader.is_passable(2, 2))  # 不可通行（地物编码）
        self.assertFalse(self.loader.is_passable(0, 2))  # 不可通行（坡度）
        
if __name__ == '__main__':
    unittest.main() 