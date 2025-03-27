"""
数据加载器模块的单元测试
"""

import os
import unittest
from pathlib import Path

import numpy as np
import pytest
from rasterio.errors import RasterioIOError

from src.data_processing.data_loader import GISDataLoader

class TestGISDataLoader(unittest.TestCase):
    """测试GIS数据加载器类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.loader = GISDataLoader()
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
    
    def test_load_dem_file_not_found(self):
        """测试加载不存在的DEM文件时的错误处理"""
        with pytest.raises(RasterioIOError):
            self.loader.load_dem(self.test_data_dir / "not_exists.tif")
    
    def test_get_pixel_coords_without_transform(self):
        """测试在未加载数据时获取像素坐标的错误处理"""
        with pytest.raises(ValueError, match="未加载GIS数据"):
            self.loader.get_pixel_coords(116.0, 40.0)
    
    def test_get_elevation_without_data(self):
        """测试在未加载DEM数据时获取高程值的错误处理"""
        with pytest.raises(ValueError, match="未加载DEM数据"):
            self.loader.get_elevation(0, 0)
    
    def test_get_slope_without_data(self):
        """测试在未加载坡度数据时获取坡度值的错误处理"""
        with pytest.raises(ValueError, match="未加载坡度数据"):
            self.loader.get_slope(0, 0)
    
    def test_get_landcover_without_data(self):
        """测试在未加载土地覆盖数据时获取土地覆盖类型的错误处理"""
        with pytest.raises(ValueError, match="未加载土地覆盖数据"):
            self.loader.get_landcover(0, 0)

if __name__ == '__main__':
    unittest.main() 