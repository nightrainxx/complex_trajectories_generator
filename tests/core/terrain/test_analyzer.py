"""
地形分析器单元测试
"""

import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.terrain import TerrainAnalyzer
from src.utils.config import config

class TestTerrainAnalyzer(unittest.TestCase):
    """地形分析器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.analyzer = TerrainAnalyzer()
        
        # 创建测试用DEM数据
        self.test_dem = np.array([
            [100, 110, 120],
            [105, 115, 125],
            [110, 120, 130]
        ])
        
        # 模拟rasterio.open
        self.rasterio_mock = MagicMock()
        self.rasterio_mock.read.return_value = self.test_dem
        self.rasterio_mock.transform = [30.0, 0.0, 0.0, 0.0, -30.0, 0.0]
        
    def test_load_dem(self):
        """测试加载DEM数据"""
        with patch('rasterio.open', return_value=self.rasterio_mock):
            self.analyzer.load_dem(Path('test.tif'))
            
        np.testing.assert_array_equal(
            self.analyzer.dem_data,
            self.test_dem
        )
        self.assertEqual(self.analyzer.resolution, 30.0)
        
    def test_calculate_slope_magnitude(self):
        """测试计算坡度大小"""
        self.analyzer.dem_data = self.test_dem
        self.analyzer.resolution = 30.0
        
        self.analyzer.calculate_slope_magnitude()
        
        # 验证坡度计算结果
        self.assertIsNotNone(self.analyzer.slope_magnitude)
        self.assertEqual(
            self.analyzer.slope_magnitude.shape,
            self.test_dem.shape
        )
        
        # 验证梯度计算结果
        self.assertIsNotNone(self.analyzer.dzdx)
        self.assertIsNotNone(self.analyzer.dzdy)
        
    def test_calculate_slope_aspect(self):
        """测试计算坡向"""
        self.analyzer.dem_data = self.test_dem
        self.analyzer.resolution = 30.0
        self.analyzer.calculate_slope_magnitude()
        
        self.analyzer.calculate_slope_aspect()
        
        # 验证坡向计算结果
        self.assertIsNotNone(self.analyzer.slope_aspect)
        self.assertEqual(
            self.analyzer.slope_aspect.shape,
            self.test_dem.shape
        )
        
        # 验证坡向范围
        self.assertTrue(np.all(
            (self.analyzer.slope_aspect >= -1) &
            (self.analyzer.slope_aspect < 360)
        ))
        
    def test_get_terrain_attributes(self):
        """测试获取地形属性"""
        self.analyzer.dem_data = self.test_dem
        self.analyzer.resolution = 30.0
        self.analyzer.calculate_slope_magnitude()
        self.analyzer.calculate_slope_aspect()
        
        slope, aspect = self.analyzer.get_terrain_attributes(1, 1)
        
        self.assertIsInstance(slope, float)
        self.assertIsInstance(aspect, float)
        self.assertTrue(0 <= slope <= 90)
        self.assertTrue(-1 <= aspect < 360)
        
    def test_save_results(self):
        """测试保存计算结果"""
        self.analyzer.dem_data = self.test_dem
        self.analyzer.resolution = 30.0
        self.analyzer.calculate_slope_magnitude()
        self.analyzer.calculate_slope_aspect()
        
        with patch('rasterio.open') as mock_open:
            self.analyzer.save_results()
            
            # 验证是否调用了rasterio.open保存结果
            self.assertEqual(mock_open.call_count, 2)
            
if __name__ == '__main__':
    unittest.main() 