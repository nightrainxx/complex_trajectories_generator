"""
起终点选择器单元测试
"""

import unittest
import numpy as np
from unittest.mock import MagicMock

from src.core.terrain import TerrainLoader
from src.core.point_selector import PointSelector
from src.utils.config import config

class TestPointSelector(unittest.TestCase):
    """起终点选择器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟的地形加载器
        self.terrain_loader = MagicMock(spec=TerrainLoader)
        
        # 设置DEM数据形状
        self.dem_shape = (100, 100)
        self.terrain_loader.dem_data = np.zeros(self.dem_shape)
        
        # 设置土地覆盖数据
        self.landcover_data = np.ones(self.dem_shape)
        self.terrain_loader.landcover_data = self.landcover_data
        
        # 设置坐标转换
        self.terrain_loader.pixel_to_coord.side_effect = lambda r, c: (float(c), float(r))
        
        # 设置可通行性检查
        self.terrain_loader.is_passable.return_value = True
        
        # 创建点选择器
        self.selector = PointSelector(
            terrain_loader=self.terrain_loader,
            min_distance=10.0,
            num_end_points=2,
            num_trajectories=4
        )
        
        # 设置城市地物编码
        config.terrain.URBAN_LANDCOVER_CODES = [1]
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.selector.min_distance, 10.0)
        self.assertEqual(self.selector.num_end_points, 2)
        self.assertEqual(self.selector.num_trajectories, 4)
        self.assertEqual(self.selector.starts_per_end, 2)
        
    def test_select_points(self):
        """测试起终点选择"""
        pairs = self.selector.select_points()
        
        # 验证返回的起终点对数量
        self.assertEqual(len(pairs), 4)
        
        # 验证每个起终点对的格式
        for start, end in pairs:
            self.assertIsInstance(start, tuple)
            self.assertIsInstance(end, tuple)
            self.assertEqual(len(start), 2)
            self.assertEqual(len(end), 2)
            
            # 验证起终点距离
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            distance = np.sqrt(dx**2 + dy**2)
            self.assertGreaterEqual(distance, self.selector.min_distance)
            
    def test_select_points_no_urban(self):
        """测试无城市区域情况"""
        # 设置所有地物编码为非城市
        self.landcover_data.fill(0)
        
        # 验证是否抛出异常
        with self.assertRaises(RuntimeError):
            self.selector.select_points()
            
    def test_select_points_impassable(self):
        """测试不可通行情况"""
        # 设置所有点不可通行
        self.terrain_loader.is_passable.return_value = False
        
        # 验证是否抛出异常
        with self.assertRaises(RuntimeError):
            self.selector.select_points()
            
    def test_is_urban_area(self):
        """测试城市区域判断"""
        # 设置部分区域为城市
        self.landcover_data[50:60, 50:60] = 1
        
        # 验证城市区域判断
        self.assertTrue(self.selector._is_urban_area(55, 55))
        self.assertFalse(self.selector._is_urban_area(0, 0))
        
    def test_check_distance(self):
        """测试距离检查"""
        point1 = (0.0, 0.0)
        point2 = (8.0, 6.0)  # 距离为10
        
        self.assertTrue(self.selector._check_distance(point1, point2))
        
        point3 = (3.0, 4.0)  # 距离为5
        self.assertFalse(self.selector._check_distance(point1, point3))
        
    def test_is_too_close_to_points(self):
        """测试点间距检查"""
        points = [(0.0, 0.0), (10.0, 10.0)]
        
        # 测试太近的点
        point1 = (0.1, 0.1)  # 距离约0.14
        self.assertTrue(
            self.selector._is_too_close_to_points(point1, points)
        )
        
        # 测试足够远的点
        point2 = (5.0, 5.0)  # 距离约7.07
        self.assertFalse(
            self.selector._is_too_close_to_points(point2, points)
        )
        
if __name__ == '__main__':
    unittest.main() 