"""
路径规划器模块的单元测试
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.data_processing import GISDataLoader
from src.generator.path_planner import PathPlanner

class TestPathPlanner(unittest.TestCase):
    """测试路径规划器类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建GISDataLoader的Mock对象
        self.mock_gis_loader = MagicMock(spec=GISDataLoader)
        
        # 设置mock返回值
        self.mock_gis_loader.get_elevation.return_value = 100.0
        self.mock_gis_loader.get_slope.return_value = 10.0
        self.mock_gis_loader.get_landcover.return_value = 3  # 可通行的土地类型
        
        # 创建规划器实例
        self.planner = PathPlanner(self.mock_gis_loader)
        
        # 设置测试用的起点和终点
        self.start_point = (116.0, 40.0)
        self.end_point = (116.1, 40.1)
    
    def test_plan_path(self):
        """测试基本的路径规划功能"""
        path = self.planner.plan_path(self.start_point, self.end_point)
        
        # 验证路径的基本属性
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 1)
        self.assertEqual(path[0], self.start_point)
        self.assertEqual(path[-1], self.end_point)
        
        # 验证相邻点之间的距离不超过步长
        for i in range(len(path) - 1):
            distance = self.planner._calculate_distance(path[i], path[i + 1])
            self.assertLessEqual(distance, self.planner.step_size * 1.5)
    
    def test_plan_path_with_obstacles(self):
        """测试有障碍物时的路径规划"""
        # 修改mock，使某些位置不可通行
        def mock_get_landcover(lon, lat):
            # 在中间位置设置障碍物
            if 116.04 <= lon <= 116.06 and 40.04 <= lat <= 40.06:
                return 1  # 水体
            return 3  # 可通行区域
        
        self.mock_gis_loader.get_landcover.side_effect = mock_get_landcover
        
        path = self.planner.plan_path(
            start_point=self.start_point,
            end_point=self.end_point
        )
        
        # 验证路径是否绕开了障碍物
        for lon, lat in path:
            if 116.04 <= lon <= 116.06 and 40.04 <= lat <= 40.06:
                self.fail("路径穿过了障碍物区域")
    
    def test_plan_path_with_steep_slope(self):
        """测试有陡坡时的路径规划"""
        # 修改mock，使某些位置坡度过大
        def mock_get_slope(lon, lat):
            # 在中间位置设置陡坡
            if 116.04 <= lon <= 116.06 and 40.04 <= lat <= 40.06:
                return 35.0  # 过陡的坡度
            return 10.0  # 正常坡度
        
        self.mock_gis_loader.get_slope.side_effect = mock_get_slope
        
        path = self.planner.plan_path(
            start_point=self.start_point,
            end_point=self.end_point
        )
        
        # 验证路径是否绕开了陡坡
        for lon, lat in path:
            if 116.04 <= lon <= 116.06 and 40.04 <= lat <= 40.06:
                self.fail("路径穿过了陡坡区域")
    
    def test_get_neighbors(self):
        """测试获取相邻节点的功能"""
        position = (116.0, 40.0)
        neighbors = self.planner._get_neighbors(position)
        
        # 验证相邻节点的数量和位置
        self.assertEqual(len(neighbors), 8)  # 应该有8个相邻节点
        for neighbor in neighbors:
            distance = self.planner._calculate_distance(position, neighbor)
            self.assertLessEqual(distance, self.planner.step_size * 1.5)
    
    def test_is_valid_position(self):
        """测试位置可通行性检查"""
        # 测试正常位置
        position = (116.0, 40.0)
        self.assertTrue(self.planner._is_valid_position(position))
        
        # 测试水体位置
        self.mock_gis_loader.get_landcover.return_value = 1
        self.assertFalse(self.planner._is_valid_position(position))
        
        # 测试陡坡位置
        self.mock_gis_loader.get_landcover.return_value = 3
        self.mock_gis_loader.get_slope.return_value = 35.0
        self.assertFalse(self.planner._is_valid_position(position))

if __name__ == '__main__':
    unittest.main() 