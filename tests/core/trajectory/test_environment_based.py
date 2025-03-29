"""
基于环境的轨迹生成器单元测试
"""

import unittest
import numpy as np
from unittest.mock import MagicMock

from src.core.terrain import TerrainLoader
from src.core.trajectory import EnvironmentBasedGenerator
from src.utils.config import config

class TestEnvironmentBasedGenerator(unittest.TestCase):
    """基于环境的轨迹生成器测试"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟的地形加载器
        self.terrain_loader = MagicMock(spec=TerrainLoader)
        self.terrain_loader.get_terrain_attributes.return_value = {
            'slope': 5.0,
            'landcover': 2
        }
        self.terrain_loader.is_valid_pixel.return_value = True
        self.terrain_loader.is_passable.return_value = True
        self.terrain_loader.coord_to_pixel.return_value = (1, 1)
        
        # 创建轨迹生成器
        self.generator = EnvironmentBasedGenerator(
            terrain_loader=self.terrain_loader,
            dt=1.0,
            max_waypoints=5,
            min_waypoint_dist=10.0,
            max_waypoint_dist=50.0
        )
        
    def test_generate_waypoints(self):
        """测试路径点生成"""
        start_point = (0.0, 0.0)
        end_point = (100.0, 100.0)
        
        waypoints = self.generator._generate_waypoints(
            start_point,
            end_point
        )
        
        # 验证路径点数量
        self.assertGreaterEqual(len(waypoints), 2)  # 至少包含起点和终点
        self.assertLessEqual(len(waypoints), 7)     # 不超过max_waypoints+2
        
        # 验证起点和终点
        self.assertEqual(waypoints[0], start_point)
        self.assertEqual(waypoints[-1], end_point)
        
        # 验证中间点的间距
        for i in range(len(waypoints)-1):
            p1 = np.array(waypoints[i])
            p2 = np.array(waypoints[i+1])
            dist = np.sqrt(np.sum((p2 - p1)**2))
            self.assertGreaterEqual(dist, self.generator.min_waypoint_dist)
            self.assertLessEqual(dist, self.generator.max_waypoint_dist)
            
    def test_generate_trajectory(self):
        """测试轨迹生成"""
        start_point = (0.0, 0.0)
        end_point = (100.0, 100.0)
        
        trajectory = self.generator.generate_trajectory(
            start_point,
            end_point
        )
        
        # 验证轨迹数据格式
        self.assertIn('timestamp', trajectory)
        self.assertIn('x', trajectory)
        self.assertIn('y', trajectory)
        self.assertIn('speed', trajectory)
        self.assertIn('orientation', trajectory)
        
        # 验证数据长度一致
        length = len(trajectory['timestamp'])
        self.assertEqual(len(trajectory['x']), length)
        self.assertEqual(len(trajectory['y']), length)
        self.assertEqual(len(trajectory['speed']), length)
        self.assertEqual(len(trajectory['orientation']), length)
        
        # 验证起点和终点
        self.assertAlmostEqual(trajectory['x'][0], start_point[0])
        self.assertAlmostEqual(trajectory['y'][0], start_point[1])
        self.assertAlmostEqual(trajectory['x'][-1], end_point[0])
        self.assertAlmostEqual(trajectory['y'][-1], end_point[1])
        
        # 验证速度范围
        speeds = np.array(trajectory['speed'])
        self.assertTrue(np.all(speeds >= config.motion.MIN_SPEED))
        self.assertTrue(np.all(speeds <= config.motion.MAX_SPEED))
        
        # 验证朝向范围
        orientations = np.array(trajectory['orientation'])
        self.assertTrue(np.all(orientations >= 0))
        self.assertTrue(np.all(orientations < 360))
        
        # 验证时间戳递增
        timestamps = np.array(trajectory['timestamp'])
        self.assertTrue(np.all(np.diff(timestamps) > 0))
        
    def test_invalid_terrain(self):
        """测试无效地形情况"""
        # 模拟无效地形
        self.terrain_loader.is_valid_pixel.return_value = False
        
        start_point = (0.0, 0.0)
        end_point = (100.0, 100.0)
        
        # 验证是否仍能生成轨迹
        trajectory = self.generator.generate_trajectory(
            start_point,
            end_point
        )
        
        self.assertIsNotNone(trajectory)
        self.assertTrue(all(key in trajectory for key in [
            'timestamp', 'x', 'y', 'speed', 'orientation'
        ]))
        
    def test_impassable_terrain(self):
        """测试不可通行地形情况"""
        # 模拟部分地形不可通行
        self.terrain_loader.is_passable.side_effect = [
            True,   # 起点可通行
            False,  # 中间点不可通行
            True    # 终点可通行
        ]
        
        start_point = (0.0, 0.0)
        end_point = (100.0, 100.0)
        
        # 验证是否仍能生成轨迹
        trajectory = self.generator.generate_trajectory(
            start_point,
            end_point
        )
        
        self.assertIsNotNone(trajectory)
        self.assertTrue(all(key in trajectory for key in [
            'timestamp', 'x', 'y', 'speed', 'orientation'
        ]))
        
if __name__ == '__main__':
    unittest.main() 