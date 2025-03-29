"""
轨迹生成器基类单元测试
"""

import unittest
import numpy as np
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

from src.core.terrain import TerrainLoader
from src.core.trajectory import TrajectoryGenerator
from src.utils.config import config

class TestTrajectoryGenerator(TrajectoryGenerator):
    """用于测试的轨迹生成器实现"""
    
    def generate_trajectory(
            self,
            start_point: Tuple[float, float],
            end_point: Tuple[float, float]
        ) -> Dict[str, List[float]]:
        """测试用轨迹生成方法"""
        waypoints = [start_point, end_point]
        x, y = self._interpolate_path(waypoints)
        orientations = self._calculate_orientations(x, y)
        speeds = self._calculate_speeds(x, y)
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        timestamps = self._calculate_timestamps(distances, speeds)
        
        return {
            'timestamp': timestamps.tolist(),
            'x': x.tolist(),
            'y': y.tolist(),
            'speed': speeds.tolist(),
            'orientation': orientations.tolist()
        }

class TestTrajectoryGeneratorBase(unittest.TestCase):
    """轨迹生成器基类测试"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟的地形加载器
        self.terrain_loader = MagicMock(spec=TerrainLoader)
        self.terrain_loader.get_terrain_attributes.return_value = {
            'slope': 5.0,
            'landcover': 2
        }
        self.terrain_loader.is_valid_pixel.return_value = True
        self.terrain_loader.coord_to_pixel.return_value = (1, 1)
        
        # 创建测试用生成器
        self.generator = TestTrajectoryGenerator(
            self.terrain_loader,
            dt=1.0
        )
        
    def test_interpolate_path(self):
        """测试路径插值"""
        waypoints = [
            (0.0, 0.0),
            (100.0, 100.0)
        ]
        
        x, y = self.generator._interpolate_path(waypoints, num_points=5)
        
        self.assertEqual(len(x), 5)
        self.assertEqual(len(y), 5)
        self.assertEqual(x[0], 0.0)
        self.assertEqual(y[0], 0.0)
        self.assertEqual(x[-1], 100.0)
        self.assertEqual(y[-1], 100.0)
        
    def test_calculate_orientations(self):
        """测试朝向计算"""
        x = np.array([0.0, 100.0, 100.0])
        y = np.array([0.0, 100.0, 200.0])
        
        orientations = self.generator._calculate_orientations(x, y)
        
        self.assertEqual(len(orientations), 3)
        self.assertTrue(np.all(orientations >= 0))
        self.assertTrue(np.all(orientations < 360))
        
    def test_calculate_speeds(self):
        """测试速度计算"""
        x = np.array([0.0, 100.0, 200.0])
        y = np.array([0.0, 100.0, 200.0])
        
        # 测试不考虑地形影响
        speeds = self.generator._calculate_speeds(x, y, terrain_influence=False)
        
        self.assertEqual(len(speeds), 3)
        self.assertTrue(np.all(speeds >= config.motion.MIN_SPEED))
        self.assertTrue(np.all(speeds <= config.motion.MAX_SPEED))
        
        # 测试考虑地形影响
        speeds = self.generator._calculate_speeds(x, y, terrain_influence=True)
        
        self.assertEqual(len(speeds), 3)
        self.assertTrue(np.all(speeds >= config.motion.MIN_SPEED))
        self.assertTrue(np.all(speeds <= config.motion.MAX_SPEED))
        
    def test_calculate_timestamps(self):
        """测试时间戳计算"""
        distances = np.array([100.0, 100.0])
        speeds = np.array([10.0, 10.0, 10.0])
        
        timestamps = self.generator._calculate_timestamps(distances, speeds)
        
        self.assertEqual(len(timestamps), 3)
        self.assertEqual(timestamps[0], 0.0)
        self.assertEqual(timestamps[1], 10.0)
        self.assertEqual(timestamps[2], 20.0)
        
    def test_generate_trajectory(self):
        """测试轨迹生成"""
        start_point = (0.0, 0.0)
        end_point = (100.0, 100.0)
        
        trajectory = self.generator.generate_trajectory(
            start_point,
            end_point
        )
        
        self.assertIn('timestamp', trajectory)
        self.assertIn('x', trajectory)
        self.assertIn('y', trajectory)
        self.assertIn('speed', trajectory)
        self.assertIn('orientation', trajectory)
        
        self.assertEqual(len(trajectory['timestamp']),
                        len(trajectory['x']))
        self.assertEqual(len(trajectory['x']),
                        len(trajectory['y']))
        self.assertEqual(len(trajectory['y']),
                        len(trajectory['speed']))
        self.assertEqual(len(trajectory['speed']),
                        len(trajectory['orientation']))
        
if __name__ == '__main__':
    unittest.main() 