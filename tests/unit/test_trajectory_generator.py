"""
轨迹生成器模块的单元测试
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.analysis import EnvironmentAnalyzer
from src.data_processing import GISDataLoader
from src.generator import TrajectoryGenerator

class TestTrajectoryGenerator(unittest.TestCase):
    """测试轨迹生成器类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建GISDataLoader的Mock对象
        self.mock_gis_loader = MagicMock(spec=GISDataLoader)
        
        # 创建EnvironmentAnalyzer的Mock对象
        self.mock_env_analyzer = MagicMock(spec=EnvironmentAnalyzer)
        
        # 设置mock返回值
        self.mock_gis_loader.get_pixel_coords.return_value = (100, 100)
        self.mock_gis_loader.get_elevation.return_value = 100.0
        self.mock_gis_loader.get_slope.return_value = 10.0
        self.mock_gis_loader.get_landcover.return_value = 1
        
        self.mock_env_analyzer.sample_speed.return_value = 10.0
        
        # 创建生成器实例
        self.generator = TrajectoryGenerator(
            gis_loader=self.mock_gis_loader,
            env_analyzer=self.mock_env_analyzer
        )
        
        # 设置测试用的时间和坐标
        self.start_time = pd.Timestamp('2024-01-01 12:00:00')
        self.start_point = (116.0, 40.0)
        self.end_point = (116.1, 40.1)
        self.region_bounds = (116.0, 40.0, 116.1, 40.1)
        self.time_range = (
            pd.Timestamp('2024-01-01 12:00:00'),
            pd.Timestamp('2024-01-01 13:00:00')
        )
    
    def test_generate_trajectory(self):
        """测试单条轨迹生成"""
        trajectory = self.generator.generate_trajectory(
            start_point=self.start_point,
            end_point=self.end_point,
            start_time=self.start_time
        )
        
        # 验证轨迹数据格式
        self.assertIsInstance(trajectory, pd.DataFrame)
        required_columns = [
            'timestamp', 'longitude', 'latitude', 'elevation',
            'speed', 'heading', 'turn_rate', 'acceleration'
        ]
        self.assertTrue(all(col in trajectory.columns for col in required_columns))
        
        # 验证轨迹起点和终点
        self.assertAlmostEqual(trajectory.iloc[0]['longitude'], self.start_point[0], places=6)
        self.assertAlmostEqual(trajectory.iloc[0]['latitude'], self.start_point[1], places=6)
        self.assertAlmostEqual(trajectory.iloc[-1]['longitude'], self.end_point[0], places=6)
        self.assertAlmostEqual(trajectory.iloc[-1]['latitude'], self.end_point[1], places=6)
        
        # 验证时间戳
        self.assertEqual(trajectory.iloc[0]['timestamp'], self.start_time)
        self.assertTrue(all(trajectory['timestamp'].diff()[1:] > pd.Timedelta(0)))
    
    def test_generate_trajectories(self):
        """测试批量轨迹生成"""
        num_trajectories = 5
        trajectories = self.generator.generate_trajectories(
            num_trajectories=num_trajectories,
            region_bounds=self.region_bounds,
            time_range=self.time_range
        )
        
        # 验证生成的轨迹数量
        self.assertEqual(len(trajectories), num_trajectories)
        
        # 验证每条轨迹的格式和内容
        for traj_id, trajectory in trajectories.items():
            self.assertIsInstance(trajectory, pd.DataFrame)
            required_columns = [
                'timestamp', 'longitude', 'latitude', 'elevation',
                'speed', 'heading', 'turn_rate', 'acceleration'
            ]
            self.assertTrue(all(col in trajectory.columns for col in required_columns))
            
            # 验证坐标在区域范围内
            self.assertTrue(all(trajectory['longitude'] >= self.region_bounds[0]))
            self.assertTrue(all(trajectory['longitude'] <= self.region_bounds[2]))
            self.assertTrue(all(trajectory['latitude'] >= self.region_bounds[1]))
            self.assertTrue(all(trajectory['latitude'] <= self.region_bounds[3]))
            
            # 验证时间在范围内
            self.assertTrue(all(trajectory['timestamp'] >= self.time_range[0]))
            self.assertTrue(all(trajectory['timestamp'] <= self.time_range[1]))
    
    def test_plan_path(self):
        """测试路径规划"""
        waypoints = self.generator._plan_path(
            start_point=self.start_point,
            end_point=self.end_point
        )
        
        # 验证路径点格式
        self.assertIsInstance(waypoints, list)
        self.assertTrue(all(isinstance(point, tuple) and len(point) == 2
                          for point in waypoints))
        
        # 验证起点和终点
        self.assertEqual(waypoints[0], self.start_point)
        self.assertEqual(waypoints[-1], self.end_point)
        
        # 验证路径点间距合理
        for i in range(len(waypoints) - 1):
            p1, p2 = waypoints[i], waypoints[i + 1]
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            self.assertLess(distance, 0.01)  # 相邻点间距不超过约1km
    
    def test_generate_motion(self):
        """测试运动生成"""
        waypoints = [
            self.start_point,
            (116.05, 40.05),
            self.end_point
        ]
        
        motion = self.generator._generate_motion(
            waypoints=waypoints,
            start_time=self.start_time
        )
        
        # 验证运动数据格式
        self.assertIsInstance(motion, pd.DataFrame)
        required_columns = [
            'timestamp', 'longitude', 'latitude', 'elevation',
            'speed', 'heading', 'turn_rate', 'acceleration'
        ]
        self.assertTrue(all(col in motion.columns for col in required_columns))
        
        # 验证运动参数合理性
        self.assertTrue(all(motion['speed'] >= 0))  # 速度非负
        self.assertTrue(all(motion['heading'] >= 0) and all(motion['heading'] < 360))  # 航向角在[0,360)范围内
        self.assertTrue(all(abs(motion['turn_rate']) <= 45))  # 转向率不超过45度/秒
        self.assertTrue(all(abs(motion['acceleration']) <= 5))  # 加速度不超过5m/s²

if __name__ == '__main__':
    unittest.main() 