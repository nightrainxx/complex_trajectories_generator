"""
轨迹加载器模块的单元测试
"""

import os
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_processing.trajectory_loader import TrajectoryLoader

class TestTrajectoryLoader(unittest.TestCase):
    """测试轨迹加载器类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.loader = TrajectoryLoader()
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # 创建测试用的轨迹数据
        self.create_test_trajectory()
    
    def create_test_trajectory(self):
        """创建用于测试的轨迹数据文件"""
        # 生成一个简单的轨迹
        timestamps = [
            datetime(2024, 1, 1, 12, 0) + timedelta(seconds=i)
            for i in range(10)
        ]
        
        data = {
            'timestamp': timestamps,
            'longitude': [116.0 + i*0.001 for i in range(10)],
            'latitude': [40.0 + i*0.001 for i in range(10)]
        }
        
        df = pd.DataFrame(data)
        
        # 保存测试轨迹
        test_file = self.test_data_dir / "test_trajectory.csv"
        df.to_csv(test_file, index=False)
    
    def test_load_trajectory(self):
        """测试加载单个轨迹文件"""
        test_file = self.test_data_dir / "test_trajectory.csv"
        df = self.loader.load_trajectory(test_file)
        
        self.assertEqual(len(df), 10)
        self.assertTrue('timestamp' in df.columns)
        self.assertTrue('longitude' in df.columns)
        self.assertTrue('latitude' in df.columns)
        self.assertTrue(isinstance(df['timestamp'].iloc[0], pd.Timestamp))
    
    def test_load_trajectory_missing_columns(self):
        """测试加载缺少必要列的轨迹文件"""
        # 创建缺少列的数据
        df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'longitude': [116.0]
            # 缺少latitude列
        })
        
        test_file = self.test_data_dir / "invalid_trajectory.csv"
        df.to_csv(test_file, index=False)
        
        with pytest.raises(ValueError, match="轨迹文件缺少必要的列"):
            self.loader.load_trajectory(test_file)
    
    def test_preprocess_trajectory(self):
        """测试轨迹预处理功能"""
        # 先加载测试轨迹
        test_file = self.test_data_dir / "test_trajectory.csv"
        self.loader.load_trajectory(test_file)
        
        # 预处理轨迹
        trajectory_id = "test_trajectory"
        df_processed = self.loader.preprocess_trajectory(trajectory_id)
        
        # 验证计算的特征
        self.assertTrue('speed' in df_processed.columns)
        self.assertTrue('heading' in df_processed.columns)
        self.assertTrue('turn_rate' in df_processed.columns)
        self.assertTrue('acceleration' in df_processed.columns)
        
        # 验证速度计算
        speeds = df_processed['speed'].dropna()
        self.assertTrue(all(speeds >= 0))  # 速度应该非负
    
    def test_haversine_distance(self):
        """测试Haversine距离计算"""
        # 测试已知距离的两点
        point1 = np.array([[116.0, 40.0]])  # 北京附近的点
        point2 = np.array([[116.1, 40.0]])  # 约8.5公里
        
        distance = self.loader._haversine_distance(point1, point2)[0]
        
        # 允许1%的误差
        self.assertAlmostEqual(distance, 8500, delta=85)
    
    def test_calculate_heading(self):
        """测试方向角计算"""
        # 创建一个向正东方向移动的轨迹
        coords = np.array([
            [116.0, 40.0],
            [116.1, 40.0]
        ])
        
        headings = self.loader._calculate_heading(coords)
        
        # 向东移动应该是90度（允许一定误差）
        self.assertAlmostEqual(headings[0], 90.0, delta=1.0)
        self.assertAlmostEqual(headings[1], 90.0, delta=1.0)
    
    def tearDown(self):
        """测试后的清理工作"""
        # 删除测试文件
        for file in self.test_data_dir.glob("*.csv"):
            file.unlink()
        
        # 删除测试目录（如果为空）
        try:
            self.test_data_dir.rmdir()
        except OSError:
            pass  # 目录不为空或其他原因无法删除时忽略

if __name__ == '__main__':
    unittest.main() 