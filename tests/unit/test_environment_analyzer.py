"""
环境分析器模块的单元测试
"""

import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.environment_analyzer import EnvironmentAnalyzer
from src.data_processing import GISDataLoader

class TestEnvironmentAnalyzer(unittest.TestCase):
    """测试环境分析器类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建GISDataLoader的Mock对象
        self.mock_gis_loader = MagicMock(spec=GISDataLoader)
        
        # 设置mock返回值
        self.mock_gis_loader.get_pixel_coords.return_value = (100, 100)
        self.mock_gis_loader.get_elevation.return_value = 100.0
        self.mock_gis_loader.get_slope.return_value = 10.0
        self.mock_gis_loader.get_landcover.return_value = 1
        
        # 创建分析器实例
        self.analyzer = EnvironmentAnalyzer(self.mock_gis_loader)
        
        # 创建测试用的轨迹数据
        self.test_trajectory = self.create_test_trajectory()
    
    def create_test_trajectory(self) -> pd.DataFrame:
        """创建用于测试的轨迹数据"""
        timestamps = [
            datetime(2024, 1, 1, 12, 0) + timedelta(seconds=i)
            for i in range(10)
        ]
        
        data = {
            'timestamp': timestamps,
            'longitude': [116.0 + i*0.001 for i in range(10)],
            'latitude': [40.0 + i*0.001 for i in range(10)],
            'speed': [10.0 + i for i in range(10)],
            'heading': [45.0 for _ in range(10)],
            'turn_rate': [0.0 for _ in range(10)],
            'acceleration': [1.0 for _ in range(10)]
        }
        
        return pd.DataFrame(data)
    
    def test_analyze_trajectory(self):
        """测试单条轨迹的环境分析"""
        enriched_df = self.analyzer.analyze_trajectory(self.test_trajectory)
        
        # 验证是否添加了所有环境特征列
        self.assertTrue('elevation' in enriched_df.columns)
        self.assertTrue('slope' in enriched_df.columns)
        self.assertTrue('landcover' in enriched_df.columns)
        self.assertTrue('slope_class' in enriched_df.columns)
        self.assertTrue('environment_group' in enriched_df.columns)
        
        # 验证环境组标签格式
        self.assertTrue(all(enriched_df['environment_group'].str.match(r'LC\d+_SS\d+')))
    
    def test_compute_environment_statistics(self):
        """测试环境统计计算"""
        # 创建测试数据
        enriched_trajectories = {
            'test_traj': self.test_trajectory.assign(
                elevation=100.0,
                slope=10.0,
                landcover=1,
                slope_class='S1',
                environment_group='LC1_SS1'
            )
        }
        
        stats = self.analyzer.compute_environment_statistics(enriched_trajectories)
        
        # 验证统计结果
        self.assertTrue('LC1_SS1' in stats)
        group_stats = stats['LC1_SS1']
        
        # 验证统计量的完整性
        self.assertTrue('speed' in group_stats)
        self.assertTrue('turn_rate' in group_stats)
        self.assertTrue('acceleration' in group_stats)
        self.assertTrue('sample_size' in group_stats)
        
        # 验证速度统计量
        speed_stats = group_stats['speed']
        self.assertTrue(all(key in speed_stats for key in 
                          ['mean', 'std', 'median', 'q25', 'q75', 'max', 'min']))
    
    def test_fit_speed_models(self):
        """测试速度分布模型拟合"""
        # 创建测试数据
        enriched_trajectories = {
            'test_traj': self.test_trajectory.assign(
                elevation=100.0,
                slope=10.0,
                landcover=1,
                slope_class='S1',
                environment_group='LC1_SS1'
            )
        }
        
        models = self.analyzer.fit_speed_models(enriched_trajectories, min_samples=5)
        
        # 验证模型结果
        self.assertTrue('LC1_SS1' in models)
        model = models['LC1_SS1']
        
        # 验证模型参数的完整性
        self.assertTrue('distribution' in model)
        self.assertTrue('parameters' in model)
        self.assertTrue('ks_statistic' in model)
    
    def test_sample_speed(self):
        """测试速度采样"""
        # 设置测试数据
        self.analyzer.environment_stats = {
            'LC1_SS1': {
                'speed': {
                    'mean': 10.0,
                    'std': 2.0,
                    'min': 5.0,
                    'max': 15.0
                }
            }
        }
        
        # 测试速度采样
        speed = self.analyzer.sample_speed(landcover=1, slope=10.0)
        
        # 验证采样结果
        self.assertTrue(isinstance(speed, float))
        self.assertTrue(5.0 <= speed <= 15.0)  # 速度应在合理范围内
    
    def test_get_environment_group_stats(self):
        """测试获取环境组统计信息"""
        # 设置测试数据
        test_stats = {
            'speed': {'mean': 10.0, 'std': 2.0},
            'turn_rate': {'mean': 0.0, 'std': 1.0},
            'acceleration': {'mean': 1.0, 'std': 0.5},
            'sample_size': 100
        }
        self.analyzer.environment_stats = {'LC1_SS1': test_stats}
        
        # 获取统计信息
        stats = self.analyzer.get_environment_group_stats(landcover=1, slope=10.0)
        
        # 验证结果
        self.assertEqual(stats, test_stats)
    
    def test_get_environment_group_stats_missing(self):
        """测试获取不存在的环境组统计信息"""
        stats = self.analyzer.get_environment_group_stats(landcover=999, slope=10.0)
        self.assertIsNone(stats)

if __name__ == '__main__':
    unittest.main() 