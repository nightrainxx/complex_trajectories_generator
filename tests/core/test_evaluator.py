"""
轨迹评估器单元测试
"""

import unittest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.core.evaluator import Evaluator

class TestEvaluator(unittest.TestCase):
    """评估器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.test_output_dir = Path('test_output')
        self.test_output_dir.mkdir(exist_ok=True)
        
        # 创建评估器
        self.evaluator = Evaluator(output_dir=self.test_output_dir)
        
        # 创建测试数据
        self._create_test_data()
        
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        import shutil
        shutil.rmtree(self.test_output_dir)
        
    def _create_test_data(self):
        """创建测试数据"""
        # 创建OORD测试数据
        self.oord_data = pd.DataFrame({
            'timestamp': np.arange(0, 10, 0.1),
            'x': np.linspace(0, 100, 100),
            'y': np.linspace(0, 100, 100),
            'speed': np.random.normal(10, 2, 100),
            'orientation': np.linspace(0, 360, 100),
            'trajectory_id': np.repeat([1, 2], 50),
            'group_label': np.repeat(['A', 'B'], 50)
        })
        
        # 保存OORD数据
        self.oord_file = self.test_output_dir / 'test_oord.csv'
        self.oord_data.to_csv(self.oord_file, index=False)
        
        # 创建合成轨迹测试数据
        self.synthetic_dir = self.test_output_dir / 'synthetic'
        self.synthetic_dir.mkdir(exist_ok=True)
        
        for i in range(2):
            trajectory = {
                'timestamp': list(np.arange(0, 10, 0.1)),
                'x': list(np.linspace(0, 100, 100)),
                'y': list(np.linspace(0, 100, 100)),
                'speed': list(np.random.normal(10, 2, 100)),
                'orientation': list(np.linspace(0, 360, 100)),
                'metadata': {
                    'index': i,
                    'start_point': [0, 0],
                    'end_point': [100, 100]
                }
            }
            
            with open(self.synthetic_dir / f'trajectory_{i+1}.json', 'w') as f:
                json.dump(trajectory, f)
                
    def test_load_data(self):
        """测试数据加载"""
        self.evaluator.load_data(self.oord_file, self.synthetic_dir)
        
        # 验证OORD数据
        self.assertIsNotNone(self.evaluator.oord_data)
        self.assertEqual(len(self.evaluator.oord_data), 100)
        
        # 验证合成数据
        self.assertIsNotNone(self.evaluator.synthetic_data)
        self.assertEqual(len(self.evaluator.synthetic_data), 200)
        
    def test_evaluate(self):
        """测试评估"""
        self.evaluator.load_data(self.oord_file, self.synthetic_dir)
        metrics = self.evaluator.evaluate()
        
        # 验证指标
        self.assertIn('speed_ks_stat', metrics)
        self.assertIn('speed_ks_p_value', metrics)
        self.assertIn('acceleration_ks_stat', metrics)
        self.assertIn('acceleration_ks_p_value', metrics)
        self.assertIn('turn_rate_ks_stat', metrics)
        self.assertIn('turn_rate_ks_p_value', metrics)
        self.assertIn('mean_group_speed_diff', metrics)
        
        # 验证输出文件
        self.assertTrue(
            (self.test_output_dir / 'speed_distribution.png').exists()
        )
        self.assertTrue(
            (self.test_output_dir / 'acceleration_distribution.png').exists()
        )
        self.assertTrue(
            (self.test_output_dir / 'turn_rate_distribution.png').exists()
        )
        self.assertTrue(
            (self.test_output_dir / 'environment_interaction.png').exists()
        )
        self.assertTrue(
            (self.test_output_dir / 'evaluation_report.txt').exists()
        )
        
    def test_compare_speed_distributions(self):
        """测试速度分布比较"""
        self.evaluator.load_data(self.oord_file, self.synthetic_dir)
        metrics = self.evaluator._compare_speed_distributions()
        
        self.assertIn('speed_ks_stat', metrics)
        self.assertIn('speed_ks_p_value', metrics)
        self.assertIn('speed_mean_diff', metrics)
        self.assertIn('speed_std_diff', metrics)
        
    def test_compare_acceleration_distributions(self):
        """测试加速度分布比较"""
        self.evaluator.load_data(self.oord_file, self.synthetic_dir)
        metrics = self.evaluator._compare_acceleration_distributions()
        
        self.assertIn('acceleration_ks_stat', metrics)
        self.assertIn('acceleration_ks_p_value', metrics)
        self.assertIn('acceleration_mean_diff', metrics)
        self.assertIn('acceleration_std_diff', metrics)
        
    def test_compare_turn_rate_distributions(self):
        """测试转向率分布比较"""
        self.evaluator.load_data(self.oord_file, self.synthetic_dir)
        metrics = self.evaluator._compare_turn_rate_distributions()
        
        self.assertIn('turn_rate_ks_stat', metrics)
        self.assertIn('turn_rate_ks_p_value', metrics)
        self.assertIn('turn_rate_mean_diff', metrics)
        self.assertIn('turn_rate_std_diff', metrics)
        
    def test_compare_environment_interaction(self):
        """测试环境交互比较"""
        self.evaluator.load_data(self.oord_file, self.synthetic_dir)
        metrics = self.evaluator._compare_environment_interaction()
        
        self.assertIn('mean_group_speed_diff', metrics)
        self.assertIn('speed_diff_group_A', metrics)
        self.assertIn('speed_diff_group_B', metrics)
        
    def test_error_handling(self):
        """测试错误处理"""
        # 测试未加载数据时评估
        with self.assertRaises(ValueError):
            self.evaluator.evaluate()
            
        # 测试加载不存在的文件
        with self.assertRaises(FileNotFoundError):
            self.evaluator.load_data(
                Path('not_exist.csv'),
                self.synthetic_dir
            )
            
        # 测试加载空目录
        empty_dir = self.test_output_dir / 'empty'
        empty_dir.mkdir()
        with self.assertRaises(ValueError):
            self.evaluator.load_data(self.oord_file, empty_dir)
            
if __name__ == '__main__':
    unittest.main() 