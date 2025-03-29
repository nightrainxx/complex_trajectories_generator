"""
批量轨迹生成器单元测试
"""

import unittest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.terrain import TerrainLoader
from src.core.batch_generator import BatchGenerator
from src.utils.config import config

class TestBatchGenerator(unittest.TestCase):
    """批量轨迹生成器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟的地形加载器
        self.terrain_loader = MagicMock(spec=TerrainLoader)
        
        # 创建临时输出目录
        self.test_output_dir = Path('test_output')
        self.test_output_dir.mkdir(exist_ok=True)
        
        # 创建批量生成器
        self.generator = BatchGenerator(
            terrain_loader=self.terrain_loader,
            output_dir=self.test_output_dir,
            num_workers=2
        )
        
        # 模拟配置参数
        config.generation.NUM_TRAJECTORIES_TO_GENERATE = 4
        config.generation.NUM_END_POINTS = 2
        
        # 模拟起终点选择器返回的点对
        self.test_pairs = [
            ((0.0, 0.0), (100.0, 100.0)),
            ((0.0, 100.0), (100.0, 0.0)),
            ((50.0, 0.0), (50.0, 100.0)),
            ((0.0, 50.0), (100.0, 50.0))
        ]
        self.generator.point_selector.select_points = MagicMock(
            return_value=self.test_pairs
        )
        
        # 模拟轨迹生成器返回的轨迹数据
        self.test_trajectory = {
            'timestamp': [0.0, 1.0, 2.0],
            'x': [0.0, 50.0, 100.0],
            'y': [0.0, 50.0, 100.0],
            'speed': [10.0, 10.0, 10.0],
            'orientation': [45.0, 45.0, 45.0]
        }
        self.generator.trajectory_generator.generate_trajectory = MagicMock(
            return_value=self.test_trajectory
        )
        
    def tearDown(self):
        """测试后清理"""
        # 删除测试输出目录
        import shutil
        shutil.rmtree(self.test_output_dir)
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.generator.num_workers, 2)
        self.assertEqual(self.generator.output_dir, self.test_output_dir)
        self.assertTrue(self.test_output_dir.exists())
        
    def test_generate_batch(self):
        """测试批量生成"""
        # 生成轨迹
        trajectory_files = self.generator.generate_batch()
        
        # 验证生成的文件数量
        self.assertEqual(len(trajectory_files), 4)
        
        # 验证文件内容
        for file_path in trajectory_files:
            self.assertTrue(file_path.exists())
            
            with open(file_path) as f:
                trajectory = json.load(f)
                
            # 验证轨迹数据
            self.assertIn('timestamp', trajectory)
            self.assertIn('x', trajectory)
            self.assertIn('y', trajectory)
            self.assertIn('speed', trajectory)
            self.assertIn('orientation', trajectory)
            
            # 验证元数据
            self.assertIn('metadata', trajectory)
            metadata = trajectory['metadata']
            self.assertIn('start_point', metadata)
            self.assertIn('end_point', metadata)
            self.assertIn('index', metadata)
            self.assertIn('generation_time', metadata)
            
    def test_generate_single_trajectory(self):
        """测试单条轨迹生成"""
        start_point = (0.0, 0.0)
        end_point = (100.0, 100.0)
        idx = 0
        
        # 生成轨迹
        trajectory_file = self.generator._generate_single_trajectory(
            start_point,
            end_point,
            idx
        )
        
        # 验证文件是否存在
        self.assertTrue(trajectory_file.exists())
        
        # 验证文件内容
        with open(trajectory_file) as f:
            trajectory = json.load(f)
            
        # 验证轨迹数据
        self.assertEqual(trajectory['x'], self.test_trajectory['x'])
        self.assertEqual(trajectory['y'], self.test_trajectory['y'])
        self.assertEqual(trajectory['speed'], self.test_trajectory['speed'])
        self.assertEqual(
            trajectory['orientation'],
            self.test_trajectory['orientation']
        )
        
        # 验证元数据
        metadata = trajectory['metadata']
        self.assertEqual(metadata['start_point'], list(start_point))
        self.assertEqual(metadata['end_point'], list(end_point))
        self.assertEqual(metadata['index'], idx)
        self.assertIsNotNone(metadata['generation_time'])
        
    def test_save_trajectory(self):
        """测试轨迹保存"""
        trajectory = self.test_trajectory.copy()
        file_path = self.test_output_dir / "test_trajectory.json"
        
        # 保存轨迹
        self.generator._save_trajectory(trajectory, file_path)
        
        # 验证文件是否存在
        self.assertTrue(file_path.exists())
        
        # 验证文件内容
        with open(file_path) as f:
            saved_trajectory = json.load(f)
            
        # 验证数据完整性
        for key in self.test_trajectory:
            self.assertEqual(
                saved_trajectory[key],
                self.test_trajectory[key]
            )
            
    def test_error_handling(self):
        """测试错误处理"""
        # 模拟轨迹生成失败
        self.generator.trajectory_generator.generate_trajectory.side_effect = \
            Exception("测试错误")
            
        # 生成轨迹
        trajectory_files = self.generator.generate_batch()
        
        # 验证返回空列表
        self.assertEqual(len(trajectory_files), 0)
        
if __name__ == '__main__':
    unittest.main() 