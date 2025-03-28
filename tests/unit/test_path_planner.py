"""路径规划器单元测试"""

import pytest
import numpy as np
import rasterio
import tempfile
import os
from pathlib import Path
from src.generator.path_planner import PathPlanner

@pytest.fixture
def test_data():
    """生成测试用的环境数据"""
    size = (100, 100)  # 使用较大的尺寸以便测试路径规划
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 生成测试数据
        cost_map = np.ones(size, dtype=np.float32)  # 默认成本为1
        
        # 添加障碍物（成本无穷大）
        cost_map[0:10, 0:10] = float('inf')  # 左上角障碍物
        cost_map[90:100, 90:100] = float('inf')  # 右下角障碍物
        
        # 添加高成本区域
        cost_map[40:60, 40:60] = 5.0  # 中心区域高成本
        
        # 创建测试文件
        meta = {
            'driver': 'GTiff',
            'height': size[0],
            'width': size[1],
            'count': 1,
            'crs': '+proj=latlong',
            'transform': rasterio.transform.Affine(0.001, 0, 0, 0, 0.001, 0),  # 约111米/像素
            'dtype': np.float32
        }
        
        # 保存成本地图
        cost_map_path = os.path.join(temp_dir, 'cost_map.tif')
        with rasterio.open(cost_map_path, 'w', **meta) as dst:
            dst.write(cost_map, 1)
        
        yield {
            'cost_map_path': cost_map_path,
            'cost_map_data': cost_map,
            'transform': meta['transform']
        }

class TestPathPlanner:
    """路径规划器测试类"""
    
    def test_init(self, test_data):
        """测试初始化"""
        planner = PathPlanner(test_data['cost_map_path'])
        assert planner.height == 100
        assert planner.width == 100
        assert np.array_equal(planner.cost_map, test_data['cost_map_data'])
        assert planner.transform == test_data['transform']
    
    def test_is_valid_position(self, test_data):
        """测试位置有效性检查"""
        planner = PathPlanner(test_data['cost_map_path'])
        
        # 测试正常点
        assert planner.is_valid_position((50, 50)) == True
        
        # 测试障碍物
        assert planner.is_valid_position((5, 5)) == False
        
        # 测试边界外的点
        assert planner.is_valid_position((-1, 0)) == False
        assert planner.is_valid_position((100, 0)) == False
    
    def test_calculate_heuristic(self, test_data):
        """测试启发式函数"""
        planner = PathPlanner(test_data['cost_map_path'])
        
        # 测试水平距离
        assert planner.calculate_heuristic((0, 0), (0, 3)) == 3.0
        
        # 测试垂直距离
        assert planner.calculate_heuristic((0, 0), (4, 0)) == 4.0
        
        # 测试对角线距离
        assert planner.calculate_heuristic((0, 0), (3, 4)) == 5.0
    
    def test_calculate_turn_cost(self, test_data):
        """测试转弯代价计算"""
        planner = PathPlanner(test_data['cost_map_path'])
        
        # 测试直线运动（无转弯）
        cost = planner.calculate_turn_cost(
            (1, 1),
            (1, 2),
            (1, 0)
        )
        assert cost == 0.0
        
        # 测试90度转弯
        cost = planner.calculate_turn_cost(
            (1, 1),
            (2, 1),
            (1, 0)
        )
        assert cost == pytest.approx(planner.smoothness_weight * np.pi/2, rel=1e-6)
        
        # 测试180度转弯
        cost = planner.calculate_turn_cost(
            (1, 1),
            (1, 0),
            (1, 2)
        )
        assert cost == pytest.approx(planner.smoothness_weight * np.pi, rel=1e-6)
    
    def test_find_path(self, test_data):
        """测试路径搜索"""
        planner = PathPlanner(test_data['cost_map_path'])
        
        # 测试简单路径（无障碍）
        start = (20, 20)
        goal = (25, 25)
        path = planner.find_path(start, goal)
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal
        
        # 测试绕过障碍物的路径
        start = (5, 15)  # 靠近左上角障碍物
        goal = (15, 5)
        path = planner.find_path(start, goal)
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal
        # 验证路径上的点都是可通行的
        for point in path:
            assert planner.is_valid_position(point)
        
        # 测试不可能的路径（起点或终点在障碍物中）
        with pytest.raises(ValueError):
            planner.find_path((5, 5), (20, 20))  # 起点在障碍物中
        with pytest.raises(ValueError):
            planner.find_path((20, 20), (95, 95))  # 终点在障碍物中
    
    def test_smooth_path(self, test_data):
        """测试路径平滑"""
        planner = PathPlanner(test_data['cost_map_path'])
        
        # 创建一个锯齿状路径
        original_path = [
            (20, 20), (21, 20), (22, 20), (23, 21),
            (24, 22), (25, 23), (26, 24), (27, 25)
        ]
        
        # 平滑路径
        smooth_path = planner.smooth_path(original_path)
        
        # 验证平滑后的路径
        assert len(smooth_path) > 0
        assert smooth_path[0] == original_path[0]  # 起点应该保持不变
        assert smooth_path[-1] == original_path[-1]  # 终点应该保持不变
        
        # 验证平滑后的路径点都是可通行的
        for point in smooth_path:
            assert planner.is_valid_position(point)
        
        # 验证平滑效果（通过计算路径的总转弯代价）
        original_turn_cost = sum(
            planner.calculate_turn_cost(original_path[i], original_path[i+1], original_path[i-1])
            for i in range(1, len(original_path)-1)
        )
        smooth_turn_cost = sum(
            planner.calculate_turn_cost(smooth_path[i], smooth_path[i+1], smooth_path[i-1])
            for i in range(1, len(smooth_path)-1)
        )
        assert smooth_turn_cost < original_turn_cost
    
    def test_plan(self, test_data):
        """测试完整的路径规划过程"""
        planner = PathPlanner(test_data['cost_map_path'])
        
        # 测试正常规划（包含平滑）
        start = (20, 20)
        goal = (80, 80)
        path = planner.plan(start, goal, smooth=True)
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal
        
        # 测试不进行平滑的规划
        path_no_smooth = planner.plan(start, goal, smooth=False)
        assert len(path_no_smooth) > 0
        assert path_no_smooth[0] == start
        assert path_no_smooth[-1] == goal
        
        # 验证平滑路径的转弯代价更小
        if len(path) > 2 and len(path_no_smooth) > 2:
            smooth_cost = sum(
                planner.calculate_turn_cost(path[i], path[i+1], path[i-1])
                for i in range(1, len(path)-1)
            )
            no_smooth_cost = sum(
                planner.calculate_turn_cost(path_no_smooth[i], path_no_smooth[i+1], path_no_smooth[i-1])
                for i in range(1, len(path_no_smooth)-1)
            )
            assert smooth_cost < no_smooth_cost 