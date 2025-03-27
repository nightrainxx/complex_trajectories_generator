"""运动模拟器的单元测试"""

import unittest
import numpy as np
import sys
import os
import pytest

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.motion_simulator import (
    MotionSimulator,
    MotionConstraints,
    EnvironmentParams,
    TerrainConstraints
)

class TestMotionSimulator(unittest.TestCase):
    """测试运动模拟器类"""

    def setUp(self):
        """测试前的设置"""
        # 创建一个具有默认约束的模拟器实例
        self.simulator = MotionSimulator()

        # 创建一个简单的测试路径
        self.test_path = [
            (0.0, 0.0),    # 起点
            (1.0, 1.0),    # 对角线上的点
            (2.0, 2.0)     # 终点
        ]

        # 创建一个返回固定环境参数的函数
        def mock_env_params(lon, lat):
            return EnvironmentParams(
                max_speed=10.0,
                typical_speed=5.0,
                speed_stddev=1.0,
                slope_magnitude=0.0,  # 平地
                slope_aspect=0.0
            )
        self.mock_env_params_func = mock_env_params

    def test_simulate_motion(self):
        """测试基本的运动模拟功能"""
        trajectory = self.simulator.simulate_motion(
            self.test_path,
            self.mock_env_params_func
        )

        # 验证轨迹的基本属性
        self.assertGreater(len(trajectory), 0)
        
        # 检查每个轨迹点的合理性
        prev_time = -1
        for point in trajectory:
            time, _, _, speed, heading = point
            # 时间应该递增
            self.assertGreater(time, prev_time)
            prev_time = time
            # 速度应在限制范围内
            self.assertGreaterEqual(speed, self.simulator.motion_constraints.min_speed)
            self.assertLessEqual(speed, 10.0)  # max_speed from mock_env_params
            # 朝向应在0-360度范围内
            self.assertGreaterEqual(heading, 0)
            self.assertLess(heading, 360)

        # 检查加速度约束
        for i in range(1, len(trajectory)):
            dt = trajectory[i][0] - trajectory[i-1][0]
            if dt > 0:  # 避免除以零
                acceleration = (trajectory[i][3] - trajectory[i-1][3]) / dt
                # 加速度应在限制范围内
                self.assertLessEqual(abs(acceleration), 
                                max(self.simulator.motion_constraints.max_acceleration,
                                    self.simulator.motion_constraints.max_deceleration))

    def test_speed_constraints(self):
        """测试速度约束"""
        trajectory = self.simulator.simulate_motion(
            self.test_path,
            self.mock_env_params_func
        )

        # 验证所有速度都在约束范围内
        for point in trajectory:
            _, _, _, speed, _ = point
            self.assertGreaterEqual(speed, self.simulator.motion_constraints.min_speed)
            self.assertLessEqual(speed, 10.0)  # max_speed from mock_env_params

    def test_heading_constraints(self):
        """测试转向约束"""
        trajectory = self.simulator.simulate_motion(
            self.test_path,
            self.mock_env_params_func
        )

        # 检查转向率约束
        for i in range(1, len(trajectory)):
            dt = trajectory[i][0] - trajectory[i-1][0]
            if dt > 0:  # 避免除以零
                # 计算朝向变化（考虑角度环绕）
                heading_change = trajectory[i][4] - trajectory[i-1][4]
                if heading_change > 180:
                    heading_change -= 360
                elif heading_change < -180:
                    heading_change += 360
                    
                turn_rate = abs(heading_change / dt)
                # 转向率应在限制范围内
                self.assertLessEqual(turn_rate, self.simulator.motion_constraints.max_turn_rate * 1.1)  # 允许10%的误差

    def test_invalid_path(self):
        """测试无效路径的处理"""
        # 测试空路径
        with self.assertRaises(ValueError):
            self.simulator.simulate_motion([], self.mock_env_params_func)
            
        # 测试单点路径
        with self.assertRaises(ValueError):
            self.simulator.simulate_motion([(0.0, 0.0)], self.mock_env_params_func)

    def test_calculate_target_speed(self):
        """测试目标速度计算"""
        env_params = EnvironmentParams(
            max_speed=10.0,
            typical_speed=5.0,
            speed_stddev=1.0
        )

        # 多次测试以验证随机性和限制
        for _ in range(100):
            speed = self.simulator._calculate_target_speed(env_params, 0.0)  # 添加朝向参数
            self.assertGreaterEqual(speed, self.simulator.motion_constraints.min_speed)
            self.assertLessEqual(speed, env_params.max_speed)

    def test_heading_calculation(self):
        """测试朝向角计算"""
        # 测试基本方向
        test_cases = [
            # (start_pos, end_pos, expected_heading)
            ((0, 0), (0, 1), 0),    # 正北
            ((0, 0), (1, 0), 90),   # 正东
            ((0, 0), (0, -1), 180), # 正南
            ((0, 0), (-1, 0), 270)  # 正西
        ]
        
        for start_pos, end_pos, expected_heading in test_cases:
            heading = self.simulator._calculate_heading(start_pos, end_pos)
            self.assertAlmostEqual(heading, expected_heading, places=1)

def test_slope_effects_calculation():
    """测试坡度对速度的影响计算"""
    # 初始化模拟器，使用自定义地形约束
    terrain_constraints = TerrainConstraints(
        max_uphill_slope=30.0,
        max_downhill_slope=35.0,
        max_cross_slope=25.0,
        k_uphill=0.1,
        k_downhill=0.05,
        k_cross=0.2,
        min_speed_steep_slope=0.5
    )
    simulator = MotionSimulator(terrain_constraints=terrain_constraints)
    
    # 测试用例1：平地（无坡度）
    env_params = EnvironmentParams(
        max_speed=10.0,
        typical_speed=5.0,
        speed_stddev=0.5,
        slope_magnitude=0.0,
        slope_aspect=0.0
    )
    slope_along, cross_slope, speed_factor = simulator._calculate_slope_effects(0.0, env_params)
    assert slope_along == 0.0
    assert cross_slope == 0.0
    assert speed_factor == 1.0

    # 测试用例2：正向上坡
    env_params = EnvironmentParams(
        max_speed=10.0,
        typical_speed=5.0,
        speed_stddev=0.5,
        slope_magnitude=20.0,  # 20度坡
        slope_aspect=0.0       # 正北方向
    )
    # 车辆朝向正北（与坡向一致）
    slope_along, cross_slope, speed_factor = simulator._calculate_slope_effects(0.0, env_params)
    assert slope_along == pytest.approx(20.0, abs=0.1)
    assert cross_slope == pytest.approx(0.0, abs=0.1)
    assert speed_factor < 1.0  # 上坡应该减速
    assert speed_factor > 0.0  # 但仍可通行

    # 测试用例3：横坡
    # 车辆朝向东方（与坡向垂直）
    env_params = EnvironmentParams(
        max_speed=10.0,
        typical_speed=5.0,
        speed_stddev=0.5,
        slope_magnitude=20.0,
        slope_aspect=0.0       # 正北方向的坡
    )
    slope_along, cross_slope, speed_factor = simulator._calculate_slope_effects(90.0, env_params)
    assert slope_along == pytest.approx(0.0, abs=0.1)
    assert cross_slope == pytest.approx(20.0, abs=0.1)
    assert speed_factor < 1.0  # 横坡应该减速
    assert speed_factor > 0.0  # 但仍可通行

    # 测试用例4：超过最大上坡限制
    env_params = EnvironmentParams(
        max_speed=10.0,
        typical_speed=5.0,
        speed_stddev=0.5,
        slope_magnitude=35.0,  # 超过最大上坡限制
        slope_aspect=0.0
    )
    slope_along, cross_slope, speed_factor = simulator._calculate_slope_effects(0.0, env_params)
    assert speed_factor == 0.0  # 无法通行

    # 测试用例5：下坡
    env_params = EnvironmentParams(
        max_speed=10.0,
        typical_speed=5.0,
        speed_stddev=0.5,
        slope_magnitude=15.0,
        slope_aspect=0.0
    )
    # 车辆朝向正南（与坡向相反）
    slope_along, cross_slope, speed_factor = simulator._calculate_slope_effects(180.0, env_params)
    assert slope_along == pytest.approx(-15.0, abs=0.1)
    assert cross_slope == pytest.approx(0.0, abs=0.1)
    assert speed_factor > 1.0  # 下坡应该轻微加速
    assert speed_factor <= 1.2  # 但不超过1.2倍速度

def test_target_speed_with_slope():
    """测试目标速度计算（包含坡度影响）"""
    simulator = MotionSimulator()
    
    # 测试平地情况
    env_params = EnvironmentParams(
        max_speed=10.0,
        typical_speed=5.0,
        speed_stddev=0.5,
        slope_magnitude=0.0,
        slope_aspect=0.0
    )
    
    # 多次测试以验证随机性
    for _ in range(10):
        speed = simulator._calculate_target_speed(env_params, 0.0)
        assert 3.0 <= speed <= 7.0  # 考虑随机扰动的范围
        
    # 测试上坡情况
    env_params = EnvironmentParams(
        max_speed=10.0,
        typical_speed=5.0,
        speed_stddev=0.5,
        slope_magnitude=20.0,
        slope_aspect=0.0
    )
    
    # 车辆朝向正北（上坡）
    speed_uphill = simulator._calculate_target_speed(env_params, 0.0)
    assert speed_uphill < 5.0  # 上坡速度应该小于典型速度
    
    # 测试下坡情况
    # 车辆朝向正南（下坡）
    speed_downhill = simulator._calculate_target_speed(env_params, 180.0)
    assert speed_downhill > speed_uphill  # 下坡速度应该大于上坡速度

def test_motion_simulation_with_slopes():
    """测试在有坡度的环境下的运动模拟"""
    simulator = MotionSimulator()
    
    # 创建一个简单的路径
    path = [(0.0, 0.0), (0.0, 10.0)]  # 向北的直线路径
    
    # 模拟上坡环境
    def env_params_uphill(lon, lat):
        return EnvironmentParams(
            max_speed=10.0,
            typical_speed=5.0,
            speed_stddev=0.5,
            slope_magnitude=20.0,  # 20度坡
            slope_aspect=0.0       # 正北方向的坡
        )
    
    # 模拟运动
    trajectory_uphill = simulator.simulate_motion(path, env_params_uphill)
    
    # 检查上坡轨迹的特征
    speeds_uphill = [point[3] for point in trajectory_uphill]
    assert max(speeds_uphill) < 5.0  # 上坡最大速度应该小于典型速度
    
    # 模拟下坡环境
    def env_params_downhill(lon, lat):
        return EnvironmentParams(
            max_speed=10.0,
            typical_speed=5.0,
            speed_stddev=0.5,
            slope_magnitude=20.0,
            slope_aspect=180.0     # 正南方向的坡
        )
    
    # 模拟运动
    trajectory_downhill = simulator.simulate_motion(path, env_params_downhill)
    
    # 检查下坡轨迹的特征
    speeds_downhill = [point[3] for point in trajectory_downhill]
    assert max(speeds_downhill) > max(speeds_uphill)  # 下坡最大速度应该大于上坡最大速度

if __name__ == '__main__':
    unittest.main(verbosity=2) 