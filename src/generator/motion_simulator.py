"""运动模拟器模块

负责模拟目标在给定路径上的运动，考虑环境因素（坡度、土地覆盖）对速度和方向的影响。

输入参数:
- 路径点列表
- 环境参数（最大速度、典型速度、速度标准差）
- 运动约束（最大加速度、最大减速度、最大转向率）

输出:
- 轨迹点列表，包含时间戳、位置、速度和朝向
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class MotionConstraints:
    """运动约束参数"""
    max_acceleration: float = 2.0  # 最大加速度 (m/s^2)
    max_deceleration: float = 4.0  # 最大减速度 (m/s^2)
    max_turn_rate: float = 45.0    # 最大转向率 (度/秒)
    min_speed: float = 0.0         # 最小速度 (m/s)
    time_step: float = 1.0         # 模拟时间步长 (秒)
    max_iterations: int = 10000    # 最大迭代次数
    position_threshold: float = 0.1  # 位置判断阈值 (m)

@dataclass
class EnvironmentParams:
    """环境参数
    
    Attributes:
        max_speed: 最大允许速度 (m/s)
        typical_speed: 典型速度 (m/s)
        speed_stddev: 速度标准差 (m/s)
        slope_magnitude: 坡度大小 (度)
        slope_aspect: 坡向角度 (度，北为0，顺时针)
        landcover_code: 土地覆盖类型编码
    """
    max_speed: float       # 最大允许速度 (m/s)
    typical_speed: float   # 典型速度 (m/s)
    speed_stddev: float   # 速度标准差 (m/s)
    slope_magnitude: float = 0.0  # 坡度大小 (度)
    slope_aspect: float = 0.0     # 坡向角度 (度，北为0，顺时针)
    landcover_code: int = 0       # 土地覆盖类型编码

@dataclass
class TerrainConstraints:
    """地形约束参数"""
    max_uphill_slope: float = 30.0    # 最大上坡坡度 (度)
    max_downhill_slope: float = 35.0  # 最大下坡坡度 (度)
    max_cross_slope: float = 25.0     # 最大横坡坡度 (度)
    k_uphill: float = 0.1             # 上坡减速系数
    k_downhill: float = 0.05          # 下坡加速系数
    k_cross: float = 0.2              # 横坡减速系数
    min_speed_steep_slope: float = 0.5 # 陡坡最小速度 (m/s)

class MotionSimulator:
    """运动模拟器类"""

    def __init__(self, motion_constraints: Optional[MotionConstraints] = None,
                 terrain_constraints: Optional[TerrainConstraints] = None):
        """初始化运动模拟器

        Args:
            motion_constraints: 运动约束参数，如果为None则使用默认值
            terrain_constraints: 地形约束参数，如果为None则使用默认值
        """
        self.motion_constraints = motion_constraints or MotionConstraints()
        self.terrain_constraints = terrain_constraints or TerrainConstraints()
        
        # 用于速度随机性的高斯噪声参数
        self.speed_noise_scale = 0.1  # 速度噪声比例
        self.heading_noise_scale = 2.0  # 朝向噪声（度）

    def _calculate_slope_effects(self, current_heading: float,
                              env_params: EnvironmentParams) -> Tuple[float, float, float]:
        """计算坡度对速度的影响

        Args:
            current_heading: 当前朝向 (度)
            env_params: 环境参数

        Returns:
            Tuple[float, float, float]: (纵向坡度, 横向坡度, 速度调整系数)
        """
        # 如果没有坡度信息，返回默认值
        if env_params.slope_magnitude == 0:
            return 0.0, 0.0, 1.0

        # 计算行驶方向与坡向的夹角
        delta_angle = current_heading - env_params.slope_aspect
        delta_angle_rad = np.radians(delta_angle)

        # 计算纵向坡度（上下坡）和横向坡度
        slope_along_path = env_params.slope_magnitude * np.cos(delta_angle_rad)
        cross_slope = env_params.slope_magnitude * abs(np.sin(delta_angle_rad))

        # 计算速度调整系数
        speed_factor = 1.0

        # 上坡减速
        if slope_along_path > 0:
            if slope_along_path > self.terrain_constraints.max_uphill_slope:
                speed_factor = 0.0  # 超过最大上坡坡度，无法通行
            else:
                reduction = max(0.1, 1 - self.terrain_constraints.k_uphill * slope_along_path)
                speed_factor *= reduction

        # 下坡加速/减速
        else:
            slope_down = -slope_along_path
            if slope_down > self.terrain_constraints.max_downhill_slope:
                speed_factor = 0.0  # 超过最大下坡坡度，无法通行
            else:
                # 下坡时轻微加速，但需要考虑制动限制
                factor = 1 + self.terrain_constraints.k_downhill * slope_down
                speed_factor *= min(factor, 1.2)  # 最多增加20%速度

        # 横坡减速
        if cross_slope > 0:
            if cross_slope > self.terrain_constraints.max_cross_slope:
                speed_factor = 0.0  # 超过最大横坡坡度，无法通行
            else:
                # 使用二次函数使横坡影响更敏感
                reduction = max(0.1, 1 - self.terrain_constraints.k_cross * (cross_slope ** 2))
                speed_factor *= reduction

        # 确保速度不会过低（除非完全无法通行）
        if speed_factor > 0:
            speed_factor = max(speed_factor, 
                             self.terrain_constraints.min_speed_steep_slope / env_params.typical_speed)

        return slope_along_path, cross_slope, speed_factor

    def _calculate_target_speed(self, env_params: EnvironmentParams,
                              current_heading: float) -> float:
        """计算目标速度，考虑环境限制、坡度影响和随机性

        Args:
            env_params: 环境参数
            current_heading: 当前朝向 (度)

        Returns:
            目标速度 (m/s)
        """
        # 基础目标速度是典型速度
        base_speed = env_params.typical_speed
        
        # 计算坡度影响
        _, _, speed_factor = self._calculate_slope_effects(current_heading, env_params)
        
        # 应用坡度影响
        base_speed *= speed_factor
        
        # 添加随机扰动（高斯噪声）
        noise = np.random.normal(0, env_params.speed_stddev * self.speed_noise_scale)
        target_speed = base_speed + noise
        
        # 限制在合理范围内
        target_speed = np.clip(
            target_speed,
            self.motion_constraints.min_speed,
            min(env_params.max_speed, env_params.typical_speed * 1.5)
        )
        
        return target_speed

    def simulate_motion(self, path: List[Tuple[float, float]],
                       env_params_func) -> List[Tuple[float, float, float, float, float]]:
        """模拟目标在给定路径上的运动

        Args:
            path: 路径点列表 [(lon, lat), ...]
            env_params_func: 函数，输入(lon, lat)返回对应位置的EnvironmentParams

        Returns:
            轨迹点列表 [(timestamp, lon, lat, speed, heading), ...]
        """
        if len(path) < 2:
            raise ValueError("路径至少需要包含两个点")

        # 初始化轨迹
        trajectory = []
        current_time = 0.0
        current_pos = path[0]
        current_speed = 0.0
        current_heading = self._calculate_initial_heading(path[0], path[1])
        
        # 添加初始点
        trajectory.append((
            current_time,
            current_pos[0],
            current_pos[1],
            current_speed,
            current_heading
        ))

        # 遍历路径点
        path_index = 1
        iteration_count = 0
        
        while path_index < len(path) and iteration_count < self.motion_constraints.max_iterations:
            # 获取目标点
            target_pos = path[path_index]
            
            # 获取当前位置的环境参数
            env_params = env_params_func(current_pos[0], current_pos[1])
            
            # 计算到目标点的距离和方向
            distance = self._calculate_distance(current_pos, target_pos)
            target_heading = self._calculate_heading(current_pos, target_pos)
            
            # 如果已经接近目标点，移动到下一个目标点
            if distance < self.motion_constraints.position_threshold:
                current_pos = target_pos  # 直接移动到目标点
                path_index += 1
                
                # 更新时间和轨迹点
                current_time += self.motion_constraints.time_step
                trajectory.append((
                    current_time,
                    current_pos[0],
                    current_pos[1],
                    current_speed,
                    current_heading
                ))
                
                if path_index < len(path):
                    target_pos = path[path_index]
                    target_heading = self._calculate_heading(current_pos, target_pos)
                    distance = self._calculate_distance(current_pos, target_pos)
                else:
                    # 到达终点，添加最后一个点并退出
                    current_time += self.motion_constraints.time_step
                    trajectory.append((
                        current_time,
                        current_pos[0],
                        current_pos[1],
                        0.0,  # 终点速度为0
                        current_heading
                    ))
                    break
                continue
            
            # 计算目标速度（考虑环境限制、坡度影响和随机性）
            target_speed = self._calculate_target_speed(env_params, current_heading)
            
            # 调整速度和朝向
            current_speed = self._adjust_speed(
                current_speed, target_speed, self.motion_constraints.time_step)
            current_heading = self._adjust_heading(
                current_heading, target_heading, self.motion_constraints.time_step)
            
            # 计算新位置（使用弧度进行计算）
            heading_rad = np.radians(current_heading)
            movement = current_speed * self.motion_constraints.time_step
            
            # 根据当前位置到目标点的距离来调整移动量
            scale = min(1.0, distance / movement) if movement > 0 else 1.0
            movement *= scale
            
            # 计算位置增量
            dx = movement * np.sin(heading_rad)  # 东西方向
            dy = movement * np.cos(heading_rad)  # 南北方向
            
            # 更新位置
            new_pos = (
                current_pos[0] + dx,
                current_pos[1] + dy
            )
            
            # 更新时间和位置
            current_time += self.motion_constraints.time_step
            current_pos = new_pos
            
            # 添加轨迹点
            trajectory.append((
                current_time,
                current_pos[0],
                current_pos[1],
                current_speed,
                current_heading
            ))
            
            iteration_count += 1

        if iteration_count >= self.motion_constraints.max_iterations:
            logger.warning(f"达到最大迭代次数 {self.motion_constraints.max_iterations}")

        return trajectory

    def _adjust_speed(self, current_speed: float,
                     target_speed: float,
                     dt: float) -> float:
        """调整速度，考虑加速度限制

        Args:
            current_speed: 当前速度 (m/s)
            target_speed: 目标速度 (m/s)
            dt: 时间步长 (秒)

        Returns:
            新的速度 (m/s)
        """
        speed_diff = target_speed - current_speed
        
        # 根据加速或减速选择合适的限制
        max_change = (self.motion_constraints.max_acceleration if speed_diff > 0
                     else self.motion_constraints.max_deceleration) * dt
        
        # 限制速度变化
        actual_change = np.clip(speed_diff, -max_change, max_change)
        new_speed = current_speed + actual_change
        
        return max(new_speed, self.motion_constraints.min_speed)

    def _adjust_heading(self, current_heading: float,
                       target_heading: float,
                       dt: float) -> float:
        """调整朝向，考虑转向率限制

        Args:
            current_heading: 当前朝向 (度)
            target_heading: 目标朝向 (度)
            dt: 时间步长 (秒)

        Returns:
            新的朝向 (度)
        """
        # 计算需要转向的角度（处理角度环绕）
        heading_diff = target_heading - current_heading
        if heading_diff > 180:
            heading_diff -= 360
        elif heading_diff < -180:
            heading_diff += 360
            
        # 添加随机扰动
        noise = np.random.normal(0, self.heading_noise_scale)
        heading_diff += noise
        
        # 限制转向率
        max_turn = self.motion_constraints.max_turn_rate * dt
        actual_turn = np.clip(heading_diff, -max_turn, max_turn)
        
        # 计算新的朝向（保持在0-360度范围内）
        new_heading = (current_heading + actual_turn) % 360
            
        return new_heading

    def _calculate_distance(self, pos1: Tuple[float, float],
                          pos2: Tuple[float, float]) -> float:
        """计算两点之间的欧氏距离"""
        return np.sqrt(
            (pos2[0] - pos1[0]) ** 2 +
            (pos2[1] - pos1[1]) ** 2
        )

    def _calculate_heading(self, pos1: Tuple[float, float],
                         pos2: Tuple[float, float]) -> float:
        """计算从pos1到pos2的朝向角（北为0度，顺时针）

        Args:
            pos1: 起始位置 (lon, lat)
            pos2: 目标位置 (lon, lat)

        Returns:
            朝向角（度）：北为0度，东为90度，南为180度，西为270度
        """
        dx = pos2[0] - pos1[0]  # 经度差（东西方向）
        dy = pos2[1] - pos1[1]  # 纬度差（南北方向）
        
        # 使用arctan2计算角度（弧度）
        # 注意：arctan2(y, x)的参数顺序，我们需要交换dx和dy来得到正确的角度
        angle_rad = np.arctan2(dx, dy)
        
        # 转换为度数并调整为以北为0度，东为90度
        angle_deg = np.degrees(angle_rad)
        # 如果角度为负，加360度使其在[0, 360)范围内
        if angle_deg < 0:
            angle_deg += 360
            
        return float(angle_deg)  # 确保返回Python float类型

    def _calculate_initial_heading(self, start_pos: Tuple[float, float],
                                 next_pos: Tuple[float, float]) -> float:
        """计算初始朝向"""
        return self._calculate_heading(start_pos, next_pos) 