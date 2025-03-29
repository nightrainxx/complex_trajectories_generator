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

# 从统一配置文件导入配置
from config import (
    MAX_UPHILL_SLOPE, MAX_DOWNHILL_SLOPE, MAX_CROSS_SLOPE,
    UP_SLOPE_FACTOR, DOWN_SLOPE_FACTOR, CROSS_SLOPE_FACTOR,
    TIME_STEP, MAX_ITERATIONS, POSITION_THRESHOLD,
    MAX_ACCELERATION, MAX_DECELERATION, MAX_TURN_RATE,
    MIN_SPEED, MIN_SPEED_STEEP_SLOPE
)

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class MotionConstraints:
    """运动约束参数"""
    max_acceleration: float = MAX_ACCELERATION  # 最大加速度 (m/s^2)
    max_deceleration: float = MAX_DECELERATION  # 最大减速度 (m/s^2)
    max_turn_rate: float = MAX_TURN_RATE       # 最大转向率 (度/秒)
    min_speed: float = MIN_SPEED              # 最小速度 (m/s)
    time_step: float = TIME_STEP              # 模拟时间步长 (秒)
    max_iterations: int = MAX_ITERATIONS      # 最大迭代次数
    position_threshold: float = POSITION_THRESHOLD  # 位置判断阈值 (m)

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
    max_uphill_slope: float = MAX_UPHILL_SLOPE    # 最大上坡坡度 (度)
    max_downhill_slope: float = MAX_DOWNHILL_SLOPE  # 最大下坡坡度 (度)
    max_cross_slope: float = MAX_CROSS_SLOPE     # 最大横坡坡度 (度)
    k_uphill: float = UP_SLOPE_FACTOR            # 上坡减速系数
    k_downhill: float = DOWN_SLOPE_FACTOR         # 下坡加速系数
    k_cross: float = CROSS_SLOPE_FACTOR             # 横坡减速系数
    min_speed_steep_slope: float = MIN_SPEED_STEEP_SLOPE # 陡坡最小速度 (m/s)

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
        
        # 记录使用的约束参数
        logger.info(f"使用的运动约束参数: 最大加速度={self.motion_constraints.max_acceleration}m/s², "
                  f"最大减速度={self.motion_constraints.max_deceleration}m/s², "
                  f"最大转向率={self.motion_constraints.max_turn_rate}°/s")
        logger.info(f"使用的地形约束参数: 最大上坡={self.terrain_constraints.max_uphill_slope}°, "
                  f"最大下坡={self.terrain_constraints.max_downhill_slope}°, "
                  f"最大横坡={self.terrain_constraints.max_cross_slope}°")

    def _calculate_direction_effects(self, current_heading: float,
                              env_params: EnvironmentParams) -> Tuple[float, float, float]:
        """计算行驶方向与坡向的关系对速度的影响
        
        注意：这里只考虑行驶方向与坡向的关系，不重复计算坡度大小的基础影响，
        因为坡度大小的影响已经包含在环境地图的typical_speed中。

        Args:
            current_heading: 当前朝向 (度)
            env_params: 环境参数

        Returns:
            Tuple[float, float, float]: (纵向坡度, 横向坡度, 方向性速度调整系数)
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

        # 计算方向性速度调整系数 (不重复考虑坡度大小的基础影响)
        direction_factor = 1.0

        # 上坡/下坡调整（轻微）
        if slope_along_path > 0:  # 上坡
            # 使用较小的系数，因为坡度大小的基础影响已在典型速度中
            direction_factor *= max(0.9, 1 - self.terrain_constraints.k_uphill * 0.3)
        else:  # 下坡
            direction_factor *= min(1.1, 1 + self.terrain_constraints.k_downhill * 0.3)

        # 横坡调整（仍然需要较强的影响，因为环境地图中没有考虑横坡）
        if cross_slope > 0:
            if cross_slope > self.terrain_constraints.max_cross_slope:
                direction_factor *= 0.7  # 严重横坡，显著减速
            else:
                factor = 1 - self.terrain_constraints.k_cross * (cross_slope / self.terrain_constraints.max_cross_slope)
                direction_factor *= max(0.8, factor)

        return slope_along_path, cross_slope, direction_factor

    def _calculate_target_speed(self, env_params: EnvironmentParams,
                              current_heading: float) -> float:
        """计算目标速度，考虑环境限制、行驶方向与坡向的关系和随机性
        
        注意：典型速度(env_params.typical_speed)已经包含了坡度大小和地物类型对速度的基础影响，
        这里只需要考虑行驶方向与坡向的关系产生的额外影响。

        Args:
            env_params: 环境参数
            current_heading: 当前朝向 (度)

        Returns:
            目标速度 (m/s)
        """
        # 基础目标速度是从环境地图获取的典型速度
        base_speed = env_params.typical_speed
        
        if base_speed <= 0:
            # 不可通行区域
            return 0.0
        
        # 计算方向性影响
        _, _, direction_factor = self._calculate_direction_effects(current_heading, env_params)
        
        # 应用方向性影响
        target_speed = base_speed * direction_factor
        
        # 添加随机扰动（高斯噪声）
        noise = np.random.normal(0, env_params.speed_stddev * self.speed_noise_scale)
        target_speed += noise
        
        # 限制在合理范围内
        target_speed = np.clip(
            target_speed,
            self.motion_constraints.min_speed,
            min(env_params.max_speed, env_params.typical_speed * 1.5)
        )
        
        return target_speed

    def simulate_motion(self, path: List[Tuple[float, float]],
                       env_params_func, force_path: bool = False) -> List[Tuple[float, float, float, float, float]]:
        """模拟目标在给定路径上的运动

        Args:
            path: 路径点列表 [(lon, lat), ...]
            env_params_func: 函数，输入(lon, lat)返回对应位置的EnvironmentParams
            force_path: 是否强制沿路径移动，如果为True，则忽略部分物理约束

        Returns:
            轨迹点列表 [(timestamp, lon, lat, speed, heading), ...]
        """
        if len(path) < 2:
            raise ValueError("路径至少需要包含两个点")
            
        # 记录force_path参数
        if force_path:
            logger.warning("使用force_path=True，将强制沿路径移动，可能导致速度不自然")

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
            
            # 记录调试信息
            if iteration_count % 100 == 0:
                logger.debug(f"迭代 {iteration_count}: 位置={current_pos}, 速度={current_speed:.2f}m/s, "
                           f"距目标点{distance:.2f}m, 典型速度={env_params.typical_speed:.2f}m/s")
            
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
            
            # 计算目标速度（考虑环境约束）
            target_speed = self._calculate_target_speed(env_params, current_heading)
            
            # 计算目标朝向
            # 如果force_path为True，则直接朝向目标点；否则，根据当前速度和朝向限制转向
            if force_path:
                next_heading = target_heading
            else:
                # 计算最大转向角
                max_turn_angle = self.motion_constraints.max_turn_rate * self.motion_constraints.time_step
                # 计算当前朝向与目标朝向之间的角度差
                heading_diff = self._normalize_angle(target_heading - current_heading)
                # 限制转向角度
                heading_change = np.clip(heading_diff, -max_turn_angle, max_turn_angle)
                next_heading = self._normalize_angle(current_heading + heading_change)
            
            # 计算加速度限制
            if target_speed > current_speed:
                # 加速
                speed_change = min(
                    target_speed - current_speed,
                    self.motion_constraints.max_acceleration * self.motion_constraints.time_step
                )
            else:
                # 减速
                speed_change = max(
                    target_speed - current_speed,
                    -self.motion_constraints.max_deceleration * self.motion_constraints.time_step
                )
            
            # 更新速度
            next_speed = current_speed + speed_change
            
            # 计算下一个位置
            # 如果force_path为True，则确保移动方向严格沿目标方向
            move_heading = next_heading if not force_path else target_heading
            move_distance = next_speed * self.motion_constraints.time_step
            
            # 计算移动增量
            delta_x = move_distance * np.sin(np.radians(move_heading))
            delta_y = move_distance * np.cos(np.radians(move_heading))
            
            # 如果force_path为True，则调整移动距离以确保朝向目标点
            if force_path and distance > 0:
                # 计算目标点方向的单位向量
                target_dx = (target_pos[0] - current_pos[0]) / distance
                target_dy = (target_pos[1] - current_pos[1]) / distance
                
                # 替换为沿目标方向移动
                delta_x = move_distance * target_dx
                delta_y = move_distance * target_dy
                
                # 确保不会越过目标点
                if move_distance > distance:
                    delta_x = target_pos[0] - current_pos[0]
                    delta_y = target_pos[1] - current_pos[1]
            
            # 更新位置
            next_pos = (current_pos[0] + delta_x, current_pos[1] + delta_y)
            
            # 更新时间和状态
            current_time += self.motion_constraints.time_step
            current_pos = next_pos
            current_speed = next_speed
            current_heading = next_heading
            
            # 添加轨迹点
            trajectory.append((
                current_time,
                current_pos[0],
                current_pos[1],
                current_speed,
                current_heading
            ))
            
            iteration_count += 1
        
        # 检查是否达到最大迭代次数
        if iteration_count >= self.motion_constraints.max_iterations:
            logger.warning(f"达到最大迭代次数 {self.motion_constraints.max_iterations}，模拟提前终止")
        
        # 记录模拟统计信息
        speeds = [point[3] for point in trajectory]
        if speeds:
            logger.info(f"模拟轨迹速度统计:")
            logger.info(f"  最大速度: {max(speeds):.2f} m/s")
            logger.info(f"  最小速度: {min(speeds):.2f} m/s")
            logger.info(f"  平均速度: {sum(speeds)/len(speeds):.2f} m/s")
            logger.info(f"  不同速度值数量: {len(set([round(s, 2) for s in speeds]))}")
            
            # 检查速度是否几乎不变
            if len(set([round(s, 1) for s in speeds])) < 5:
                logger.warning("模拟轨迹的速度几乎不变，可能存在问题")
        
        return trajectory
    
    def _calculate_initial_heading(self, start_pos: Tuple[float, float], next_pos: Tuple[float, float]) -> float:
        """计算初始朝向
        
        Args:
            start_pos: 起始位置
            next_pos: 下一个位置
            
        Returns:
            初始朝向（度）
        """
        return self._calculate_heading(start_pos, next_pos)
    
    def _calculate_heading(self, from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> float:
        """计算从一个位置到另一个位置的朝向
        
        Args:
            from_pos: 起始位置
            to_pos: 目标位置
            
        Returns:
            朝向（度，北为0，顺时针）
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # 计算方位角（弧度）
        heading_rad = np.arctan2(dx, dy)
        
        # 转换为度，并确保在[0, 360)范围内
        heading_deg = np.degrees(heading_rad)
        if heading_deg < 0:
            heading_deg += 360
            
        return heading_deg
    
    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到[0, 360)范围
        
        Args:
            angle: 输入角度（度）
            
        Returns:
            归一化后的角度（度）
        """
        angle = angle % 360
        if angle < 0:
            angle += 360
        return angle
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点之间的欧几里得距离
        
        Args:
            pos1: 第一个位置
            pos2: 第二个位置
            
        Returns:
            距离
        """
        return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2) 