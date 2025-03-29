"""
运动模拟器模块
基于物理模型和环境约束进行运动模拟
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentMaps:
    """环境地图集合"""
    typical_speed: np.ndarray  # 典型速度图（来自学习）
    max_speed: np.ndarray     # 最大速度图（来自学习）
    speed_stddev: np.ndarray  # 速度标准差图（来自学习）
    slope_magnitude: np.ndarray  # 坡度大小图
    slope_aspect: np.ndarray    # 坡向图
    landcover: Optional[np.ndarray] = None  # 土地覆盖类型图

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
class AgentState:
    """智能体状态"""
    position: np.ndarray  # [easting, northing] (UTM坐标，米)
    speed: float         # 当前速度 (m/s)
    heading: float       # 当前朝向 (弧度)
    target_point: np.ndarray  # 目标点 [easting, northing] (UTM坐标，米)

@dataclass
class TrajectoryPoint:
    """轨迹点"""
    timestamp: float  # 时间戳(秒)
    easting: float   # UTM东坐标(米)
    northing: float  # UTM北坐标(米)
    lon: float       # 经度
    lat: float       # 纬度
    speed: float     # 速度(米/秒)
    heading: float   # 朝向(弧度)
    acceleration: float = 0.0  # 加速度(米/秒²)
    turn_rate: float = 0.0    # 转向率(度/秒)

class MotionSimulator:
    """运动模拟器"""
    
    def __init__(
            self,
            env_maps: EnvironmentMaps,
            terrain_loader: Any,  # TerrainLoader实例
            config: Dict
        ):
        """
        初始化运动模拟器
        
        Args:
            env_maps: 环境地图集合（包含从OORD数据学习到的速度图）
            terrain_loader: 地形加载器实例，用于坐标转换
            config: 配置参数字典（motion子字典）
        """
        self.env_maps = env_maps
        self.terrain_loader = terrain_loader
        
        # 使用环境地图中的速度值来设置配置参数
        self.config = config.copy()  # 创建配置的副本
        
        # 从环境地图中获取速度统计信息
        # 使用学习到的值设置速度参数
        self.config['DEFAULT_SPEED'] = float(np.median(env_maps.typical_speed))
        self.config['MAX_SPEED'] = float(np.max(env_maps.max_speed))
        self.config['MIN_SPEED'] = 0.1  # 设置一个合理的最小速度
        self.config['SPEED_STDDEV'] = float(np.median(env_maps.speed_stddev))
        
        # 输出学习到的速度参数
        logger.info(f"从环境地图学习到的速度参数:")
        logger.info(f"默认速度: {self.config['DEFAULT_SPEED']:.2f} m/s")
        logger.info(f"最大速度: {self.config['MAX_SPEED']:.2f} m/s")
        logger.info(f"最小速度: {self.config['MIN_SPEED']:.2f} m/s")
        logger.info(f"速度标准差: {self.config['SPEED_STDDEV']:.2f} m/s")
        
        # 验证地图尺寸一致性
        self._validate_maps()
        
        # 获取地图尺寸
        self.height, self.width = env_maps.typical_speed.shape
        
    def _validate_maps(self) -> None:
        """验证所有地图尺寸一致"""
        shape = self.env_maps.typical_speed.shape
        if self.env_maps.max_speed.shape != shape:
            raise ValueError("最大速度图尺寸不匹配")
        if self.env_maps.speed_stddev.shape != shape:
            raise ValueError("速度标准差图尺寸不匹配")
        if self.env_maps.slope_magnitude.shape != shape:
            raise ValueError("坡度图尺寸不匹配")
        if self.env_maps.slope_aspect.shape != shape:
            raise ValueError("坡向图尺寸不匹配")
        if (self.env_maps.landcover is not None and 
            self.env_maps.landcover.shape != shape):
            raise ValueError("土地覆盖图尺寸不匹配")
            
    def _get_environment_at_position(self, easting: float, northing: float) -> Dict:
        """
        获取指定位置的环境参数
        
        Args:
            easting: UTM东坐标(米)
            northing: UTM北坐标(米)
            
        Returns:
            Dict: 环境参数字典
        """
        # 转换UTM坐标为像素坐标
        pixel_i, pixel_j = self.terrain_loader.utm_to_pixel(easting, northing)
        
        # 检查像素是否在地图范围内
        if (0 <= pixel_i < self.height and 0 <= pixel_j < self.width):
            i, j = int(pixel_i), int(pixel_j)
            
            # 从环境地图获取参数
            typical_speed = float(self.env_maps.typical_speed[i, j])
            max_speed = float(self.env_maps.max_speed[i, j])
            speed_stddev = float(self.env_maps.speed_stddev[i, j])
            slope_magnitude = float(self.env_maps.slope_magnitude[i, j])
            slope_aspect = float(self.env_maps.slope_aspect[i, j])
            
            # 使用学习到的值，不再硬编码速度范围
            min_speed = max(self.config['MIN_SPEED'], typical_speed * 0.5)
            
            return {
                'typical_speed': typical_speed,
                'max_speed': max_speed,
                'min_speed': min_speed,
                'speed_stddev': speed_stddev,
                'slope_magnitude': slope_magnitude,
                'slope_aspect': slope_aspect,
                'max_acceleration': self.config['MAX_ACCELERATION'],
                'max_deceleration': self.config['MAX_DECELERATION'],
                'max_turn_rate': self.config['MAX_TURN_RATE']
            }
        else:
            # 超出地图范围，使用默认值
            return {
                'typical_speed': self.config['DEFAULT_SPEED'],
                'max_speed': self.config['MAX_SPEED'],
                'min_speed': self.config['MIN_SPEED'],
                'speed_stddev': self.config['SPEED_STDDEV'],
                'slope_magnitude': 0.0,
                'slope_aspect': 0.0,
                'max_acceleration': self.config['MAX_ACCELERATION'],
                'max_deceleration': self.config['MAX_DECELERATION'],
                'max_turn_rate': self.config['MAX_TURN_RATE']
            }
        
    def _calculate_slope_effects(
            self,
            slope_magnitude: float,
            slope_aspect: float,
            heading: float
        ) -> Tuple[float, float]:
        """
        计算坡度对运动的影响
        
        Args:
            slope_magnitude: 坡度大小(度)
            slope_aspect: 坡向(度)
            heading: 运动朝向(弧度)
            
        Returns:
            Tuple[float, float]: (沿路径坡度, 横向坡度)
        """
        # 将坡向和朝向转换为弧度
        aspect_rad = np.deg2rad(slope_aspect)
        
        # 计算坡度在运动方向上的分量
        slope_along_path = (
            slope_magnitude * np.cos(aspect_rad - heading)
        )
        
        # 计算横向坡度
        cross_slope = (
            slope_magnitude * np.sin(aspect_rad - heading)
        )
        
        return slope_along_path, cross_slope
        
    def _adjust_speed_for_terrain(
            self,
            base_speed: float,
            slope_along_path: float,
            cross_slope: float
        ) -> float:
        """
        根据地形调整速度
        
        Args:
            base_speed: 基础速度
            slope_along_path: 沿路径坡度
            cross_slope: 横向坡度
            
        Returns:
            float: 调整后的速度
        """
        # 获取配置参数
        slope_factor = self.config['SLOPE_SPEED_FACTOR']
        cross_slope_factor = self.config['CROSS_SLOPE_FACTOR']
        min_speed = self.config['MIN_SPEED']
        
        # 计算坡度影响
        slope_effect = 1.0 - slope_factor * abs(slope_along_path)
        cross_slope_effect = 1.0 - cross_slope_factor * abs(cross_slope)
        
        # 应用地形影响
        adjusted_speed = base_speed * slope_effect * cross_slope_effect
        
        # 确保速度不低于最小值
        return max(adjusted_speed, min_speed)
        
    def _update_agent_state(
            self,
            agent: AgentState,
            dt: float,
            force_target: Optional[np.ndarray] = None
        ) -> Tuple[AgentState, float, float]:
        """
        更新智能体状态
        
        Args:
            agent: 当前智能体状态
            dt: 时间步长
            force_target: 强制目标点（可选）
            
        Returns:
            更新后的智能体状态、加速度和转向率
        """
        # 获取当前位置的环境参数
        environment = self._get_environment_at_position(*agent.position)
        
        # 从环境中获取速度参数
        typical_speed = environment['typical_speed']
        max_speed = environment['max_speed']
        min_speed = environment['min_speed']
        speed_stddev = environment['speed_stddev']
        
        # 基于学习到的速度分布生成目标速度
        # 使用正态分布生成速度，中心在典型速度，标准差由学习得到
        target_speed = np.random.normal(typical_speed, speed_stddev)
        
        # 确保目标速度在合理范围内
        target_speed = np.clip(target_speed, min_speed, max_speed)
        
        # 随机添加小的波动，模拟30秒窗口内的速度变化
        # 每隔一定时间有30%的概率进行明显的速度调整
        if agent.speed > 0 and np.random.random() < 0.3:
            # 速度变化的方向（加速或减速）
            direction = np.random.choice([-1, 1])
            # 变化的幅度（基于学习到的标准差）
            magnitude = np.random.uniform(0.1, speed_stddev)
            # 应用变化
            target_speed = np.clip(agent.speed + direction * magnitude, min_speed, max_speed)
        
        # 计算到目标点的方向和距离
        target = force_target if force_target is not None else agent.target_point
        direction = target - agent.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            
            # 计算目标朝向
            target_heading = np.arctan2(direction[1], direction[0])
            
            # 计算转向角度
            heading_diff = (target_heading - agent.heading + np.pi) % (2 * np.pi) - np.pi
            
            # 根据转向角度调整速度
            angle_factor = np.cos(heading_diff)
            target_speed *= max(0.7, angle_factor)  # 转弯时保持至少70%的速度
            
            # 根据距离调整速度
            distance_factor = max(0.7, min(1.0, distance / 20.0))  # 接近目标点时保持至少70%的速度
            target_speed *= distance_factor
            
            # 计算加速度和转向率
            speed_diff = target_speed - agent.speed
            acceleration = np.clip(
                speed_diff / dt,
                -self.config['MAX_DECELERATION'],
                self.config['MAX_ACCELERATION']
            )
            
            turn_rate = np.clip(
                heading_diff / dt,
                -np.deg2rad(self.config['MAX_TURN_RATE']),
                np.deg2rad(self.config['MAX_TURN_RATE'])
            )
            
            # 更新状态
            new_speed = np.clip(
                agent.speed + acceleration * dt,
                min_speed,
                max_speed
            )
            
            new_heading = (agent.heading + turn_rate * dt) % (2 * np.pi)
            new_position = agent.position + new_speed * dt * np.array([
                np.cos(new_heading),
                np.sin(new_heading)
            ])
            
            return (
                AgentState(
                    position=new_position,
                    speed=new_speed,
                    heading=new_heading,
                    target_point=agent.target_point
                ),
                acceleration,
                turn_rate
            )
        else:
            # 如果已到达目标点，保持当前状态
            return agent, 0.0, 0.0
        
    def simulate_motion(
            self,
            path_points: List[Tuple[float, float]],  # UTM坐标点列表
            dt: float = 0.1,
            force_path: bool = False
        ) -> List[TrajectoryPoint]:
        """
        模拟运动轨迹
        
        Args:
            path_points: 路径点列表（UTM坐标，米）
            dt: 时间步长（秒）
            force_path: 是否强制沿路径运动
            
        Returns:
            List[TrajectoryPoint]: 轨迹点列表
        """
        if len(path_points) < 2:
            raise ValueError("路径点数量不足")
            
        # 初始化轨迹点列表
        trajectory = []
        speeds = []  # 记录所有速度值，用于调试
        speed_changes = []  # 记录速度变化
        
        # 获取起点和终点
        start_point = np.array(path_points[0], dtype=float)
        end_point = np.array(path_points[-1], dtype=float)
        
        # 打印路径信息
        print(f"路径起点: ({start_point[0]}, {start_point[1]})")
        print(f"路径终点: ({end_point[0]}, {end_point[1]})")
        print(f"路径总长度: {len(path_points)} 个点")
        
        # 确保坐标是UTM形式
        if not (100000 <= start_point[0] <= 1000000 and 100000 <= start_point[1] <= 10000000):
            print("警告: 起点坐标似乎不是UTM格式，这可能导致模拟错误")
            
        # 获取起点位置的环境参数
        environment = self._get_environment_at_position(*start_point)
        
        # 打印环境地图的统计信息，用于调试
        print(f"环境地图统计：")
        print(f"典型速度范围: {np.min(self.env_maps.typical_speed):.2f} - {np.max(self.env_maps.typical_speed):.2f} m/s")
        print(f"最大速度范围: {np.min(self.env_maps.max_speed):.2f} - {np.max(self.env_maps.max_speed):.2f} m/s")
        print(f"速度标准差范围: {np.min(self.env_maps.speed_stddev):.2f} - {np.max(self.env_maps.speed_stddev):.2f} m/s")
        
        print(f"起点位置速度参数: 典型={environment['typical_speed']:.2f}, 最大={environment['max_speed']:.2f}, 标准差={environment['speed_stddev']:.2f}")
        
        # 生成初始速度，基于学习到的速度分布
        initial_speed = np.random.normal(environment['typical_speed'], environment['speed_stddev'])
        initial_speed = np.clip(initial_speed, environment['min_speed'], environment['max_speed'])
        
        print(f"初始速度: {initial_speed:.2f} m/s")
        
        # 初始化智能体状态
        agent = AgentState(
            position=start_point,
            speed=initial_speed,
            heading=0.0,  # 初始朝向将在第一次更新时调整
            target_point=end_point
        )
        
        # 初始化时间和路径索引
        t = 0.0
        path_index = 1
        
        # 用于控制30秒窗口的速度变化
        last_speed_change_time = 0.0
        window_size_seconds = 30.0  # 30秒窗口
        
        while True:
            # 转换UTM坐标为经纬度
            lon, lat = self.terrain_loader.utm_to_lonlat(
                agent.position[0],
                agent.position[1]
            )
            
            # 记录当前状态
            trajectory.append(TrajectoryPoint(
                timestamp=t,
                easting=agent.position[0],
                northing=agent.position[1],
                lon=lon,
                lat=lat,
                speed=agent.speed,
                heading=agent.heading,
                acceleration=0.0,  # 将在更新后设置
                turn_rate=0.0      # 将在更新后设置
            ))
            speeds.append(agent.speed)
            
            # 检查是否到达终点
            if np.linalg.norm(agent.position - end_point) < 5.0:  # 5米阈值
                break
                
            # 更新目标点
            if force_path:
                # 强制沿路径运动时，使用路径点作为目标
                while (path_index < len(path_points) and
                       np.linalg.norm(agent.position - np.array(path_points[path_index])) < 5.0):
                    path_index += 1
                    
                if path_index < len(path_points):
                    force_target = np.array(path_points[path_index], dtype=float)
                else:
                    force_target = end_point
            else:
                force_target = None
                
            # 检查是否需要在30秒窗口内更新速度
            if t - last_speed_change_time >= window_size_seconds:
                # 到达30秒窗口边界，获取当前环境并更新目标速度
                environment = self._get_environment_at_position(*agent.position)
                
                # 生成新的目标速度
                new_target_speed = np.random.normal(environment['typical_speed'], environment['speed_stddev'])
                new_target_speed = np.clip(new_target_speed, environment['min_speed'], environment['max_speed'])
                
                # 如果新速度与当前速度差异很大，逐渐调整
                if abs(new_target_speed - agent.speed) > 2.0:
                    # 设置一个中间速度，使变化更加平滑
                    target_speed = agent.speed + np.sign(new_target_speed - agent.speed) * 1.0
                else:
                    target_speed = new_target_speed
                
                # 更新上次速度变化时间
                last_speed_change_time = t
                
            # 更新智能体状态
            agent, acceleration, turn_rate = self._update_agent_state(
                agent,
                dt,
                force_target
            )
            
            # 将加速度和转向率添加到前一个轨迹点
            if len(trajectory) > 0:
                trajectory[-1].acceleration = acceleration
                trajectory[-1].turn_rate = np.rad2deg(turn_rate)
                
            # 更新时间
            t += dt
            
            # 如果模拟时间过长，退出循环
            max_simulation_time = self.config.get('MAX_SIMULATION_TIME', 3600)  # 默认1小时
            if t > max_simulation_time:
                logger.warning(f"模拟达到最大时间限制: {t} 秒")
                break
                
        # 统计和打印速度分布情况
        final_speeds = np.array([point.speed for point in trajectory])
        
        print(f"\n速度分布统计:")
        print(f"平均速度: {np.mean(final_speeds):.2f} m/s")
        print(f"最大速度: {np.max(final_speeds):.2f} m/s")
        print(f"最小速度: {np.min(final_speeds):.2f} m/s")
        print(f"速度标准差: {np.std(final_speeds):.2f} m/s")
        print(f"不同速度值数量: {len(np.unique(final_speeds))}")
        
        # 确保有足够的点计算速度变化
        if len(final_speeds) > 1:
            speed_changes = np.diff(final_speeds)
            print(f"平均速度变化: {np.mean(np.abs(speed_changes)):.2f} m/s")
            print(f"最大速度变化: {np.max(np.abs(speed_changes)):.2f} m/s")
        else:
            print("速度变化统计: 轨迹点数量不足，无法计算速度变化")
        
        return trajectory 