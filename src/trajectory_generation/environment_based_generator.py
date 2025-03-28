"""
基于环境的轨迹生成器
根据地形特征和环境约束生成合理的轨迹
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from .trajectory_generator import TrajectoryGenerator
from ..data_processing import TerrainLoader

logger = logging.getLogger(__name__)

class EnvironmentBasedGenerator(TrajectoryGenerator):
    """基于环境的轨迹生成器"""
    
    def __init__(self, terrain_loader: TerrainLoader):
        """
        初始化基于环境的轨迹生成器
        
        Args:
            terrain_loader: 地形数据加载器实例
        """
        super().__init__(terrain_loader)
        
        # 环境相关参数
        self.env_params = {
            # 坡度对速度的影响因子
            'slope_speed_factors': {
                'flat': 1.0,      # 平地 (0-5度)
                'gentle': 0.9,    # 缓坡 (5-15度)
                'moderate': 0.7,  # 中坡 (15-30度)
                'steep': 0.5      # 陡坡 (>30度)
            },
            # 坡向影响参数
            'slope_direction_params': {
                'k_uphill': 0.1,    # 上坡减速系数
                'k_downhill': 0.05,  # 下坡加速系数
                'k_cross': 0.2,      # 横坡减速系数
                'max_cross_slope': 30.0,  # 最大允许横坡角度(度)
                'min_speed_factor': 0.1   # 最小速度因子
            },
            # 地表类型对速度的影响因子
            'landcover_speed_factors': {
                1: 1.0,  # 道路
                2: 0.8,  # 植被
                3: 0.6   # 水体
            },
            'path_smoothness': 0.8,  # 路径平滑度 (0-1)
            'waypoint_spacing': 100   # 路径点间距（米）
        }
    
    def generate_trajectory(
            self,
            start_point: Tuple[float, float],
            end_point: Tuple[float, float],
            params: Optional[Dict] = None
        ) -> pd.DataFrame:
        """
        生成轨迹
        
        Args:
            start_point: 起点坐标（经度, 纬度）
            end_point: 终点坐标（经度, 纬度）
            params: 生成参数，可选
            
        Returns:
            pd.DataFrame: 生成的轨迹数据
        """
        # 更新参数
        if params:
            self.update_params(params)
        
        # 检查地形数据
        self._check_terrain_data()
        
        # 验证起终点
        if not (self._validate_point(*start_point) and self._validate_point(*end_point)):
            raise ValueError("起点或终点超出有效范围")
        
        # 生成路径点
        waypoints = self._generate_waypoints(start_point, end_point)
        
        # 生成轨迹点
        trajectory = self._generate_trajectory_points(waypoints)
        
        # 验证轨迹
        if not self.validate_trajectory(trajectory):
            logger.warning("生成的轨迹不满足约束条件，尝试重新生成")
            return self.generate_trajectory(start_point, end_point, params)
        
        return trajectory
    
    def validate_trajectory(self, trajectory: pd.DataFrame) -> bool:
        """
        验证轨迹是否满足约束条件
        
        Args:
            trajectory: 轨迹数据
            
        Returns:
            bool: 是否有效
        """
        # 检查速度约束
        speed = np.sqrt(
            trajectory['velocity_north_ms']**2 + 
            trajectory['velocity_east_ms']**2
        )
        if not (self.params['min_speed'] <= speed).all() or \
           not (speed <= self.params['max_speed']).all():
            return False
        
        # 检查加速度约束
        acceleration = np.sqrt(
            trajectory['acceleration_x_ms2']**2 + 
            trajectory['acceleration_y_ms2']**2 + 
            trajectory['acceleration_z_ms2']**2
        )
        if not (acceleration <= self.params['max_acceleration']).all():
            return False
        
        # 检查转向率约束
        turn_rate = np.sqrt(
            trajectory['angular_velocity_x_rads']**2 + 
            trajectory['angular_velocity_y_rads']**2 + 
            trajectory['angular_velocity_z_rads']**2
        )
        if not (turn_rate <= np.radians(self.params['max_turn_rate'])).all():
            return False
        
        return True
    
    def _generate_waypoints(
            self,
            start_point: Tuple[float, float],
            end_point: Tuple[float, float]
        ) -> List[Tuple[float, float]]:
        """
        生成路径点
        
        Args:
            start_point: 起点坐标
            end_point: 终点坐标
            
        Returns:
            List[Tuple[float, float]]: 路径点列表
        """
        # 计算直线距离
        distance = self._calculate_distance(start_point, end_point)
        
        # 计算需要的路径点数量
        num_points = max(3, int(distance / self.env_params['waypoint_spacing']))
        
        # 生成初始路径点（直线插值）
        t = np.linspace(0, 1, num_points)
        lon = np.interp(t, [0, 1], [start_point[0], end_point[0]])
        lat = np.interp(t, [0, 1], [start_point[1], end_point[1]])
        
        # 根据地形调整路径点
        waypoints = []
        for i in range(num_points):
            point = self._adjust_point_by_terrain(lon[i], lat[i])
            waypoints.append(point)
        
        return waypoints
    
    def _calculate_speed_factors(
            self,
            slope_magnitude: float,
            slope_aspect: float,
            heading: float,
            landcover: int
        ) -> Tuple[float, float]:
        """
        计算速度影响因子
        
        Args:
            slope_magnitude: 坡度大小(度)
            slope_aspect: 坡向(度,北为0,顺时针)
            heading: 行进方向(度,北为0,顺时针)
            landcover: 地表类型编码
            
        Returns:
            Tuple[float, float]: (最大速度因子, 典型速度因子)
        """
        # 计算坡向影响
        delta_angle = heading - slope_aspect
        # 处理角度环绕
        if delta_angle > 180:
            delta_angle -= 360
        elif delta_angle < -180:
            delta_angle += 360
            
        # 计算沿路径坡度和横向坡度
        slope_along_path = slope_magnitude * np.cos(np.radians(delta_angle))
        cross_slope = slope_magnitude * abs(np.sin(np.radians(delta_angle)))
        
        # 获取坡向参数
        params = self.env_params['slope_direction_params']
        
        # 计算上下坡影响
        if slope_along_path > 0:  # 上坡
            along_factor = max(
                params['min_speed_factor'],
                1 - params['k_uphill'] * slope_along_path
            )
        else:  # 下坡
            along_factor = min(
                1.2,  # 限制下坡最大加速
                1 + params['k_downhill'] * abs(slope_along_path)
            )
        
        # 计算横坡影响
        cross_factor = max(
            params['min_speed_factor'],
            1 - params['k_cross'] * (cross_slope / params['max_cross_slope'])**2
        )
        
        # 如果横坡超过最大允许值,显著降低速度
        if cross_slope > params['max_cross_slope']:
            cross_factor = params['min_speed_factor']
        
        # 获取基础坡度影响因子
        if slope_magnitude <= 5:
            base_factor = self.env_params['slope_speed_factors']['flat']
        elif slope_magnitude <= 15:
            base_factor = self.env_params['slope_speed_factors']['gentle']
        elif slope_magnitude <= 30:
            base_factor = self.env_params['slope_speed_factors']['moderate']
        else:
            base_factor = self.env_params['slope_speed_factors']['steep']
        
        # 获取地表影响因子
        landcover_factor = self.env_params['landcover_speed_factors'].get(
            landcover,
            0.5  # 默认因子
        )
        
        # 综合各种影响
        max_speed_factor = base_factor * along_factor * cross_factor * landcover_factor
        typical_speed_factor = max_speed_factor * 0.8  # 典型速度略低于最大速度
        
        return max_speed_factor, typical_speed_factor
    
    def _generate_trajectory_points(
            self,
            waypoints: List[Tuple[float, float]]
        ) -> pd.DataFrame:
        """
        根据路径点生成轨迹点
        
        Args:
            waypoints: 路径点列表
            
        Returns:
            pd.DataFrame: 轨迹数据
        """
        # 使用三次样条插值生成平滑路径
        t = np.arange(len(waypoints))
        lon = [p[0] for p in waypoints]
        lat = [p[1] for p in waypoints]
        cs_lon = CubicSpline(t, lon)
        cs_lat = CubicSpline(t, lat)
        
        # 生成时间序列
        total_distance = sum(
            self._calculate_distance(waypoints[i], waypoints[i+1])
            for i in range(len(waypoints)-1)
        )
        total_time = total_distance / (self.params['max_speed'] * 0.5)  # 估计总时间
        num_points = int(total_time / self.params['time_step']) + 1
        t_fine = np.linspace(0, len(waypoints)-1, num_points)
        
        # 生成位置序列
        longitude = cs_lon(t_fine)
        latitude = cs_lat(t_fine)
        
        # 初始化数组
        timestamp_ms = np.zeros(num_points, dtype=np.int64)
        altitude = np.zeros(num_points)
        velocity_north = np.zeros(num_points)
        velocity_east = np.zeros(num_points)
        velocity_down = np.zeros(num_points)
        acceleration_x = np.zeros(num_points)
        acceleration_y = np.zeros(num_points)
        acceleration_z = np.zeros(num_points)
        angular_velocity_z = np.zeros(num_points)
        
        # 计算初始航向
        heading = np.degrees(np.arctan2(
            longitude[1] - longitude[0],
            latitude[1] - latitude[0]
        ))
        if heading < 0:
            heading += 360
            
        # 初始速度设为最小速度
        speed = self.params['min_speed']
        
        # 逐点生成轨迹
        for i in range(num_points):
            # 更新时间戳
            timestamp_ms[i] = int(i * self.params['time_step'] * 1000)
            
            # 获取当前位置的环境信息
            terrain_attrs = self.terrain_analyzer.get_terrain_attributes(
                longitude[i],
                latitude[i]
            )
            landcover = self.terrain_loader.get_landcover(
                longitude[i],
                latitude[i]
            )
            
            # 计算速度影响因子
            max_factor, typ_factor = self._calculate_speed_factors(
                terrain_attrs['slope_magnitude'],
                terrain_attrs['slope_aspect'],
                heading,
                landcover
            )
            
            # 计算目标速度
            target_speed = self.params['max_speed'] * typ_factor
            
            # 应用加速度限制
            speed_diff = target_speed - speed
            if speed_diff > 0:
                accel = min(speed_diff / self.params['time_step'],
                          self.params['max_acceleration'])
            else:
                accel = max(speed_diff / self.params['time_step'],
                          self.params['max_deceleration'])
            
            # 更新速度
            speed = speed + accel * self.params['time_step']
            speed = np.clip(speed,
                          self.params['min_speed'],
                          self.params['max_speed'] * max_factor)
            
            # 分解速度到南北和东西方向
            velocity_north[i] = speed * np.cos(np.radians(heading))
            velocity_east[i] = speed * np.sin(np.radians(heading))
            
            # 计算高程变化引起的垂直速度
            if i > 0:
                altitude[i] = self.terrain_loader.get_elevation(
                    longitude[i],
                    latitude[i]
                )
                dz = altitude[i] - altitude[i-1]
                dt = self.params['time_step']
                velocity_down[i] = -dz / dt  # 注意符号：向上为负
            
            # 计算加速度
            if i > 0:
                acceleration_x[i] = (velocity_east[i] - velocity_east[i-1]) / dt
                acceleration_y[i] = (velocity_north[i] - velocity_north[i-1]) / dt
                acceleration_z[i] = (velocity_down[i] - velocity_down[i-1]) / dt
            
            # 如果不是最后一个点，更新航向
            if i < num_points - 1:
                next_heading = np.degrees(np.arctan2(
                    longitude[i+1] - longitude[i],
                    latitude[i+1] - latitude[i]
                ))
                if next_heading < 0:
                    next_heading += 360
                
                # 计算航向变化
                heading_change = next_heading - heading
                if heading_change > 180:
                    heading_change -= 360
                elif heading_change < -180:
                    heading_change += 360
                
                # 应用转向率限制
                max_change = self.params['max_turn_rate'] * self.params['time_step']
                heading_change = np.clip(heading_change, -max_change, max_change)
                
                # 更新航向
                heading = heading + heading_change
                if heading < 0:
                    heading += 360
                elif heading >= 360:
                    heading -= 360
                
                # 计算角速度
                angular_velocity_z[i] = np.radians(heading_change / self.params['time_step'])
        
        # 创建轨迹数据
        data = {
            'timestamp_ms': timestamp_ms,
            'longitude': longitude,
            'latitude': latitude,
            'altitude_m': altitude,
            'velocity_north_ms': velocity_north,
            'velocity_east_ms': velocity_east,
            'velocity_down_ms': velocity_down,
            'acceleration_x_ms2': acceleration_x,
            'acceleration_y_ms2': acceleration_y,
            'acceleration_z_ms2': acceleration_z,
            'angular_velocity_x_rads': np.zeros_like(longitude),
            'angular_velocity_y_rads': np.zeros_like(longitude),
            'angular_velocity_z_rads': angular_velocity_z
        }
        
        return pd.DataFrame(data)
    
    def _adjust_point_by_terrain(
            self,
            lon: float,
            lat: float
        ) -> Tuple[float, float]:
        """
        根据地形特征调整路径点
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            Tuple[float, float]: 调整后的坐标
        """
        # 获取地形特征
        terrain_attrs = self.terrain_analyzer.get_terrain_attributes(lon, lat)
        slope = terrain_attrs['slope_magnitude']
        
        # 获取地表类型
        landcover = self.terrain_loader.get_landcover(lon, lat)
        
        # 根据坡度和地表类型调整点位置（简单示例）
        if slope > 30:  # 陡坡
            # 尝试在周围找到更平缓的位置
            for offset in [(0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001)]:
                new_lon = lon + offset[0]
                new_lat = lat + offset[1]
                if self._validate_point(new_lon, new_lat):
                    new_slope = self.terrain_analyzer.get_terrain_attributes(
                        new_lon, new_lat
                    )['slope_magnitude']
                    if new_slope < slope:
                        return new_lon, new_lat
        
        return lon, lat
    
    def _calculate_distance(
            self,
            point1: Tuple[float, float],
            point2: Tuple[float, float]
        ) -> float:
        """
        计算两点间的距离（米）
        
        Args:
            point1: 第一个点的坐标（经度, 纬度）
            point2: 第二个点的坐标（经度, 纬度）
            
        Returns:
            float: 距离（米）
        """
        # 使用简化的距离计算（平面近似）
        lon1, lat1 = point1
        lon2, lat2 = point2
        
        dx = (lon2 - lon1) * 111000 * np.cos(np.radians((lat1 + lat2) / 2))
        dy = (lat2 - lat1) * 111000
        
        return np.sqrt(dx**2 + dy**2) 