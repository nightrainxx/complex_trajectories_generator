"""
基于环境的轨迹生成器
考虑地形和环境约束生成轨迹
"""

import logging
import numpy as np
from typing import List, Dict, Tuple

from src.utils.config import config
from src.core.terrain import TerrainLoader
from .generator import TrajectoryGenerator

logger = logging.getLogger(__name__)

class EnvironmentBasedGenerator(TrajectoryGenerator):
    """基于环境的轨迹生成器"""
    
    def __init__(
            self,
            terrain_loader: TerrainLoader,
            dt: float = config.motion.DT,
            max_waypoints: int = 10,
            min_waypoint_dist: float = 1000.0,
            max_waypoint_dist: float = 5000.0
        ):
        """
        初始化生成器
        
        Args:
            terrain_loader: 地形数据加载器
            dt: 时间步长（秒）
            max_waypoints: 最大路径点数
            min_waypoint_dist: 最小路径点间距（米）
            max_waypoint_dist: 最大路径点间距（米）
        """
        super().__init__(terrain_loader, dt)
        self.max_waypoints = max_waypoints
        self.min_waypoint_dist = min_waypoint_dist
        self.max_waypoint_dist = max_waypoint_dist
        
    def generate_trajectory(
            self,
            start_point: Tuple[float, float],
            end_point: Tuple[float, float]
        ) -> Dict[str, List[float]]:
        """
        生成轨迹
        
        Args:
            start_point: 起点坐标 (x, y)
            end_point: 终点坐标 (x, y)
            
        Returns:
            Dict[str, List[float]]: 轨迹数据
        """
        # 生成路径点
        waypoints = self._generate_waypoints(start_point, end_point)
        logger.info(f"生成{len(waypoints)}个路径点")
        
        # 插值生成轨迹点
        x, y = self._interpolate_path(waypoints)
        logger.info(f"插值得到{len(x)}个轨迹点")
        
        # 计算速度和朝向
        speeds = self._calculate_speeds(x, y)
        orientations = self._calculate_orientations(x, y)
        
        # 计算时间戳
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        timestamps = self._calculate_timestamps(distances, speeds)
        
        return {
            'timestamp': timestamps.tolist(),
            'x': x.tolist(),
            'y': y.tolist(),
            'speed': speeds.tolist(),
            'orientation': orientations.tolist()
        }
        
    def _generate_waypoints(
            self,
            start_point: Tuple[float, float],
            end_point: Tuple[float, float]
        ) -> List[Tuple[float, float]]:
        """
        生成路径点
        
        Args:
            start_point: 起点坐标 (x, y)
            end_point: 终点坐标 (x, y)
            
        Returns:
            List[Tuple[float, float]]: 路径点列表
        """
        waypoints = [start_point]
        current_point = start_point
        
        # 计算起终点直线距离
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        total_dist = np.sqrt(dx**2 + dy**2)
        
        # 确保总距离满足最小要求（80km）
        if total_dist < config.generation.MIN_START_END_DISTANCE:
            raise ValueError(
                f"起终点直线距离（{total_dist/1000:.1f}km）小于最小要求"
                f"（{config.generation.MIN_START_END_DISTANCE/1000:.1f}km）"
            )
        
        # 生成中间路径点
        while len(waypoints) < self.max_waypoints:
            # 计算到终点的剩余距离
            dx = end_point[0] - current_point[0]
            dy = end_point[1] - current_point[1]
            dist_to_end = np.sqrt(dx**2 + dy**2)
            
            # 如果已经足够接近终点，直接添加终点
            if dist_to_end <= self.max_waypoint_dist:
                waypoints.append(end_point)
                break
            
            # 生成新的路径点
            for _ in range(50):  # 最多尝试50次
                # 在当前点附近随机选择一个方向和距离
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(
                    self.min_waypoint_dist,
                    self.max_waypoint_dist
                )
                
                # 计算新点坐标
                new_x = current_point[0] + dist * np.cos(angle)
                new_y = current_point[1] + dist * np.sin(angle)
                
                # 转换为像素坐标
                row, col = self.terrain_loader.coord_to_pixel(new_x, new_y)
                
                # 检查新点是否可通行
                if not self.terrain_loader.is_valid_pixel(row, col):
                    continue
                    
                if not self.terrain_loader.is_passable(row, col):
                    continue
                
                # 检查新点是否朝向终点（夹角不超过90度）
                dx_new = end_point[0] - new_x
                dy_new = end_point[1] - new_y
                dist_new_to_end = np.sqrt(dx_new**2 + dy_new**2)
                
                # 计算前进方向与终点方向的夹角
                dot_product = dx_new * np.cos(angle) + dy_new * np.sin(angle)
                angle_to_end = np.arccos(
                    dot_product / (dist_new_to_end * dist)
                )
                
                if angle_to_end > np.pi/2:  # 夹角不超过90度
                    continue
                
                # 接受新点
                current_point = (new_x, new_y)
                waypoints.append(current_point)
                break
            else:
                # 如果50次尝试都失败，直接添加终点
                waypoints.append(end_point)
                break
        
        # 如果还没有添加终点，添加终点
        if waypoints[-1] != end_point:
            waypoints.append(end_point)
        
        return waypoints 