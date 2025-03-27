"""
轨迹生成器模块
负责生成符合环境约束的合成轨迹
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..analysis import EnvironmentAnalyzer
from ..config import TRAJECTORY_COLUMNS
from ..data_processing import GISDataLoader

# 配置日志
logger = logging.getLogger(__name__)

class TrajectoryGenerator:
    """轨迹生成器，用于生成符合环境约束的合成轨迹"""
    
    def __init__(self, 
                 gis_loader: GISDataLoader,
                 env_analyzer: EnvironmentAnalyzer,
                 time_step: float = 1.0):
        """
        初始化轨迹生成器
        
        Args:
            gis_loader: GIS数据加载器实例
            env_analyzer: 环境分析器实例，包含已训练的环境-运动特征模型
            time_step: 轨迹采样时间步长（秒），默认为1秒
        """
        self.gis_loader = gis_loader
        self.env_analyzer = env_analyzer
        self.time_step = time_step
    
    def generate_trajectory(self,
                          start_point: Tuple[float, float],
                          end_point: Tuple[float, float],
                          start_time: pd.Timestamp) -> pd.DataFrame:
        """
        生成一条从起点到终点的合成轨迹
        
        Args:
            start_point: 起点坐标 (longitude, latitude)
            end_point: 终点坐标 (longitude, latitude)
            start_time: 轨迹起始时间
            
        Returns:
            trajectory_df: 生成的轨迹数据，包含时间戳、位置、速度等信息
        """
        # 规划路径
        waypoints = self._plan_path(start_point, end_point)
        
        # 生成运动轨迹
        trajectory_df = self._generate_motion(waypoints, start_time)
        
        return trajectory_df
    
    def generate_trajectories(self,
                            num_trajectories: int,
                            region_bounds: Tuple[float, float, float, float],
                            time_range: Tuple[pd.Timestamp, pd.Timestamp]) -> Dict[str, pd.DataFrame]:
        """
        在指定区域和时间范围内生成多条轨迹
        
        Args:
            num_trajectories: 要生成的轨迹数量
            region_bounds: 区域边界 (min_lon, min_lat, max_lon, max_lat)
            time_range: 时间范围 (start_time, end_time)
            
        Returns:
            trajectories: 生成的轨迹字典，key为轨迹ID
        """
        min_lon, min_lat, max_lon, max_lat = region_bounds
        start_time, end_time = time_range
        time_range_seconds = (end_time - start_time).total_seconds()
        
        trajectories = {}
        for i in range(num_trajectories):
            # 随机生成起点和终点
            start_point = (
                np.random.uniform(min_lon, max_lon),
                np.random.uniform(min_lat, max_lat)
            )
            end_point = (
                np.random.uniform(min_lon, max_lon),
                np.random.uniform(min_lat, max_lat)
            )
            
            # 随机生成起始时间
            start_offset = np.random.uniform(0, time_range_seconds)
            traj_start_time = start_time + pd.Timedelta(seconds=start_offset)
            
            try:
                # 生成轨迹
                trajectory = self.generate_trajectory(
                    start_point=start_point,
                    end_point=end_point,
                    start_time=traj_start_time
                )
                
                # 生成轨迹ID并保存
                traj_id = f"TRAJ_{i+1:04d}"
                trajectories[traj_id] = trajectory
                
                logger.info(f"成功生成轨迹 {traj_id}")
                
            except Exception as e:
                logger.error(f"生成第 {i+1} 条轨迹时出错: {str(e)}")
                continue
        
        return trajectories
    
    def _plan_path(self,
                   start_point: Tuple[float, float],
                   end_point: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        规划从起点到终点的路径
        
        Args:
            start_point: 起点坐标 (longitude, latitude)
            end_point: 终点坐标 (longitude, latitude)
            
        Returns:
            waypoints: 路径上的关键点列表
        """
        # 计算起点和终点之间的距离
        lon_diff = end_point[0] - start_point[0]
        lat_diff = end_point[1] - start_point[1]
        total_distance = np.sqrt(lon_diff**2 + lat_diff**2)
        
        # 根据总距离确定路径点数量（每0.005度约0.5km）
        num_points = max(2, int(total_distance / 0.005) + 1)
        
        # 生成路径点
        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)  # 插值参数，从0到1
            lon = start_point[0] + t * lon_diff
            lat = start_point[1] + t * lat_diff
            
            # 对于最后一个点，使用精确的终点坐标
            if i == num_points - 1:
                lon, lat = end_point
            
            waypoints.append((lon, lat))
        
        return waypoints
    
    def _generate_motion(self,
                        waypoints: List[Tuple[float, float]],
                        start_time: pd.Timestamp) -> pd.DataFrame:
        """
        根据路径点生成符合环境约束的运动轨迹
        
        Args:
            waypoints: 路径关键点列表
            start_time: 起始时间
            
        Returns:
            motion_df: 生成的运动轨迹数据
        """
        # 初始化轨迹数据列表
        timestamps = []
        longitudes = []
        latitudes = []
        elevations = []
        speeds = []
        headings = []
        turn_rates = []
        accelerations = []
        
        current_time = start_time
        current_speed = 0.0
        current_heading = 0.0
        
        # 遍历相邻路径点对
        for i in range(len(waypoints) - 1):
            p1, p2 = waypoints[i], waypoints[i + 1]
            
            # 计算两点之间的距离和方向
            lon_diff = p2[0] - p1[0]
            lat_diff = p2[1] - p1[1]
            segment_distance = np.sqrt(lon_diff**2 + lat_diff**2)
            target_heading = np.degrees(np.arctan2(lat_diff, lon_diff)) % 360
            
            # 获取当前位置的环境信息
            pixel_coords = self.gis_loader.get_pixel_coords(p1[0], p1[1])
            elevation = self.gis_loader.get_elevation(*pixel_coords)
            slope = self.gis_loader.get_slope(*pixel_coords)
            landcover = self.gis_loader.get_landcover(*pixel_coords)
            
            # 根据环境条件采样目标速度
            target_speed = self.env_analyzer.sample_speed(landcover, slope)
            
            # 计算航向角变化
            heading_diff = (target_heading - current_heading + 180) % 360 - 180
            num_steps = max(1, int(segment_distance / (target_speed * self.time_step)))
            heading_step = heading_diff / num_steps
            
            # 生成该段轨迹的运动数据
            for step in range(num_steps):
                t = step / num_steps  # 插值参数
                
                # 更新位置
                if step == num_steps - 1 and i == len(waypoints) - 2:
                    # 最后一个点使用精确的终点坐标
                    lon, lat = p2
                else:
                    lon = p1[0] + t * lon_diff
                    lat = p1[1] + t * lat_diff
                
                # 更新航向
                prev_heading = current_heading
                current_heading = (current_heading + heading_step) % 360
                turn_rate = (current_heading - prev_heading) / self.time_step
                
                # 更新速度
                prev_speed = current_speed
                current_speed = prev_speed + (target_speed - prev_speed) * 0.1  # 平滑加速
                acceleration = (current_speed - prev_speed) / self.time_step
                
                # 记录数据点
                timestamps.append(current_time)
                longitudes.append(lon)
                latitudes.append(lat)
                elevations.append(elevation)
                speeds.append(current_speed)
                headings.append(current_heading)
                turn_rates.append(turn_rate)
                accelerations.append(acceleration)
                
                current_time += pd.Timedelta(seconds=self.time_step)
        
        # 创建DataFrame
        motion_df = pd.DataFrame({
            'timestamp': timestamps,
            'longitude': longitudes,
            'latitude': latitudes,
            'elevation': elevations,
            'speed': speeds,
            'heading': headings,
            'turn_rate': turn_rates,
            'acceleration': accelerations
        })
        
        return motion_df 