"""
OORD数据处理模块
负责处理和分析OORD轨迹数据
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

from .terrain_loader import TerrainLoader
from .terrain_analyzer import TerrainAnalyzer

logger = logging.getLogger(__name__)

class OORDProcessor:
    """OORD数据处理器"""
    
    def __init__(self, terrain_loader: Optional[TerrainLoader] = None):
        """
        初始化OORD数据处理器
        
        Args:
            terrain_loader: 地形数据加载器实例，如果为None则不进行地形分析
        """
        self.terrain_loader = terrain_loader
        self.terrain_analyzer = TerrainAnalyzer() if terrain_loader is not None else None
        self.trajectories: Dict[str, pd.DataFrame] = {}
        self.processed_trajectories: Dict[str, pd.DataFrame] = {}
        self.environment_stats: Optional[Dict] = None
        
    def load_trajectory(self, trajectory_file: Union[str, Path]) -> pd.DataFrame:
        """
        加载轨迹数据
        
        Args:
            trajectory_file: 轨迹文件路径
            
        Returns:
            pd.DataFrame: 处理后的轨迹数据
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(trajectory_file)
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
            
            # 计算速度和加速度
            df['speed'] = np.sqrt(
                df['velocity_north_ms']**2 + 
                df['velocity_east_ms']**2
            )
            df['acceleration'] = np.sqrt(
                df['acceleration_x_ms2']**2 + 
                df['acceleration_y_ms2']**2 + 
                df['acceleration_z_ms2']**2
            )
            
            # 计算航向角
            df['heading'] = np.degrees(np.arctan2(
                df['velocity_east_ms'],
                df['velocity_north_ms']
            )) % 360
            
            # 计算转向率
            df['turn_rate'] = np.sqrt(
                df['angular_velocity_x_rads']**2 + 
                df['angular_velocity_y_rads']**2 + 
                df['angular_velocity_z_rads']**2
            )
            
            # 如果有地形数据，添加地形相关信息
            if self.terrain_loader is not None:
                df['elevation'] = df.apply(
                    lambda row: self.terrain_loader.get_elevation(row['longitude'], row['latitude']),
                    axis=1
                )
                df['landcover'] = df.apply(
                    lambda row: self.terrain_loader.get_landcover(row['longitude'], row['latitude']),
                    axis=1
                )
            
            # 保存轨迹数据
            trajectory_id = Path(trajectory_file).stem
            self.trajectories[trajectory_id] = df
            
            return df
            
        except Exception as e:
            logger.error(f"处理轨迹文件 {trajectory_file} 失败: {str(e)}")
            raise
            
    def process_trajectory(
            self,
            trajectory_id: str,
            max_speed: float = 50.0
        ) -> pd.DataFrame:
        """
        处理轨迹数据
        
        Args:
            trajectory_id: 轨迹ID
            max_speed: 最大速度阈值，单位：米/秒
            
        Returns:
            pd.DataFrame: 处理后的轨迹数据
        """
        if trajectory_id not in self.trajectories:
            raise ValueError(f"未找到轨迹 {trajectory_id}")
        
        df = self.trajectories[trajectory_id].copy()
        
        # 速度过滤
        df = df[df['speed'] <= max_speed].copy()
        
        # 如果有地形数据，添加环境分组
        if self.terrain_loader is not None and self.terrain_analyzer is not None:
            # 确保地形分析器已初始化
            if self.terrain_analyzer.slope_magnitude is None:
                self.terrain_analyzer.load_dem(
                    self.terrain_loader.dem_data,
                    self.terrain_loader.resolution
                )
                self.terrain_analyzer.calculate_slope_magnitude()
            
            # 获取每个点的坡度
            df['slope_magnitude'] = df.apply(
                lambda row: self.terrain_analyzer.get_terrain_attributes(
                    row['longitude'], row['latitude']
                )['slope_magnitude'],
                axis=1
            )
            
            # 坡度分组
            df['slope_group'] = pd.cut(
                df['slope_magnitude'],
                bins=[0, 5, 15, 30, np.inf],
                labels=['flat', 'gentle', 'moderate', 'steep'],
                include_lowest=True  # 包含最小值
            )
            
            # 环境分组标签
            df['group_label'] = df.apply(
                lambda row: f"{row['slope_group']}_{row['landcover']}",
                axis=1
            )
        else:
            # 如果没有地形数据，使用默认值
            df['slope_magnitude'] = 0.0
            df['slope_group'] = 'flat'
            df['group_label'] = 'flat_0'
        
        # 保存处理后的轨迹
        self.processed_trajectories[trajectory_id] = df
        
        return df
        
    def analyze_environment_interaction(self) -> Dict:
        """
        分析轨迹与环境的交互关系
        
        Returns:
            Dict: 环境交互统计信息
        """
        if not self.processed_trajectories:
            raise ValueError("没有处理过的轨迹数据")
        
        # 合并所有处理过的轨迹
        all_trajectories = pd.concat(self.processed_trajectories.values())
        
        def safe_std(x: pd.Series) -> float:
            """安全计算标准差，当样本数小于2时返回0"""
            return float(x.std()) if len(x) > 1 else 0.0
        
        # 如果没有地形数据，只分析基本运动特征
        if self.terrain_loader is None:
            stats = {
                'overall': {
                    'speed_mean': float(all_trajectories['speed'].mean()),
                    'speed_std': safe_std(all_trajectories['speed']),
                    'speed_median': float(all_trajectories['speed'].median()),
                    'speed_max': float(all_trajectories['speed'].max()),
                    'acceleration_std': safe_std(all_trajectories['acceleration']),
                    'turn_rate_std': safe_std(all_trajectories['turn_rate']),
                    'sample_size': int(len(all_trajectories))
                }
            }
            return stats
        
        # 如果有地形数据，按环境分组分析
        stats = {}
        for group_label in all_trajectories['group_label'].unique():
            group_data = all_trajectories[all_trajectories['group_label'] == group_label]
            if len(group_data) > 0:  # 只处理非空组
                stats[group_label] = {
                    'speed_mean': float(group_data['speed'].mean()),
                    'speed_std': safe_std(group_data['speed']),
                    'speed_median': float(group_data['speed'].median()),
                    'speed_max': float(group_data['speed'].max()),
                    'acceleration_std': safe_std(group_data['acceleration']),
                    'turn_rate_std': safe_std(group_data['turn_rate']),
                    'sample_size': int(len(group_data))
                }
        
        return stats
        
    def _get_slope_magnitude(self, lon: float, lat: float) -> float:
        """
        获取指定位置的坡度大小
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            slope_magnitude: 坡度大小（度）
        """
        if self.terrain_analyzer.slope_magnitude is None:
            return 0.0
            
        row, col = self.terrain_loader.get_pixel_coords(lon, lat)
        if 0 <= row < self.terrain_analyzer.slope_magnitude.shape[0] and \
           0 <= col < self.terrain_analyzer.slope_magnitude.shape[1]:
            return float(self.terrain_analyzer.slope_magnitude[row, col])
        return 0.0
        
    @staticmethod
    def _calculate_haversine_distance(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """
        计算相邻点间的Haversine距离
        
        Args:
            lon: 经度数组
            lat: 纬度数组
            
        Returns:
            distances: 距离数组（米）
        """
        R = 6371000  # 地球半径（米）
        
        # 转换为弧度
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        
        # 计算差值
        dlon = np.diff(lon_rad)
        dlat = np.diff(lat_rad)
        
        # Haversine公式
        a = np.sin(dlat/2)**2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        distances = np.zeros_like(lon)
        distances[1:] = R * c
        return distances
        
    @staticmethod
    def _calculate_heading(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """
        计算航向角（度，北为0，顺时针）
        
        Args:
            lon: 经度数组
            lat: 纬度数组
            
        Returns:
            headings: 航向角数组（度）
        """
        # 转换为弧度
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        
        # 计算差值
        dlon = np.diff(lon_rad)
        dlat = np.diff(lat_rad)
        
        # 计算方位角
        y = np.sin(dlon) * np.cos(lat_rad[1:])
        x = np.cos(lat_rad[:-1]) * np.sin(lat_rad[1:]) - \
            np.sin(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.cos(dlon)
        
        heading_rad = np.arctan2(y, x)
        heading_deg = np.degrees(heading_rad) % 360
        
        headings = np.zeros_like(lon)
        headings[1:] = heading_deg
        headings[0] = headings[1]  # 第一个点使用第二个点的航向
        return headings

    def calculate_haversine_distance(
            self,
            lon1: float,
            lat1: float,
            lon2: float,
            lat2: float
        ) -> float:
        """
        计算两点间的Haversine距离
        
        Args:
            lon1: 起点经度
            lat1: 起点纬度
            lon2: 终点经度
            lat2: 终点纬度
            
        Returns:
            float: 两点间的距离，单位：公里
        """
        # 地球平均半径（公里）
        R = 6371.0
        
        # 将经纬度转换为弧度
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # 计算差值
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine公式
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance

    def calculate_heading(
            self,
            velocity_north: float,
            velocity_east: float
        ) -> float:
        """
        计算航向角
        
        Args:
            velocity_north: 北向速度分量
            velocity_east: 东向速度分量
            
        Returns:
            float: 航向角，单位：度，范围[0, 360)
        """
        heading = np.degrees(np.arctan2(velocity_east, velocity_north)) % 360
        return heading 