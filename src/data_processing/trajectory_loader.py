"""
OORD轨迹数据加载和预处理模块
负责加载、清洗和预处理OORD轨迹数据
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ..config import CORE_TRAJECTORIES_DIR

# 配置日志
logger = logging.getLogger(__name__)

class TrajectoryLoader:
    """OORD轨迹数据加载器"""
    
    def __init__(self):
        """初始化轨迹加载器"""
        self.trajectories: Dict[str, pd.DataFrame] = {}
        self.processed_trajectories: Dict[str, pd.DataFrame] = {}
    
    def load_trajectory(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        加载单个轨迹文件
        
        Args:
            file_path: 轨迹文件路径
            
        Returns:
            trajectory_df: 包含轨迹数据的DataFrame
        """
        try:
            # 假设轨迹文件是CSV格式，包含timestamp,longitude,latitude列
            df = pd.read_csv(file_path)
            
            # 验证必要的列是否存在
            required_columns = ['timestamp', 'longitude', 'latitude']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"轨迹文件缺少必要的列: {missing_columns}")
            
            # 确保时间戳是datetime类型
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 按时间戳排序
            df = df.sort_values('timestamp')
            
            # 将轨迹存储在字典中
            trajectory_id = Path(file_path).stem
            self.trajectories[trajectory_id] = df
            
            logger.info(f"成功加载轨迹 {trajectory_id}，共 {len(df)} 个点")
            return df
            
        except Exception as e:
            logger.error(f"加载轨迹文件 {file_path} 失败: {str(e)}")
            raise
    
    def load_all_trajectories(self, directory: Union[str, Path] = CORE_TRAJECTORIES_DIR) -> Dict[str, pd.DataFrame]:
        """
        加载目录中的所有轨迹文件
        
        Args:
            directory: 轨迹文件目录
            
        Returns:
            trajectories: 轨迹数据字典，键为轨迹ID
        """
        directory = Path(directory)
        try:
            for file_path in directory.glob("*.csv"):
                self.load_trajectory(file_path)
            
            logger.info(f"成功加载 {len(self.trajectories)} 条轨迹")
            return self.trajectories
            
        except Exception as e:
            logger.error(f"加载轨迹目录 {directory} 失败: {str(e)}")
            raise
    
    def preprocess_trajectory(self, trajectory_id: str, 
                            min_speed: float = 0.1,
                            max_speed: float = 50.0,
                            min_distance: float = 1.0) -> pd.DataFrame:
        """
        预处理单条轨迹，计算速度、方向等特征，并进行异常值过滤
        
        Args:
            trajectory_id: 轨迹ID
            min_speed: 最小合理速度（米/秒）
            max_speed: 最大合理速度（米/秒）
            min_distance: 最小点间距离（米）
            
        Returns:
            processed_df: 处理后的轨迹DataFrame
        """
        if trajectory_id not in self.trajectories:
            raise KeyError(f"轨迹 {trajectory_id} 不存在")
            
        df = self.trajectories[trajectory_id].copy()
        
        # 计算时间差（秒）
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        
        # 计算相邻点间的距离（米）
        coords = df[['longitude', 'latitude']].values
        distances = np.zeros(len(df))
        distances[1:] = self._haversine_distance(coords[:-1], coords[1:])
        df['distance'] = distances
        
        # 计算速度（米/秒）
        df['speed'] = df['distance'] / df['time_diff']
        
        # 计算方向角（度）
        df['heading'] = self._calculate_heading(coords)
        
        # 计算转向率（度/秒）
        df['turn_rate'] = df['heading'].diff() / df['time_diff']
        
        # 过滤异常值
        mask = (
            (df['speed'] >= min_speed) & 
            (df['speed'] <= max_speed) &
            (df['distance'] >= min_distance)
        )
        
        df_filtered = df[mask].copy()
        
        # 重新计算过滤后的特征
        df_filtered['time_diff'] = df_filtered['timestamp'].diff().dt.total_seconds()
        df_filtered['acceleration'] = df_filtered['speed'].diff() / df_filtered['time_diff']
        
        # 存储处理后的轨迹
        self.processed_trajectories[trajectory_id] = df_filtered
        
        logger.info(f"轨迹 {trajectory_id} 预处理完成，保留 {len(df_filtered)}/{len(df)} 个点")
        return df_filtered
    
    def preprocess_all_trajectories(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        预处理所有已加载的轨迹
        
        Args:
            **kwargs: 传递给preprocess_trajectory的参数
            
        Returns:
            processed_trajectories: 处理后的轨迹数据字典
        """
        for trajectory_id in self.trajectories:
            self.preprocess_trajectory(trajectory_id, **kwargs)
        
        logger.info(f"完成 {len(self.processed_trajectories)} 条轨迹的预处理")
        return self.processed_trajectories
    
    @staticmethod
    def _haversine_distance(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
        """
        计算两点间的Haversine距离（米）
        
        Args:
            point1: [longitude, latitude] 数组
            point2: [longitude, latitude] 数组
            
        Returns:
            distances: 距离数组（米）
        """
        R = 6371000  # 地球半径（米）
        
        # 转换为弧度
        lat1, lon1 = np.radians(point1[:, 1]), np.radians(point1[:, 0])
        lat2, lon2 = np.radians(point2[:, 1]), np.radians(point2[:, 0])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    @staticmethod
    def _calculate_heading(coords: np.ndarray) -> np.ndarray:
        """
        计算轨迹点的方向角（度）
        
        Args:
            coords: [[longitude, latitude], ...] 数组
            
        Returns:
            headings: 方向角数组（度）
        """
        # 初始化方向角数组
        headings = np.zeros(len(coords))
        
        # 计算相邻点的经纬度差
        dlon = np.radians(np.diff(coords[:, 0]))
        lat1 = np.radians(coords[:-1, 1])
        lat2 = np.radians(coords[1:, 1])
        
        # 计算方位角
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        heading_rad = np.arctan2(y, x)
        
        # 转换为度数并调整到[0, 360)范围
        headings[1:] = (np.degrees(heading_rad) + 360) % 360
        headings[0] = headings[1]  # 第一个点的方向与第二个点相同
        
        return headings 