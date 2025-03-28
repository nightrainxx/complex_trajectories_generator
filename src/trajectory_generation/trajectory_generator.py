"""
轨迹生成器抽象基类
定义轨迹生成的基本接口和通用功能
"""

import abc
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..data_processing import TerrainLoader, TerrainAnalyzer

class TrajectoryGenerator(abc.ABC):
    """轨迹生成器抽象基类"""
    
    def __init__(self, terrain_loader: TerrainLoader):
        """
        初始化轨迹生成器
        
        Args:
            terrain_loader: 地形数据加载器实例
        """
        self.terrain_loader = terrain_loader
        self.terrain_analyzer = TerrainAnalyzer(terrain_loader)
        
        # 基本运动参数
        self.params = {
            'min_speed': 0.1,           # 最小速度 (m/s)
            'max_speed': 20.0,          # 最大速度 (m/s)
            'max_acceleration': 2.0,     # 最大加速度 (m/s^2)
            'max_deceleration': -3.0,    # 最大减速度 (m/s^2)
            'max_turn_rate': 45.0,       # 最大转向率 (度/秒)
            'time_step': 0.1             # 时间步长 (秒)
        }
    
    @abc.abstractmethod
    def generate_trajectory(
            self,
            start_point: Tuple[float, float],
            end_point: Tuple[float, float],
            params: Optional[Dict] = None
        ) -> pd.DataFrame:
        """
        生成轨迹（抽象方法）
        
        Args:
            start_point: 起点坐标（经度, 纬度）
            end_point: 终点坐标（经度, 纬度）
            params: 生成参数，可选
            
        Returns:
            pd.DataFrame: 生成的轨迹数据
        """
        pass
    
    def update_params(self, params: Dict) -> None:
        """
        更新参数
        
        Args:
            params: 新的参数字典
        """
        self.params.update(params)
    
    def _check_terrain_data(self) -> None:
        """检查地形数据是否已加载"""
        if self.terrain_loader.dem_data is None:
            raise ValueError("DEM数据未加载")
        if self.terrain_loader.landcover_data is None:
            raise ValueError("土地覆盖数据未加载")
    
    def _validate_point(self, lon: float, lat: float) -> bool:
        """
        验证点是否在有效范围内
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            bool: 是否有效
        """
        # 检查是否在中国范围内
        if not (73 <= lon <= 135 and 18 <= lat <= 53):
            return False
        
        # 检查是否在地形数据范围内
        try:
            row, col = self.terrain_loader.transform_coordinates(lon, lat)
            if not (0 <= row < self.terrain_loader.dem_data.shape[0] and
                   0 <= col < self.terrain_loader.dem_data.shape[1]):
                return False
        except:
            return False
        
        # 检查是否为有效数据点
        elevation = self.terrain_loader.get_elevation(lon, lat)
        if np.isnan(elevation):
            return False
        
        return True 