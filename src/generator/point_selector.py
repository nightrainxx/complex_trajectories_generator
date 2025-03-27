"""起点选择器模块

此模块负责为给定的终点选择合适的起点。选择过程需要考虑：
1. 起点的可通行性（基于土地覆盖类型和坡度）
2. 起终点之间的最小距离约束
3. 起点的随机性（在满足约束的情况下）

输入:
    - 土地覆盖栅格文件 (.tif)
    - 坡度栅格文件 (.tif)
    - 终点坐标（像素坐标）
    - 约束参数（最小距离等）

输出:
    - 满足约束的起点坐标列表
"""

import numpy as np
import rasterio
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import random

from config import *

class PointSelector:
    """起点选择器类"""
    
    def __init__(self, landcover_path: str, slope_path: str):
        """初始化起点选择器
        
        Args:
            landcover_path: 土地覆盖栅格文件路径
            slope_path: 坡度栅格文件路径
        """
        # 检查文件是否存在
        for path in [landcover_path, slope_path]:
            if not Path(path).exists():
                raise FileNotFoundError(f"找不到文件: {path}")
        
        # 读取栅格数据
        with rasterio.open(landcover_path) as src:
            self.landcover_data = src.read(1)
            self.transform = src.transform
            self.meta = src.meta.copy()
            self.height = src.height
            self.width = src.width
        
        with rasterio.open(slope_path) as src:
            self.slope_data = src.read(1)
        
        # 验证数据形状一致
        if self.landcover_data.shape != self.slope_data.shape:
            raise ValueError("土地覆盖和坡度数据形状不一致")
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        
        # 计算像素大小（米）
        self.pixel_size_degrees = abs(self.transform[0])  # 经纬度分辨率（度）
        self.meters_per_degree = 111000  # 1度约等于111km
        self.pixel_size_meters = self.pixel_size_degrees * self.meters_per_degree
    
    def is_point_accessible(self, row: int, col: int) -> bool:
        """判断指定点是否可通行
        
        Args:
            row: 行号
            col: 列号
            
        Returns:
            bool: 是否可通行
        """
        # 检查边界
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        
        # 检查土地覆盖类型
        landcover = self.landcover_data[row, col]
        if landcover in IMPASSABLE_LANDCOVER_CODES:
            return False
        
        # 检查坡度
        slope = self.slope_data[row, col]
        if slope > MAX_SLOPE_THRESHOLD:
            return False
        
        return True
    
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """计算两点间的像素距离
        
        Args:
            point1: 第一个点的坐标 (row, col)
            point2: 第二个点的坐标 (row, col)
            
        Returns:
            float: 像素距离
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_geo_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """计算两点间的实际地理距离（米）
        
        Args:
            point1: 第一个点的坐标 (row, col)
            point2: 第二个点的坐标 (row, col)
            
        Returns:
            float: 地理距离（米）
        """
        # 转换为地理坐标
        lon1, lat1 = self.pixel_to_geo(point1)
        lon2, lat2 = self.pixel_to_geo(point2)
        
        # 使用简化的距离计算（平面近似）
        dx = (lon2 - lon1) * self.meters_per_degree * np.cos(np.radians((lat1 + lat2) / 2))
        dy = (lat2 - lat1) * self.meters_per_degree
        
        return np.sqrt(dx * dx + dy * dy)
    
    def select_start_points(
        self,
        end_point: Tuple[int, int],
        num_points: int = 1,
        min_distance: float = MIN_START_END_DISTANCE_METERS,
        max_attempts: int = 10000
    ) -> List[Tuple[int, int]]:
        """为给定终点选择合适的起点
        
        Args:
            end_point: 终点坐标 (row, col)
            num_points: 需要选择的起点数量
            min_distance: 起终点最小距离（米）
            max_attempts: 最大尝试次数
            
        Returns:
            List[Tuple[int, int]]: 选择的起点坐标列表
        """
        selected_points = []
        attempts = 0
        min_pixel_distance = min_distance / self.pixel_size_meters
        
        # 计算搜索范围
        search_radius = int(min_pixel_distance * 1.5)  # 增加50%的搜索范围
        
        while len(selected_points) < num_points and attempts < max_attempts:
            attempts += 1
            
            # 随机选择一个方向和距离
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(min_pixel_distance, min_pixel_distance * 2)
            
            # 计算候选起点坐标
            row = int(end_point[0] + distance * np.cos(angle))
            col = int(end_point[1] + distance * np.sin(angle))
            
            # 检查点是否可用
            if self.is_point_accessible(row, col):
                # 检查与已选点的距离
                too_close = False
                for point in selected_points:
                    if self.calculate_distance((row, col), point) < min_pixel_distance / 4:
                        too_close = True
                        break
                
                # 检查与终点的实际地理距离
                if not too_close:
                    geo_distance = self.calculate_geo_distance((row, col), end_point)
                    if geo_distance >= min_distance:
                        selected_points.append((row, col))
                        self.logger.debug(f"已选择起点: ({row}, {col})")
        
        if len(selected_points) < num_points:
            self.logger.warning(
                f"未能找到足够的起点（要求{num_points}个，找到{len(selected_points)}个）"
            )
        
        return selected_points
    
    def select_start_points_for_all_ends(
        self,
        end_points: List[Dict],
        points_per_end: int = NUM_TRAJECTORIES_PER_END
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """为所有终点选择起点
        
        Args:
            end_points: 终点列表，每个终点是一个字典，包含 'pixel' 和 'coord' 键
            points_per_end: 每个终点需要的起点数量
            
        Returns:
            List[Tuple[Tuple[int, int], Tuple[int, int]]]: 起终点对列表
        """
        start_end_pairs = []
        
        for end_point in end_points:
            # 为当前终点选择起点
            start_points = self.select_start_points(
                end_point['pixel'],
                num_points=points_per_end
            )
            
            # 添加起终点对
            for start_point in start_points:
                start_end_pairs.append((start_point, end_point['pixel']))
        
        self.logger.info(f"共生成{len(start_end_pairs)}对起终点")
        return start_end_pairs
    
    def pixel_to_geo(self, pixel: Tuple[int, int]) -> Tuple[float, float]:
        """将像素坐标转换为地理坐标
        
        Args:
            pixel: 像素坐标 (row, col)
            
        Returns:
            Tuple[float, float]: 地理坐标 (lon, lat)
        """
        lon, lat = self.transform * (pixel[1], pixel[0])
        return lon, lat
    
    def geo_to_pixel(self, coord: Tuple[float, float]) -> Tuple[int, int]:
        """将地理坐标转换为像素坐标
        
        Args:
            coord: 地理坐标 (lon, lat)
            
        Returns:
            Tuple[int, int]: 像素坐标 (row, col)
        """
        row, col = ~self.transform * coord
        return int(row), int(col) 