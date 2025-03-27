"""环境地图生成器模块

此模块负责生成环境地图，包括：
1. 最大速度地图：基于坡度大小和土地覆盖类型
2. 典型速度地图：考虑坡度方向对速度的影响
3. 速度标准差地图：反映速度的变化程度
4. 成本地图：用于路径规划

输入:
    - 土地覆盖栅格文件 (.tif)
    - 坡度大小栅格文件 (.tif)
    - 坡度方向栅格文件 (.tif)

输出:
    - 最大速度地图 (max_speed_map.tif)
    - 典型速度地图 (typical_speed_map.tif)
    - 速度标准差地图 (speed_stddev_map.tif)
    - 成本地图 (cost_map.tif)
"""

import numpy as np
import rasterio
from pathlib import Path
import logging
from typing import Tuple, Optional
import os

from src.generator.config import (
    MAX_SPEED, MAX_SLOPE_THRESHOLD, SLOPE_SPEED_FACTOR,
    TYPICAL_SPEED_FACTOR, UP_SLOPE_FACTOR, DOWN_SLOPE_FACTOR, CROSS_SLOPE_FACTOR,
    BASE_SPEED_STDDEV_FACTOR, SLOPE_STDDEV_FACTOR, COMPLEX_TERRAIN_STDDEV_FACTOR,
    COMPLEX_TERRAIN_CODES, LANDCOVER_SPEED_FACTORS, LANDCOVER_COST_FACTORS,
    IMPASSABLE_LANDCOVER_CODES
)

class EnvironmentMapper:
    """环境地图生成器类"""
    
    def __init__(self, landcover_path: str, slope_magnitude_path: str, slope_aspect_path: str):
        """初始化环境地图生成器
        
        Args:
            landcover_path: 土地覆盖栅格文件路径
            slope_magnitude_path: 坡度大小栅格文件路径
            slope_aspect_path: 坡度方向栅格文件路径
        """
        # 检查文件是否存在
        for path in [landcover_path, slope_magnitude_path, slope_aspect_path]:
            if not Path(path).exists():
                raise FileNotFoundError(f"找不到文件: {path}")
        
        # 读取栅格数据
        with rasterio.open(landcover_path) as src:
            self.landcover_data = src.read(1)
            self.transform = src.transform
            self.meta = src.meta.copy()
            self.height = src.height
            self.width = src.width
        
        with rasterio.open(slope_magnitude_path) as src:
            self.slope_magnitude_data = src.read(1)
        
        with rasterio.open(slope_aspect_path) as src:
            self.slope_aspect_data = src.read(1)
        
        # 验证数据形状一致
        if not (self.landcover_data.shape == self.slope_magnitude_data.shape == self.slope_aspect_data.shape):
            raise ValueError("输入数据形状不一致")
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
    
    def calculate_max_speed_map(self) -> np.ndarray:
        """计算最大速度地图
        
        基于坡度大小和土地覆盖类型计算每个像素的最大可能速度。
        不可通行区域（水体、冰川、陡峭区域）的速度设为0。
        
        Returns:
            np.ndarray: 最大速度地图（米/秒）
        """
        # 初始化最大速度地图
        max_speed = np.full(self.landcover_data.shape, MAX_SPEED, dtype=np.float32)
        
        # 处理不可通行区域
        impassable_mask = np.isin(self.landcover_data, IMPASSABLE_LANDCOVER_CODES)
        steep_mask = self.slope_magnitude_data > MAX_SLOPE_THRESHOLD
        max_speed[impassable_mask | steep_mask] = 0
        
        # 应用坡度影响
        slope_factor = np.clip(1 - SLOPE_SPEED_FACTOR * self.slope_magnitude_data, 0.1, 1.0)
        max_speed *= slope_factor
        
        # 应用土地覆盖影响
        for code, factor in LANDCOVER_SPEED_FACTORS.items():
            landcover_mask = self.landcover_data == code
            max_speed[landcover_mask] *= factor
        
        return max_speed
    
    def calculate_typical_speed_map(self) -> np.ndarray:
        """计算典型速度地图
        
        基于最大速度，考虑坡度方向对速度的影响。
        上坡时速度降低，下坡时速度略有提升，横坡时速度显著降低。
        
        Returns:
            np.ndarray: 典型速度地图（米/秒）
        """
        # 获取基础最大速度
        typical_speed = self.calculate_max_speed_map() * TYPICAL_SPEED_FACTOR
        
        # 处理平地（坡向为-1）
        flat_mask = self.slope_aspect_data == -1
        typical_speed[flat_mask] *= 1.0  # 平地不需要额外调整
        
        # 处理有坡度的区域
        slope_mask = ~flat_mask
        if np.any(slope_mask):
            # 计算不同方向的影响因子
            # 这里假设我们主要考虑南北方向的运动
            # 坡向0度是北向，180度是南向
            north_factor = np.where(
                self.slope_aspect_data < 90,
                1 - UP_SLOPE_FACTOR * self.slope_magnitude_data,
                1.0
            )
            south_factor = np.where(
                self.slope_aspect_data > 90,
                1 + DOWN_SLOPE_FACTOR * self.slope_magnitude_data,
                1.0
            )
            
            # 计算横坡影响（东西方向）
            cross_slope_factor = np.where(
                (self.slope_aspect_data >= 45) & (self.slope_aspect_data <= 135) |
                (self.slope_aspect_data >= 225) & (self.slope_aspect_data <= 315),
                1 - CROSS_SLOPE_FACTOR * self.slope_magnitude_data,
                1.0
            )
            
            # 组合所有影响因子
            combined_factor = np.minimum(north_factor, south_factor) * cross_slope_factor
            combined_factor = np.clip(combined_factor, 0.1, 1.2)  # 限制因子范围
            
            # 应用到典型速度
            typical_speed[slope_mask] *= combined_factor[slope_mask]
        
        return typical_speed
    
    def calculate_speed_stddev_map(self) -> np.ndarray:
        """计算速度标准差地图
        
        基于典型速度和地形复杂度计算速度的标准差。
        复杂地形（如山地）的标准差较大，平坦区域的标准差较小。
        
        Returns:
            np.ndarray: 速度标准差地图（米/秒）
        """
        # 获取典型速度
        typical_speed = self.calculate_typical_speed_map()
        
        # 初始化标准差地图
        speed_stddev = typical_speed * BASE_SPEED_STDDEV_FACTOR
        
        # 处理不可通行区域
        impassable_mask = np.isin(self.landcover_data, IMPASSABLE_LANDCOVER_CODES)
        steep_mask = self.slope_magnitude_data > MAX_SLOPE_THRESHOLD
        speed_stddev[impassable_mask | steep_mask] = 0
        
        # 增加复杂地形的标准差
        complex_mask = np.isin(self.landcover_data, COMPLEX_TERRAIN_CODES)
        speed_stddev[complex_mask] *= COMPLEX_TERRAIN_STDDEV_FACTOR
        
        # 根据坡度增加标准差
        slope_stddev_factor = np.clip(
            1 + SLOPE_STDDEV_FACTOR * (self.slope_magnitude_data / MAX_SLOPE_THRESHOLD),
            1.0,
            2.0
        )
        speed_stddev *= slope_stddev_factor
        
        return speed_stddev
    
    def calculate_cost_map(self) -> np.ndarray:
        """计算成本地图
        
        基于典型速度和土地覆盖类型计算通行成本。
        不可通行区域的成本设为无穷大。
        
        Returns:
            np.ndarray: 成本地图（秒/米）
        """
        # 获取典型速度
        typical_speed = self.calculate_typical_speed_map()
        
        # 初始化成本地图
        cost = np.zeros_like(typical_speed)
        
        # 处理不可通行区域
        impassable_mask = np.isin(self.landcover_data, IMPASSABLE_LANDCOVER_CODES)
        steep_mask = self.slope_magnitude_data > MAX_SLOPE_THRESHOLD
        cost[impassable_mask | steep_mask] = np.inf
        
        # 计算可通行区域的成本
        passable_mask = ~(impassable_mask | steep_mask)
        cost[passable_mask] = 1 / typical_speed[passable_mask]  # 基础成本：单位距离所需时间
        
        # 应用土地覆盖成本因子
        for code, factor in LANDCOVER_COST_FACTORS.items():
            landcover_mask = self.landcover_data == code
            cost[landcover_mask & passable_mask] *= factor
        
        return cost
    
    def save_environment_maps(
        self,
        output_dir: str,
        max_speed_map: np.ndarray,
        typical_speed_map: np.ndarray,
        speed_stddev_map: np.ndarray,
        cost_map: np.ndarray
    ) -> None:
        """保存环境地图
        
        Args:
            output_dir: 输出目录路径
            max_speed_map: 最大速度地图
            typical_speed_map: 典型速度地图
            speed_stddev_map: 速度标准差地图
            cost_map: 成本地图
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备元数据
        meta = self.meta.copy()
        meta.update(dtype=np.float32)
        
        # 保存最大速度地图
        with rasterio.open(os.path.join(output_dir, "max_speed_map.tif"), 'w', **meta) as dst:
            dst.write(max_speed_map.astype(np.float32), 1)
        
        # 保存典型速度地图
        with rasterio.open(os.path.join(output_dir, "typical_speed_map.tif"), 'w', **meta) as dst:
            dst.write(typical_speed_map.astype(np.float32), 1)
        
        # 保存速度标准差地图
        with rasterio.open(os.path.join(output_dir, "speed_stddev_map.tif"), 'w', **meta) as dst:
            dst.write(speed_stddev_map.astype(np.float32), 1)
        
        # 保存成本地图
        with rasterio.open(os.path.join(output_dir, "cost_map.tif"), 'w', **meta) as dst:
            dst.write(cost_map.astype(np.float32), 1)
        
        self.logger.info(f"环境地图已保存到目录: {output_dir}")
    
    def get_environment_params(self, row: int, col: int) -> dict:
        """获取指定位置的环境参数
        
        Args:
            row: 像素行号（从0开始）
            col: 像素列号（从0开始）
            
        Returns:
            包含环境参数的字典：
            {
                'max_speed': 最大速度 (m/s),
                'typical_speed': 典型速度 (m/s),
                'speed_stddev': 速度标准差 (m/s),
                'cost': 移动成本 (s/m),
                'landcover': 土地覆盖类型代码,
                'slope_magnitude': 坡度大小 (度),
                'slope_aspect': 坡向 (度),
                'is_passable': 是否可通行
            }
        
        Raises:
            ValueError: 如果位置超出范围
        """
        # 检查位置是否在有效范围内
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"位置 ({row}, {col}) 超出范围")
        
        # 获取土地覆盖和坡度信息
        landcover = self.landcover_data[row, col]
        slope_magnitude = self.slope_magnitude_data[row, col]
        slope_aspect = self.slope_aspect_data[row, col]
        
        # 判断是否可通行
        is_passable = (
            landcover not in IMPASSABLE_LANDCOVER_CODES and
            slope_magnitude <= MAX_SLOPE_THRESHOLD
        )
        
        # 获取环境参数
        max_speed = self.calculate_max_speed_map()[row, col]
        typical_speed = self.calculate_typical_speed_map()[row, col]
        speed_stddev = self.calculate_speed_stddev_map()[row, col]
        cost = self.calculate_cost_map()[row, col]
        
        return {
            'max_speed': float(max_speed),
            'typical_speed': float(typical_speed),
            'speed_stddev': float(speed_stddev),
            'cost': float(cost),
            'landcover': int(landcover),
            'slope_magnitude': float(slope_magnitude),
            'slope_aspect': float(slope_aspect),
            'is_passable': bool(is_passable)
        } 