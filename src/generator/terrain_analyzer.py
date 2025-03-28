"""地形分析器模块

此模块负责从DEM数据计算地形属性,包括:
1. 坡度大小(度)
2. 坡向(度,北为0,顺时针)

输入:
    - DEM栅格文件 (.tif)

输出:
    - 坡度大小栅格文件 (slope_magnitude_30m_100km.tif)
    - 坡向栅格文件 (slope_aspect_30m_100km.tif)
"""

import numpy as np
import rasterio
from pathlib import Path
import logging
from typing import Tuple
import os

class TerrainAnalyzer:
    """地形分析器类"""
    
    def __init__(self, dem_path: str):
        """初始化地形分析器
        
        Args:
            dem_path: DEM栅格文件路径
        """
        # 检查文件是否存在
        if not Path(dem_path).exists():
            raise FileNotFoundError(f"找不到DEM文件: {dem_path}")
        
        # 读取DEM数据
        with rasterio.open(dem_path) as src:
            self.dem_data = src.read(1)
            self.transform = src.transform
            self.meta = src.meta.copy()
            # 计算实际的像素大小(米)
            self.pixel_width = abs(self.transform[0]) * 111320  # 转换为米
            self.pixel_height = abs(self.transform[4]) * 111320  # 转换为米
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
    
    def calculate_slope_magnitude(self) -> np.ndarray:
        """计算坡度大小
        
        使用中心差分法计算每个像素的坡度大小(度)。
        边缘像素使用前向或后向差分。
        
        Returns:
            np.ndarray: 坡度大小(度)
        """
        rows, cols = self.dem_data.shape
        
        # 计算x和y方向的高程梯度
        dx = np.zeros_like(self.dem_data)
        dy = np.zeros_like(self.dem_data)
        
        # x方向梯度(中心差分)
        dx[:, 1:-1] = (self.dem_data[:, 2:] - self.dem_data[:, :-2]) / (2 * self.pixel_width)
        # 边缘使用前向/后向差分
        dx[:, 0] = (self.dem_data[:, 1] - self.dem_data[:, 0]) / self.pixel_width
        dx[:, -1] = (self.dem_data[:, -1] - self.dem_data[:, -2]) / self.pixel_width
        
        # y方向梯度(中心差分)
        dy[1:-1, :] = (self.dem_data[2:, :] - self.dem_data[:-2, :]) / (2 * self.pixel_height)
        # 边缘使用前向/后向差分
        dy[0, :] = (self.dem_data[1, :] - self.dem_data[0, :]) / self.pixel_height
        dy[-1, :] = (self.dem_data[-1, :] - self.dem_data[-2, :]) / self.pixel_height
        
        # 计算坡度(度)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        # 处理无效值
        slope = np.nan_to_num(slope, 0)  # 将NaN替换为0
        slope = np.clip(slope, 0, 90)  # 限制在0-90度范围内
        
        return slope
    
    def calculate_slope_aspect(self) -> np.ndarray:
        """计算坡向
        
        使用中心差分法计算每个像素的坡向(度,北为0,顺时针)。
        边缘像素使用前向或后向差分。
        平坦区域(坡度<0.1度)的坡向设为-1。
        
        Returns:
            np.ndarray: 坡向(度)
        """
        rows, cols = self.dem_data.shape
        
        # 计算x和y方向的高程梯度
        dx = np.zeros_like(self.dem_data)
        dy = np.zeros_like(self.dem_data)
        
        # x方向梯度(中心差分)
        dx[:, 1:-1] = (self.dem_data[:, 2:] - self.dem_data[:, :-2]) / (2 * self.pixel_width)
        # 边缘使用前向/后向差分
        dx[:, 0] = (self.dem_data[:, 1] - self.dem_data[:, 0]) / self.pixel_width
        dx[:, -1] = (self.dem_data[:, -1] - self.dem_data[:, -2]) / self.pixel_width
        
        # y方向梯度(中心差分)
        dy[1:-1, :] = (self.dem_data[2:, :] - self.dem_data[:-2, :]) / (2 * self.pixel_height)
        # 边缘使用前向/后向差分
        dy[0, :] = (self.dem_data[1, :] - self.dem_data[0, :]) / self.pixel_height
        dy[-1, :] = (self.dem_data[-1, :] - self.dem_data[-2, :]) / self.pixel_height
        
        # 计算坡向(弧度)
        aspect = np.arctan2(dx, dy)  # 使用arctan2确保正确的象限
        
        # 转换为度数并调整为北为0,顺时针
        aspect = np.degrees(aspect)
        aspect = 90.0 - aspect  # 转换为北为0
        aspect = np.where(aspect < 0, aspect + 360, aspect)  # 处理负值
        
        # 计算坡度用于识别平坦区域
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        # 平坦区域坡向设为-1
        aspect = np.where(slope < 0.1, -1, aspect)
        
        return aspect
    
    def save_terrain_maps(self, output_dir: str) -> Tuple[str, str]:
        """保存地形属性地图
        
        Args:
            output_dir: 输出目录路径
            
        Returns:
            Tuple[str, str]: 坡度和坡向文件的路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算坡度和坡向
        slope = self.calculate_slope_magnitude()
        aspect = self.calculate_slope_aspect()
        
        # 准备元数据
        meta = self.meta.copy()
        meta.update(dtype=np.float32)
        
        # 保存坡度地图
        slope_path = os.path.join(output_dir, "slope_magnitude_30m_100km.tif")
        with rasterio.open(slope_path, 'w', **meta) as dst:
            dst.write(slope.astype(np.float32), 1)
        
        # 保存坡向地图
        aspect_path = os.path.join(output_dir, "slope_aspect_30m_100km.tif")
        with rasterio.open(aspect_path, 'w', **meta) as dst:
            dst.write(aspect.astype(np.float32), 1)
        
        self.logger.info(f"地形属性地图已保存到目录: {output_dir}")
        return slope_path, aspect_path 