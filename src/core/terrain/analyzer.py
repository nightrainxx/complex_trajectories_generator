"""
地形分析模块
用于计算和分析地形属性
"""

import logging
import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, Optional

from src.utils.config import config

logger = logging.getLogger(__name__)

class TerrainAnalyzer:
    """地形分析器"""
    
    def __init__(self):
        """初始化地形分析器"""
        self.dem_data: Optional[np.ndarray] = None
        self.resolution: Optional[float] = None
        self.slope_magnitude: Optional[np.ndarray] = None
        self.slope_aspect: Optional[np.ndarray] = None
        self.dzdx: Optional[np.ndarray] = None
        self.dzdy: Optional[np.ndarray] = None
    
    def load_dem(self, dem_file: Path) -> None:
        """
        加载DEM数据
        
        Args:
            dem_file: DEM文件路径
        """
        logger.info(f"加载DEM数据: {dem_file}")
        with rasterio.open(dem_file) as src:
            self.dem_data = src.read(1)
            # 获取分辨率（假设x和y方向分辨率相同）
            self.resolution = src.transform[0]
            logger.info(f"DEM分辨率: {self.resolution}米")
    
    def calculate_slope_magnitude(self) -> None:
        """计算坡度大小（度）"""
        if self.dem_data is None:
            raise ValueError("请先加载DEM数据")
        
        logger.info("计算坡度大小...")
        
        # 计算x和y方向的梯度
        self.dzdx, self.dzdy = np.gradient(
            self.dem_data,
            self.resolution
        )
        
        # 计算坡度大小（度）
        self.slope_magnitude = np.degrees(
            np.arctan(
                np.sqrt(
                    np.square(self.dzdx) + 
                    np.square(self.dzdy)
                )
            )
        )
        
        logger.info(
            f"坡度范围: {self.slope_magnitude.min():.2f}° - "
            f"{self.slope_magnitude.max():.2f}°"
        )
    
    def calculate_slope_aspect(self) -> None:
        """计算坡向（度，北为0，顺时针）"""
        if self.dzdx is None or self.dzdy is None:
            self.calculate_slope_magnitude()
        
        logger.info("计算坡向...")
        
        # 计算坡向（弧度）
        aspect_rad = np.arctan2(self.dzdx, self.dzdy)
        
        # 转换为度数并调整为北为0
        self.slope_aspect = np.degrees(aspect_rad)
        self.slope_aspect = 90.0 - self.slope_aspect
        self.slope_aspect[self.slope_aspect < 0] += 360.0
        
        # 处理平坦区域（坡度<1度）
        flat_mask = self.slope_magnitude < 1.0
        self.slope_aspect[flat_mask] = -1
        
        logger.info("坡向计算完成")
    
    def get_terrain_attributes(
            self,
            row: int,
            col: int
        ) -> Tuple[float, float]:
        """
        获取指定位置的地形属性
        
        Args:
            row: 行索引
            col: 列索引
            
        Returns:
            Tuple[float, float]: (坡度大小, 坡向)
        """
        if (self.slope_magnitude is None or 
            self.slope_aspect is None):
            raise ValueError("请先计算坡度和坡向")
        
        return (
            self.slope_magnitude[row, col],
            self.slope_aspect[row, col]
        )
    
    def save_results(self) -> None:
        """保存计算结果"""
        if self.slope_magnitude is None:
            raise ValueError("请先计算坡度")
        
        # 保存坡度
        with rasterio.open(
            config.paths.SLOPE_FILE,
            'w',
            driver='GTiff',
            height=self.slope_magnitude.shape[0],
            width=self.slope_magnitude.shape[1],
            count=1,
            dtype=self.slope_magnitude.dtype,
            crs='+proj=latlong',
            transform=rasterio.transform.from_origin(
                0, 0, self.resolution, self.resolution
            )
        ) as dst:
            dst.write(self.slope_magnitude, 1)
        
        if self.slope_aspect is not None:
            # 保存坡向
            with rasterio.open(
                config.paths.ASPECT_FILE,
                'w',
                driver='GTiff',
                height=self.slope_aspect.shape[0],
                width=self.slope_aspect.shape[1],
                count=1,
                dtype=self.slope_aspect.dtype,
                crs='+proj=latlong',
                transform=rasterio.transform.from_origin(
                    0, 0, self.resolution, self.resolution
                )
            ) as dst:
                dst.write(self.slope_aspect, 1)
        
        logger.info("地形分析结果已保存") 