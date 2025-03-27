"""
数据加载模块
负责加载和预处理GIS数据（DEM、坡度、土地覆盖）和OORD轨迹数据
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

from ..config import DEM_DIR, LANDCOVER_DIR

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GISDataLoader:
    """GIS数据加载器，用于加载和处理DEM、坡度和土地覆盖数据"""
    
    def __init__(self):
        self.dem = None
        self.slope = None
        self.landcover = None
        self.transform = None
        self.crs = None
        self.nodata = None
    
    def load_dem(self, dem_path: Union[str, Path]) -> np.ndarray:
        """
        加载DEM数据
        
        Args:
            dem_path: DEM文件路径
            
        Returns:
            dem_data: DEM数组
        """
        try:
            with rasterio.open(dem_path) as src:
                self.dem = src.read(1)  # 读取第一个波段
                self.transform = src.transform
                self.crs = src.crs
                self.nodata = src.nodata
                logger.info(f"成功加载DEM数据，形状: {self.dem.shape}")
                return self.dem
        except Exception as e:
            logger.error(f"加载DEM数据失败: {str(e)}")
            raise
    
    def load_slope(self, slope_path: Union[str, Path]) -> np.ndarray:
        """
        加载坡度数据
        
        Args:
            slope_path: 坡度文件路径
            
        Returns:
            slope_data: 坡度数组
        """
        try:
            with rasterio.open(slope_path) as src:
                self.slope = src.read(1)
                # 验证与DEM的一致性
                if self.dem is not None:
                    assert self.slope.shape == self.dem.shape, "坡度数据与DEM形状不一致"
                logger.info(f"成功加载坡度数据，形状: {self.slope.shape}")
                return self.slope
        except Exception as e:
            logger.error(f"加载坡度数据失败: {str(e)}")
            raise
    
    def load_landcover(self, landcover_path: Union[str, Path]) -> np.ndarray:
        """
        加载土地覆盖数据
        
        Args:
            landcover_path: 土地覆盖文件路径
            
        Returns:
            landcover_data: 土地覆盖数组
        """
        try:
            with rasterio.open(landcover_path) as src:
                self.landcover = src.read(1)
                # 验证与DEM的一致性
                if self.dem is not None:
                    assert self.landcover.shape == self.dem.shape, "土地覆盖数据与DEM形状不一致"
                logger.info(f"成功加载土地覆盖数据，形状: {self.landcover.shape}")
                return self.landcover
        except Exception as e:
            logger.error(f"加载土地覆盖数据失败: {str(e)}")
            raise
    
    def get_pixel_coords(self, lon: float, lat: float) -> Tuple[int, int]:
        """
        将地理坐标转换为像素坐标
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            (row, col): 像素坐标元组
        """
        if self.transform is None:
            raise ValueError("未加载GIS数据，无法进行坐标转换")
        
        row, col = rowcol(self.transform, lon, lat)
        return row, col
    
    def get_elevation(self, row: int, col: int) -> float:
        """获取指定像素位置的高程值"""
        if self.dem is None:
            raise ValueError("未加载DEM数据")
        return self.dem[row, col]
    
    def get_slope(self, row: int, col: int) -> float:
        """获取指定像素位置的坡度值"""
        if self.slope is None:
            raise ValueError("未加载坡度数据")
        return self.slope[row, col]
    
    def get_landcover(self, row: int, col: int) -> int:
        """获取指定像素位置的土地覆盖类型"""
        if self.landcover is None:
            raise ValueError("未加载土地覆盖数据")
        return self.landcover[row, col] 