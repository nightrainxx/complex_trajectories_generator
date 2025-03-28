"""
GIS数据加载模块
负责加载DEM、土地覆盖等GIS数据
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

logger = logging.getLogger(__name__)

class TerrainLoader:
    """GIS地形数据加载器"""
    
    def __init__(self):
        """初始化地形数据加载器"""
        self.dem_data: Optional[np.ndarray] = None
        self.landcover_data: Optional[np.ndarray] = None
        self.transform = None
        self.crs = None
        self.resolution = None
        self.bounds = None
        
    def load_dem(self, dem_path: Union[str, Path]) -> np.ndarray:
        """
        加载DEM数据
        
        Args:
            dem_path: DEM文件路径(.tif格式)
            
        Returns:
            dem_array: DEM数据数组
        """
        try:
            with rasterio.open(dem_path) as src:
                self.dem_data = src.read(1)  # 读取第一个波段
                self.transform = src.transform
                self.crs = src.crs
                self.resolution = (src.res[0], src.res[1])
                self.bounds = src.bounds
                
            logger.info(f"成功加载DEM数据，形状: {self.dem_data.shape}")
            return self.dem_data
            
        except RasterioIOError as e:
            logger.error(f"加载DEM文件失败: {str(e)}")
            raise
            
    def load_landcover(self, landcover_path: Union[str, Path]) -> np.ndarray:
        """
        加载土地覆盖数据
        
        Args:
            landcover_path: 土地覆盖数据文件路径(.tif格式)
            
        Returns:
            landcover_array: 土地覆盖数据数组
        """
        try:
            with rasterio.open(landcover_path) as src:
                self.landcover_data = src.read(1)
                
                # 验证与DEM的一致性
                if self.dem_data is not None:
                    if src.shape != self.dem_data.shape:
                        raise ValueError("土地覆盖数据与DEM形状不一致")
                    if src.transform != self.transform:
                        raise ValueError("土地覆盖数据与DEM空间参考不一致")
                
            logger.info(f"成功加载土地覆盖数据，形状: {self.landcover_data.shape}")
            return self.landcover_data
            
        except RasterioIOError as e:
            logger.error(f"加载土地覆盖数据失败: {str(e)}")
            raise
            
    def get_pixel_coords(self, lon: float, lat: float) -> Tuple[int, int]:
        """
        将经纬度坐标转换为像素坐标
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            (row, col): 像素坐标
        """
        if self.transform is None:
            raise ValueError("未加载任何地形数据")
            
        col, row = ~self.transform * (lon, lat)
        return int(row), int(col)
        
    def get_geo_coords(self, row: int, col: int) -> Tuple[float, float]:
        """
        将像素坐标转换为经纬度坐标
        
        Args:
            row: 行号
            col: 列号
            
        Returns:
            (lon, lat): 经纬度坐标
        """
        if self.transform is None:
            raise ValueError("未加载任何地形数据")
            
        lon, lat = self.transform * (col, row)
        return lon, lat
        
    def get_elevation(self, lon: float, lat: float) -> float:
        """
        获取指定位置的高程值
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            elevation: 高程值
        """
        if self.dem_data is None:
            raise ValueError("未加载DEM数据")
            
        row, col = self.get_pixel_coords(lon, lat)
        if 0 <= row < self.dem_data.shape[0] and 0 <= col < self.dem_data.shape[1]:
            return float(self.dem_data[row, col])
        else:
            raise ValueError("坐标超出DEM范围")
            
    def get_landcover(self, lon: float, lat: float) -> int:
        """
        获取指定位置的土地覆盖类型
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            landcover_code: 土地覆盖类型代码
        """
        if self.landcover_data is None:
            raise ValueError("未加载土地覆盖数据")
            
        row, col = self.get_pixel_coords(lon, lat)
        if 0 <= row < self.landcover_data.shape[0] and 0 <= col < self.landcover_data.shape[1]:
            return int(self.landcover_data[row, col])
        else:
            raise ValueError("坐标超出土地覆盖数据范围") 