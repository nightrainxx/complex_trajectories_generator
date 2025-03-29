"""
地形加载器模块
用于加载和管理地形数据
"""

import logging
import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import pyproj
from rasterio.crs import CRS

from src.utils.config import config

logger = logging.getLogger(__name__)

class TerrainLoader:
    """地形数据加载器"""
    
    def __init__(self):
        """初始化地形加载器"""
        self.dem_data: Optional[np.ndarray] = None
        self.landcover_data: Optional[np.ndarray] = None
        self.slope_data: Optional[np.ndarray] = None
        self.aspect_data: Optional[np.ndarray] = None
        self.resolution: Optional[float] = None  # 栅格分辨率（米）
        self.transform: Optional[Any] = None
        self.crs: Optional[CRS] = None
        self.utm_proj: Optional[pyproj.Proj] = None
        self.wgs84_proj: Optional[pyproj.Proj] = None
        self.transformer: Optional[pyproj.Transformer] = None
        self.utm_origin: Optional[Tuple[float, float]] = None  # UTM坐标系原点（左上角）
        
    def load_dem(self, dem_file: Path) -> None:
        """
        加载DEM数据
        
        Args:
            dem_file: DEM文件路径
        """
        logger.info(f"加载DEM数据: {dem_file}")
        with rasterio.open(dem_file) as src:
            self.dem_data = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            
            # 获取数据中心点的经纬度
            center_lon = (src.bounds.left + src.bounds.right) / 2
            center_lat = (src.bounds.bottom + src.bounds.top) / 2
            
            # 确定UTM区域
            utm_zone = int((center_lon + 180) / 6) + 1
            hemisphere = 'north' if center_lat >= 0 else 'south'
            
            # 创建投影转换器
            self.wgs84_proj = pyproj.Proj('epsg:4326')  # WGS84
            self.utm_proj = pyproj.Proj(
                f'+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84'
            )
            self.transformer = pyproj.Transformer.from_proj(
                self.wgs84_proj,
                self.utm_proj,
                always_xy=True
            )
            
            # 计算左上角的UTM坐标作为原点
            self.utm_origin = self.transformer.transform(
                src.bounds.left,
                src.bounds.top
            )
            
            # 设置栅格分辨率（米）
            self.resolution = 30.0
            
            logger.info(f"DEM分辨率: {self.resolution}米")
            logger.info(
                f"坐标系统: WGS84 -> UTM Zone {utm_zone} "
                f"{'North' if hemisphere == 'north' else 'South'}"
            )
            
            # 计算坡度和坡向
            self._calculate_slope_aspect()
            
    def _calculate_slope_aspect(self) -> None:
        """计算坡度和坡向"""
        if self.dem_data is None:
            raise ValueError("未加载DEM数据")
            
        # 计算x和y方向的梯度
        dy, dx = np.gradient(self.dem_data, self.resolution)
        
        # 计算坡度（度）
        slope_rad = np.arctan(np.sqrt(dx*dx + dy*dy))
        self.slope_data = np.degrees(slope_rad)
        
        # 计算坡向（度）
        aspect_rad = np.arctan2(dy, dx)
        self.aspect_data = np.degrees(aspect_rad)
        self.aspect_data[self.aspect_data < 0] += 360
            
    def load_landcover(self, landcover_file: Path) -> None:
        """
        加载土地覆盖数据
        
        Args:
            landcover_file: 土地覆盖文件路径
        """
        logger.info(f"加载土地覆盖数据: {landcover_file}")
        with rasterio.open(landcover_file) as src:
            self.landcover_data = src.read(1)
            
    def load_slope(self, slope_file: Path) -> None:
        """
        加载坡度数据
        
        Args:
            slope_file: 坡度文件路径
        """
        logger.info(f"加载坡度数据: {slope_file}")
        with rasterio.open(slope_file) as src:
            self.slope_data = src.read(1)
            
    def load_aspect(self, aspect_file: Path) -> None:
        """
        加载坡向数据
        
        Args:
            aspect_file: 坡向文件路径
        """
        logger.info(f"加载坡向数据: {aspect_file}")
        with rasterio.open(aspect_file) as src:
            self.aspect_data = src.read(1)
            
    def lonlat_to_utm(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        将经纬度坐标转换为UTM坐标（米）
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            Tuple[float, float]: (easting, northing) UTM坐标（米）
        """
        if self.transformer is None:
            raise ValueError("未初始化坐标转换器")
        return self.transformer.transform(lon, lat)
        
    def utm_to_lonlat(self, easting: float, northing: float) -> Tuple[float, float]:
        """
        将UTM坐标（米）转换为经纬度坐标
        
        Args:
            easting: UTM东坐标（米）
            northing: UTM北坐标（米）
            
        Returns:
            Tuple[float, float]: (经度, 纬度)
        """
        if self.transformer is None:
            raise ValueError("未初始化坐标转换器")
        return self.transformer.transform(easting, northing, direction='INVERSE')
        
    def utm_to_pixel(self, easting: float, northing: float) -> Tuple[int, int]:
        """
        将UTM坐标（米）转换为栅格坐标
        
        Args:
            easting: UTM东坐标（米）
            northing: UTM北坐标（米）
            
        Returns:
            Tuple[int, int]: (行号, 列号)
        """
        if self.utm_origin is None:
            raise ValueError("未初始化UTM原点")
            
        # 计算相对于原点的偏移（米）
        dx = easting - self.utm_origin[0]
        dy = self.utm_origin[1] - northing  # 北向坐标需要反转
        
        # 转换为栅格坐标
        col = int(dx / self.resolution)
        row = int(dy / self.resolution)
        
        return row, col
        
    def pixel_to_utm(self, row: int, col: int) -> Tuple[float, float]:
        """
        将栅格坐标转换为UTM坐标（米）
        
        Args:
            row: 行号
            col: 列号
            
        Returns:
            Tuple[float, float]: (easting, northing) UTM坐标（米）
        """
        if self.utm_origin is None:
            raise ValueError("未初始化UTM原点")
            
        # 计算UTM坐标
        easting = self.utm_origin[0] + col * self.resolution
        northing = self.utm_origin[1] - row * self.resolution
        
        return easting, northing
        
    def get_terrain_attributes(
            self,
            easting: float,
            northing: float
        ) -> Dict[str, float]:
        """
        获取指定UTM坐标位置的地形属性
        
        Args:
            easting: UTM东坐标（米）
            northing: UTM北坐标（米）
            
        Returns:
            Dict[str, float]: 地形属性字典
        """
        # 转换为栅格坐标
        row, col = self.utm_to_pixel(easting, northing)
        
        # 检查坐标是否在范围内
        if not (0 <= row < self.dem_data.shape[0] and 0 <= col < self.dem_data.shape[1]):
            return {}
            
        attributes = {}
        
        if self.dem_data is not None:
            attributes['elevation'] = float(self.dem_data[row, col])
            
        if self.landcover_data is not None:
            attributes['landcover'] = int(self.landcover_data[row, col])
            
        if self.slope_data is not None:
            attributes['slope'] = float(self.slope_data[row, col])
            
        if self.aspect_data is not None:
            attributes['aspect'] = float(self.aspect_data[row, col])
            
        return attributes
        
    def lonlat_to_pixel(
            self,
            lon: float,
            lat: float
        ) -> Tuple[int, int]:
        """
        将经纬度坐标转换为像素索引
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            Tuple[int, int]: (行索引, 列索引)
        """
        if self.transform is None:
            raise ValueError("请先加载DEM数据")
            
        # 使用仿射变换转换为像素坐标
        col, row = ~self.transform * (lon, lat)
        return int(row), int(col)
        
    def pixel_to_lonlat(
            self,
            row: int,
            col: int
        ) -> Tuple[float, float]:
        """
        将像素索引转换为经纬度坐标
        
        Args:
            row: 行索引
            col: 列索引
            
        Returns:
            Tuple[float, float]: (经度, 纬度)
        """
        if self.transform is None:
            raise ValueError("请先加载DEM数据")
            
        # 使用仿射变换转换为经纬度
        lon, lat = self.transform * (col, row)
        return lon, lat
        
    def is_valid_pixel(
            self,
            row: int,
            col: int
        ) -> bool:
        """
        检查像素坐标是否有效
        
        Args:
            row: 行索引
            col: 列索引
            
        Returns:
            bool: 是否有效
        """
        if self.dem_data is None:
            raise ValueError("未加载地形数据")
            
        height, width = self.dem_data.shape
        return (0 <= row < height and 
                0 <= col < width)
                
    def is_passable(
            self,
            row: int,
            col: int
        ) -> bool:
        """
        检查指定位置是否可通行
        
        Args:
            row: 行索引
            col: 列索引
            
        Returns:
            bool: 是否可通行
        """
        if not self.is_valid_pixel(row, col):
            return False
            
        if self.landcover_data is not None:
            landcover = self.landcover_data[row, col]
            if landcover in config['terrain']['IMPASSABLE_LANDCOVER_CODES']:
                return False
                
        if self.slope_data is not None:
            slope = self.slope_data[row, col]
            if slope > config['motion']['MAX_SLOPE_DEGREES']:
                return False
                
        return True 

def convert_to_pixel_coords(
    coords: np.ndarray,
    transform: Any
) -> np.ndarray:
    """
    将经纬度坐标转换为像素坐标

    Args:
        coords: 形状为(N, 2)的数组，每行包含[longitude, latitude]
        transform: rasterio的仿射变换矩阵

    Returns:
        np.ndarray: 形状为(N, 2)的数组，每行包含[row, col]
    """
    # 使用仿射变换矩阵进行转换
    cols = (coords[:, 0] - transform[2]) / transform[0]
    rows = (coords[:, 1] - transform[5]) / transform[4]
    
    # 将结果组合为(N, 2)数组
    return np.column_stack([rows, cols])

def convert_to_lonlat(
    pixel_coords: np.ndarray,
    transform: Any
) -> np.ndarray:
    """
    将像素坐标转换为经纬度坐标

    Args:
        pixel_coords: 形状为(N, 2)的数组，每行包含[row, col]
        transform: rasterio的仿射变换矩阵

    Returns:
        np.ndarray: 形状为(N, 2)的数组，每行包含[longitude, latitude]
    """
    # 使用仿射变换矩阵进行转换
    lons = transform[0] * pixel_coords[:, 1] + transform[2]
    lats = transform[4] * pixel_coords[:, 0] + transform[5]
    
    # 将结果组合为(N, 2)数组
    return np.column_stack([lons, lats]) 