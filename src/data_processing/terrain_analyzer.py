"""
地形分析模块
负责计算和分析DEM数据，生成坡度、坡向等地形特征
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
import richdem as rd

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
        
    def load_dem(self, dem_data: np.ndarray, resolution: float):
        """
        加载DEM数据
        
        Args:
            dem_data: DEM数据数组
            resolution: 空间分辨率（米）
        """
        self.dem_data = dem_data
        self.resolution = resolution
        logger.info(f"加载DEM数据，形状: {dem_data.shape}, 分辨率: {resolution}米")
        
    def calculate_slope_magnitude(self) -> np.ndarray:
        """
        计算坡度大小（度）
        
        Returns:
            slope_magnitude: 坡度大小数组（度）
        """
        if self.dem_data is None or self.resolution is None:
            raise ValueError("请先加载DEM数据")
            
        # 计算X和Y方向的梯度
        self.dzdx, self.dzdy = np.gradient(self.dem_data, self.resolution)
        
        # 计算坡度（弧度）
        slope_rad = np.arctan(np.sqrt(self.dzdx**2 + self.dzdy**2))
        
        # 转换为度
        self.slope_magnitude = np.degrees(slope_rad)
        
        logger.info(f"计算坡度大小完成，范围: [{self.slope_magnitude.min():.2f}, {self.slope_magnitude.max():.2f}]度")
        return self.slope_magnitude
        
    def calculate_slope_aspect(self) -> np.ndarray:
        """
        计算坡向（度，北为0，顺时针）
        
        Returns:
            slope_aspect: 坡向数组（度）
        """
        if self.dzdx is None or self.dzdy is None:
            self.calculate_slope_magnitude()
            
        # 计算坡向（弧度）
        aspect_rad = np.arctan2(self.dzdx, self.dzdy)
        
        # 转换为度并调整为北为0
        self.slope_aspect = np.degrees(aspect_rad)
        self.slope_aspect = (450 - self.slope_aspect) % 360
        
        # 处理平坦区域（坡度接近0的区域）
        flat_mask = self.slope_magnitude < 0.1  # 坡度小于0.1度视为平地
        self.slope_aspect[flat_mask] = -1  # 平地的坡向设为-1
        
        logger.info("计算坡向完成")
        return self.slope_aspect
        
    def get_terrain_attributes(self, lon: float, lat: float) -> Dict[str, float]:
        """
        获取指定位置的地形属性
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            Dict[str, float]: 地形属性字典，包含：
                - slope_magnitude: 坡度大小（度）
                - slope_aspect: 坡向（度）
        """
        if self.dem_data is None:
            return {
                'slope_magnitude': 0.0,
                'slope_aspect': -1.0
            }
        
        # 获取像素坐标
        row = int((lat - 39.0) / (30 / 111000))  # 30米分辨率，1度约等于111公里
        col = int((lon - 116.0) / (30 / (111000 * np.cos(np.radians(lat)))))
        
        # 检查坐标是否在范围内
        if not (0 <= row < self.dem_data.shape[0] and 0 <= col < self.dem_data.shape[1]):
            return {
                'slope_magnitude': 0.0,
                'slope_aspect': -1.0
            }
        
        slope_mag = self.slope_magnitude[row, col] if self.slope_magnitude is not None else 0.0
        slope_asp = self.slope_aspect[row, col] if self.slope_aspect is not None else -1.0
        
        return {
            'slope_magnitude': float(slope_mag),
            'slope_aspect': float(slope_asp)
        }
        
    def calculate_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算X和Y方向的地形梯度
        
        Returns:
            (dzdx, dzdy): X和Y方向的梯度数组
        """
        if self.dem_data is None or self.resolution is None:
            raise ValueError("未加载DEM数据")
            
        # 使用numpy.gradient计算梯度
        dy, dx = np.gradient(self.dem_data)
        self.dzdx = dx / self.resolution  # X方向梯度
        self.dzdy = dy / self.resolution  # Y方向梯度
        
        logger.info("完成地形梯度计算")
        return self.dzdx, self.dzdy
        
    def save_results(self, output_dir: Union[str, Path]):
        """
        保存计算结果
        
        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存坡度大小
        if self.slope_magnitude is not None:
            slope_path = output_dir / "slope_magnitude_30m_100km.tif"
            self._save_array(self.slope_magnitude, slope_path, "坡度大小")
            
        # 保存坡向
        if self.slope_aspect is not None:
            aspect_path = output_dir / "slope_aspect_30m_100km.tif"
            self._save_array(self.slope_aspect, aspect_path, "坡向")
            
        # 保存梯度
        if self.dzdx is not None and self.dzdy is not None:
            dzdx_path = output_dir / "dzdx_30m_100km.tif"
            dzdy_path = output_dir / "dzdy_30m_100km.tif"
            self._save_array(self.dzdx, dzdx_path, "X方向梯度")
            self._save_array(self.dzdy, dzdy_path, "Y方向梯度")
            
    def _save_array(self, array: np.ndarray, path: Path, description: str):
        """
        保存数组为GeoTIFF文件
        
        Args:
            array: 要保存的数组
            path: 保存路径
            description: 数据描述
        """
        try:
            with rasterio.open(path, 'w',
                             driver='GTiff',
                             height=array.shape[0],
                             width=array.shape[1],
                             count=1,
                             dtype=array.dtype,
                             crs='+proj=latlong',
                             transform=None) as dst:
                dst.write(array, 1)
            logger.info(f"保存{description}数据到: {path}")
        except Exception as e:
            logger.error(f"保存{description}数据失败: {str(e)}")
            raise 