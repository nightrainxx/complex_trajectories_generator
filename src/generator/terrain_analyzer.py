"""地形分析器模块

用于处理DEM数据，计算坡度、坡向和梯度等地形属性。

输入：
    - DEM数据（GeoTIFF格式）

输出：
    - 坡度大小图（度）
    - 坡向图（度，北为0，顺时针）
    - X方向梯度图
    - Y方向梯度图
"""

import numpy as np
import rasterio
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TerrainAnalyzer:
    """地形分析器类
    
    用于从DEM数据计算和保存地形属性。
    
    属性:
        dem_path (Path): DEM文件路径
        dem_data (np.ndarray): DEM数据数组
        transform (affine.Affine): 地理变换矩阵
        nodata (float): 无数据值
    """
    
    def __init__(self, dem_path: str):
        """初始化地形分析器
        
        Args:
            dem_path: DEM文件路径
            
        Raises:
            FileNotFoundError: 如果DEM文件不存在
            rasterio.errors.RasterioIOError: 如果DEM文件无法读取
        """
        self.dem_path = Path(dem_path)
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM文件不存在: {dem_path}")
        
        try:
            with rasterio.open(dem_path) as src:
                self.dem_data = src.read(1)
                self.transform = src.transform
                self.nodata = src.nodata
                self.crs = src.crs
                self.meta = src.meta
        except rasterio.errors.RasterioIOError as e:
            logger.error(f"无法读取DEM文件: {e}")
            raise
        
        logger.info(f"成功加载DEM数据，形状: {self.dem_data.shape}")
    
    def calculate_slope_magnitude(self) -> np.ndarray:
        """计算坡度大小
        
        使用numpy计算坡度，结果以度为单位。
        
        Returns:
            np.ndarray: 坡度数组，单位为度
        """
        # 获取像元大小
        pixel_width = abs(self.transform[0])
        pixel_height = abs(self.transform[4])
        
        # 计算梯度
        dy, dx = np.gradient(self.dem_data)
        
        # 转换为实际梯度（考虑像元大小）
        dzdx = dx / pixel_width
        dzdy = dy / pixel_height
        
        # 计算坡度（度）
        slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
        
        # 处理无效值
        slope = np.where(np.isnan(slope), 0, slope)
        slope = np.clip(slope, 0, 90)  # 限制在有效范围内
        
        logger.debug(f"坡度计算完成，范围: [{np.min(slope):.2f}, {np.max(slope):.2f}]度")
        return slope
    
    def calculate_slope_aspect(self) -> np.ndarray:
        """计算坡向
        
        使用numpy计算坡向，结果以度为单位（北为0，顺时针）。
        平坦区域的坡向设为-1。
        
        Returns:
            np.ndarray: 坡向数组，单位为度
        """
        # 获取像元大小
        pixel_width = abs(self.transform[0])
        pixel_height = abs(self.transform[4])
        
        # 计算梯度
        dy, dx = np.gradient(self.dem_data)
        
        # 转换为实际梯度（考虑像元大小）
        dzdx = dx / pixel_width
        dzdy = dy / pixel_height
        
        # 计算坡向（度）
        aspect = np.degrees(np.arctan2(-dzdx, -dzdy))  # 北为0，顺时针
        
        # 转换到0-360度范围
        aspect = np.where(aspect < 0, aspect + 360, aspect)
        
        # 处理平坦区域
        slope = self.calculate_slope_magnitude()
        aspect = np.where(slope < 0.1, -1, aspect)
        
        logger.debug("坡向计算完成")
        return aspect
    
    def calculate_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        """计算X和Y方向的梯度
        
        使用numpy的梯度函数计算。
        
        Returns:
            tuple[np.ndarray, np.ndarray]: (X方向梯度, Y方向梯度)
        """
        # 获取像元大小
        pixel_width = abs(self.transform[0])
        pixel_height = abs(self.transform[4])
        
        # 计算梯度
        dy, dx = np.gradient(self.dem_data)
        
        # 转换为实际梯度（考虑像元大小）
        dzdx = dx / pixel_width
        dzdy = dy / pixel_height
        
        logger.debug("梯度计算完成")
        return dzdx, dzdy
    
    def save_terrain_attributes(
        self,
        slope_mag_path: str,
        aspect_path: str,
        dzdx_path: str,
        dzdy_path: str
    ) -> None:
        """保存地形属性到文件
        
        Args:
            slope_mag_path: 坡度图保存路径
            aspect_path: 坡向图保存路径
            dzdx_path: X方向梯度图保存路径
            dzdy_path: Y方向梯度图保存路径
        """
        # 计算地形属性
        slope_mag = self.calculate_slope_magnitude()
        aspect = self.calculate_slope_aspect()
        dzdx, dzdy = self.calculate_gradients()
        
        # 准备元数据
        meta = self.meta.copy()
        
        # 保存坡度图
        self._save_raster(slope_mag_path, slope_mag, meta)
        logger.info(f"坡度图已保存: {slope_mag_path}")
        
        # 保存坡向图
        self._save_raster(aspect_path, aspect, meta)
        logger.info(f"坡向图已保存: {aspect_path}")
        
        # 保存梯度图
        self._save_raster(dzdx_path, dzdx, meta)
        logger.info(f"X方向梯度图已保存: {dzdx_path}")
        
        self._save_raster(dzdy_path, dzdy, meta)
        logger.info(f"Y方向梯度图已保存: {dzdy_path}")
    
    def _save_raster(self, path: str, data: np.ndarray, meta: dict) -> None:
        """保存栅格数据到文件
        
        Args:
            path: 保存路径
            data: 要保存的数据数组
            meta: 元数据字典
        """
        # 更新元数据
        meta.update({
            'dtype': data.dtype,
            'count': 1
        })
        
        # 保存文件
        with rasterio.open(path, 'w', **meta) as dst:
            dst.write(data, 1) 