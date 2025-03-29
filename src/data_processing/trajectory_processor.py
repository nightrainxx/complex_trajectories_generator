"""
轨迹数据预处理模块
"""
import os
import numpy as np
import pandas as pd
from typing import Optional
import rasterio
from rasterio.transform import from_origin
import logging
from pyproj import Transformer

logger = logging.getLogger(__name__)

class TrajectoryProcessor:
    """轨迹数据预处理器"""
    
    def __init__(self, dem_file: str, landcover_file: str):
        """初始化预处理器
        
        Args:
            dem_file: DEM数据文件路径
            landcover_file: 土地覆盖数据文件路径
        """
        self.dem_file = dem_file
        self.landcover_file = landcover_file
        
        # 加载地形数据
        with rasterio.open(dem_file) as src:
            self.dem_data = src.read(1)
            self.dem_transform = src.transform
            self.dem_nodata = src.nodata
            
        # 加载土地覆盖数据
        with rasterio.open(landcover_file) as src:
            self.landcover_data = src.read(1)
            self.landcover_transform = src.transform
            self.landcover_nodata = src.nodata
            
        # 计算坡度和坡向
        self.calculate_slope_aspect()
        
    def calculate_slope_aspect(self):
        """计算坡度和坡向"""
        # 计算像素大小（米）
        dx = self.dem_transform[0] * 111000  # 近似转换为米
        dy = -self.dem_transform[4] * 111000
        
        # 计算x和y方向的梯度
        gy, gx = np.gradient(self.dem_data, dy, dx)
        
        # 计算坡度（度）
        self.slope_magnitude = np.degrees(np.arctan(np.sqrt(gx*gx + gy*gy)))
        
        # 计算坡向（度，北为0，顺时针）
        self.slope_aspect = np.degrees(np.arctan2(-gx, gy))
        self.slope_aspect = np.where(self.slope_aspect < 0, self.slope_aspect + 360, self.slope_aspect)
        
        # 处理平地和异常值
        flat_mask = self.slope_magnitude < 1.0
        self.slope_aspect[flat_mask] = -1
        self.slope_magnitude[self.slope_magnitude > 45.0] = 45.0  # 限制最大坡度
        
        logger.info(f"坡度范围: {self.slope_magnitude.min():.1f}° - {self.slope_magnitude.max():.1f}°")
        logger.info(f"平地占比: {np.mean(flat_mask) * 100:.1f}%")
        
    def get_pixel_value(self, lon: float, lat: float, data: np.ndarray, 
                       transform: rasterio.Affine) -> Optional[float]:
        """获取给定经纬度位置的栅格值
        
        Args:
            lon: 经度
            lat: 纬度
            data: 栅格数据
            transform: 栅格变换参数
            
        Returns:
            栅格值，如果位置无效则返回None
        """
        # 将经纬度转换为像素坐标
        col, row = ~transform * (lon, lat)
        col, row = int(col), int(row)
        
        # 检查坐标是否在有效范围内
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            return float(data[row, col])
        return None
        
    def add_environment_features(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """为轨迹添加环境特征
        
        Args:
            trajectory_df: 轨迹DataFrame
            
        Returns:
            添加了环境特征的DataFrame
        """
        # 计算heading_degrees（方位角）
        trajectory_df['heading_degrees'] = np.degrees(
            np.arctan2(
                trajectory_df['longitude'].diff(),
                trajectory_df['latitude'].diff()
            )
        )
        # 处理第一个点的heading
        trajectory_df.loc[trajectory_df.index[0], 'heading_degrees'] = \
            trajectory_df.loc[trajectory_df.index[1], 'heading_degrees']
        
        # 确保heading在0-360度范围内
        trajectory_df['heading_degrees'] = np.where(
            trajectory_df['heading_degrees'] < 0,
            trajectory_df['heading_degrees'] + 360,
            trajectory_df['heading_degrees']
        )
        
        # 添加环境特征
        for idx, row in trajectory_df.iterrows():
            # 获取坡度
            slope = self.get_pixel_value(
                row['longitude'], row['latitude'],
                self.slope_magnitude, self.dem_transform
            )
            trajectory_df.loc[idx, 'slope_magnitude'] = slope if slope is not None else 0.0
            
            # 获取坡向
            aspect = self.get_pixel_value(
                row['longitude'], row['latitude'],
                self.slope_aspect, self.dem_transform
            )
            trajectory_df.loc[idx, 'slope_aspect'] = aspect if aspect is not None else -1
            
            # 获取地物类型
            landcover = self.get_pixel_value(
                row['longitude'], row['latitude'],
                self.landcover_data, self.landcover_transform
            )
            trajectory_df.loc[idx, 'landcover'] = int(landcover) if landcover is not None else 0
            
        return trajectory_df
        
    def process_trajectory_file(self, input_file: str, output_file: str) -> None:
        """处理轨迹文件
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        # 加载轨迹数据
        logger.info(f"Loading trajectory from {input_file}")
        df = pd.read_csv(input_file)
        
        # 添加环境特征
        logger.info("Adding environment features...")
        df = self.add_environment_features(df)
        
        # 保存处理后的数据
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Processed trajectory saved to {output_file}")
        
    def process_trajectory(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """处理轨迹数据，添加环境特征
        
        Args:
            trajectory_df: 包含经纬度的轨迹数据DataFrame
            
        Returns:
            添加了环境特征的DataFrame
        """
        # 创建结果DataFrame
        result_df = pd.DataFrame()
        
        # 获取经纬度和航向
        lons = trajectory_df['longitude'].values
        lats = trajectory_df['latitude'].values
        headings = trajectory_df['heading_deg'].values  # 使用heading_deg而不是heading_degrees
        speeds = trajectory_df['velocity_2d_ms'].values  # 使用velocity_2d_ms作为速度
        
        # 获取DEM和土地覆盖数据
        landcover_values = []
        slope_magnitudes = []
        slope_aspects = []
        slope_along_paths = []
        cross_slopes = []
        
        for lon, lat, heading in zip(lons, lats, headings):
            # 获取土地覆盖类型
            landcover = self.get_pixel_value(
                lon, lat,
                self.landcover_data,
                self.landcover_transform
            )
            landcover_values.append(landcover if landcover is not None else 0)
            
            # 获取坡度和坡向
            slope_mag = self.get_pixel_value(
                lon, lat,
                self.slope_magnitude,
                self.dem_transform
            )
            slope_asp = self.get_pixel_value(
                lon, lat,
                self.slope_aspect,
                self.dem_transform
            )
            
            slope_magnitudes.append(slope_mag if slope_mag is not None else 0.0)
            slope_aspects.append(slope_asp if slope_asp is not None else -1)
            
            # 计算沿路径坡度和横坡
            if slope_mag is not None and slope_asp is not None and slope_asp != -1:
                # 计算坡向与航向的夹角
                aspect_diff = abs(slope_asp - heading)
                if aspect_diff > 180:
                    aspect_diff = 360 - aspect_diff
                    
                # 计算沿路径坡度（考虑上下坡）
                slope_along_path = slope_mag * np.cos(np.radians(aspect_diff))
                if abs(aspect_diff) > 90:
                    slope_along_path = -slope_along_path
                    
                # 计算横坡（始终为正值）
                cross_slope = abs(slope_mag * np.sin(np.radians(aspect_diff)))
            else:
                slope_along_path = 0.0
                cross_slope = 0.0
                
            slope_along_paths.append(slope_along_path)
            cross_slopes.append(cross_slope)
            
        # 构建结果DataFrame
        result_df['landcover'] = landcover_values
        result_df['slope_magnitude'] = slope_magnitudes
        result_df['slope_aspect'] = slope_aspects
        result_df['slope_along_path'] = slope_along_paths
        result_df['cross_slope'] = cross_slopes
        result_df['speed_mps'] = speeds  # 添加速度
        
        return result_df
        
    def calculate_distances(self, trajectory_df: pd.DataFrame) -> np.ndarray:
        """计算轨迹点之间的距离
        
        Args:
            trajectory_df: 包含经纬度的轨迹数据DataFrame
            
        Returns:
            相邻点之间的距离数组（米）
        """
        # 转换为UTM坐标
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
        x, y = transformer.transform(
            trajectory_df['longitude'].values,
            trajectory_df['latitude'].values
        )
        
        # 计算相邻点之间的距离
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        distances = np.sqrt(dx*dx + dy*dy)
        
        return distances 