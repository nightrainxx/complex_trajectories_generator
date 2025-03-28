"""
环境地图生成器
根据地形数据和学习结果构建增强的环境地图
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio
from rasterio.transform import Affine

from .terrain_analyzer import TerrainAnalyzer
from .terrain_loader import TerrainLoader

logger = logging.getLogger(__name__)

class EnvironmentMapper:
    """环境地图生成器"""
    
    def __init__(
            self,
            terrain_loader: TerrainLoader,
            motion_patterns: Dict,
            output_dir: str = "data/output/intermediate"
        ):
        """
        初始化环境地图生成器
        
        Args:
            terrain_loader: 地形数据加载器实例
            motion_patterns: 学习到的运动模式
            output_dir: 输出目录
        """
        self.terrain_loader = terrain_loader
        self.terrain_analyzer = TerrainAnalyzer(terrain_loader)
        self.motion_patterns = motion_patterns
        self.output_dir = Path(output_dir)
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化地图数组
        shape = terrain_loader.dem_data.shape
        self.max_speed_map = np.zeros(shape, dtype=np.float32)
        self.typical_speed_map = np.zeros(shape, dtype=np.float32)
        self.speed_stddev_map = np.zeros(shape, dtype=np.float32)
        self.cost_map = np.zeros(shape, dtype=np.float32)
    
    def generate_maps(self) -> None:
        """生成所有环境地图"""
        logger.info("开始生成环境地图...")
        
        # 获取地形属性
        rows, cols = self.terrain_loader.dem_data.shape
        for row in range(rows):
            for col in range(cols):
                # 获取地理坐标
                lon, lat = self.terrain_loader.transform_pixel_to_coord(row, col)
                
                # 获取地形属性
                terrain_attrs = self.terrain_analyzer.get_terrain_attributes(lon, lat)
                landcover = self.terrain_loader.get_landcover(lon, lat)
                
                # 计算速度特征
                max_s, typ_s, std_s = self._calculate_speed_features(
                    terrain_attrs['slope_magnitude'],
                    landcover
                )
                
                # 更新地图
                self.max_speed_map[row, col] = max_s
                self.typical_speed_map[row, col] = typ_s
                self.speed_stddev_map[row, col] = std_s
                
                # 计算成本（基于典型速度）
                if typ_s > 0:
                    self.cost_map[row, col] = self.terrain_loader.resolution / typ_s
                else:
                    self.cost_map[row, col] = np.inf
        
        logger.info("环境地图生成完成")
        
        # 保存地图
        self._save_maps()
    
    def _calculate_speed_features(
            self,
            slope_magnitude: float,
            landcover: int
        ) -> tuple[float, float, float]:
        """
        计算给定位置的速度特征
        
        Args:
            slope_magnitude: 坡度大小(度)
            landcover: 地表类型编码
            
        Returns:
            tuple: (最大速度, 典型速度, 速度标准差)
        """
        # 获取坡度速度模型
        slope_speed = self.motion_patterns['slope_speed_model']
        
        # 找到对应的坡度组
        for (lower, upper), row in slope_speed.iterrows():
            if lower <= slope_magnitude < upper:
                slope_factor = row['speed_factor']
                break
        else:
            slope_factor = slope_speed.iloc[-1]['speed_factor']  # 使用最陡坡度组的因子
        
        # 获取地表类型速度统计
        landcover_stats = self.motion_patterns['landcover_speed_stats']
        if landcover in landcover_stats.index:
            lc_stats = landcover_stats.loc[landcover]
            lc_factor = lc_stats['speed_factor']
            speed_std = lc_stats['std']
        else:
            lc_factor = 0.5  # 默认因子
            speed_std = landcover_stats['std'].mean()  # 使用平均标准差
        
        # 计算速度特征
        base_speed = 20.0  # 基准速度 (m/s)
        max_speed = base_speed * slope_factor * lc_factor
        typical_speed = max_speed * 0.8  # 典型速度略低于最大速度
        
        return max_speed, typical_speed, speed_std
    
    def _save_maps(self) -> None:
        """保存生成的地图"""
        # 准备元数据
        meta = self.terrain_loader.get_raster_meta()
        
        # 保存最大速度图
        self._save_raster(
            self.max_speed_map,
            self.output_dir / "max_speed_map.tif",
            meta,
            "最大速度图 (m/s)"
        )
        
        # 保存典型速度图
        self._save_raster(
            self.typical_speed_map,
            self.output_dir / "typical_speed_map.tif",
            meta,
            "典型速度图 (m/s)"
        )
        
        # 保存速度标准差图
        self._save_raster(
            self.speed_stddev_map,
            self.output_dir / "speed_stddev_map.tif",
            meta,
            "速度标准差图 (m/s)"
        )
        
        # 保存成本图
        self._save_raster(
            self.cost_map,
            self.output_dir / "cost_map.tif",
            meta,
            "成本图 (s/m)"
        )
        
        logger.info("已保存所有环境地图")
    
    def _save_raster(
            self,
            data: np.ndarray,
            filepath: Path,
            meta: Dict,
            description: str
        ) -> None:
        """
        保存栅格数据
        
        Args:
            data: 栅格数据
            filepath: 保存路径
            meta: 元数据
            description: 数据描述
        """
        meta = meta.copy()
        meta.update({
            'dtype': 'float32',
            'description': description
        })
        
        with rasterio.open(filepath, 'w', **meta) as dst:
            dst.write(data, 1)
        logger.info(f"已保存 {description} 到: {filepath}")
    
    def get_maps(self) -> Dict[str, np.ndarray]:
        """
        获取生成的地图
        
        Returns:
            Dict: 包含所有生成的地图
        """
        return {
            'max_speed_map': self.max_speed_map,
            'typical_speed_map': self.typical_speed_map,
            'speed_stddev_map': self.speed_stddev_map,
            'cost_map': self.cost_map
        } 