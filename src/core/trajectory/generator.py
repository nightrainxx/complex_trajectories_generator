"""
轨迹生成器模块
整合路径规划和运动模拟，生成完整轨迹
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

from src.core.path.planner import PathPlanner
from src.core.motion.simulator import MotionSimulator, EnvironmentMaps, TrajectoryPoint
from src.core.terrain.loader import TerrainLoader
from src.utils.config import config

logger = logging.getLogger(__name__)

class TrajectoryGenerator:
    """轨迹生成器"""
    
    def __init__(
            self,
            terrain_loader: TerrainLoader,
            config: Dict
        ):
        """
        初始化轨迹生成器
        
        Args:
            terrain_loader: 地形加载器实例
            config: 配置参数字典
        """
        self.terrain_loader = terrain_loader
        self.config = config
        
        # 生成成本图
        self.cost_map = self._generate_cost_map()
        
        # 初始化路径规划器
        self.path_planner = PathPlanner(self.cost_map)
        
        # 初始化运动模拟器
        self.motion_simulator = MotionSimulator(
            env_maps=self._prepare_environment_maps(),
            terrain_loader=terrain_loader,
            config=config
        )
        
    def _generate_cost_map(self) -> np.ndarray:
        """
        生成成本图
        
        Returns:
            np.ndarray: 成本图
        """
        # 获取地图尺寸
        height, width = self.terrain_loader.dem_data.shape
        cost_map = np.ones((height, width), dtype=np.float32)
        
        # 设置不可通行区域的成本为无穷大
        for row in range(height):
            for col in range(width):
                if not self.terrain_loader.is_passable(row, col):
                    cost_map[row, col] = float('inf')
                    continue
                    
                # 获取地形属性
                attrs = self.terrain_loader.get_terrain_attributes(row, col)
                
                # 基础成本（与典型速度成反比）
                if 'typical_speed' in attrs:
                    cost_map[row, col] = 1.0 / max(
                        attrs['typical_speed'],
                        self.config['motion']['MIN_SPEED']
                    )
                    
                # 坡度影响
                if 'slope' in attrs:
                    slope_factor = 1.0 + (
                        attrs['slope'] / 
                        self.config['motion']['MAX_SLOPE_DEGREES']
                    )
                    cost_map[row, col] *= slope_factor
                    
        return cost_map
        
    def _prepare_environment_maps(self) -> EnvironmentMaps:
        """
        准备环境地图集合
        
        Returns:
            EnvironmentMaps: 环境地图集合
        """
        # 获取地图尺寸
        height, width = self.terrain_loader.dem_data.shape
        
        # 创建典型速度图（基于坡度和土地覆盖类型）
        typical_speed = np.full(
            (height, width),
            self.config['motion']['DEFAULT_SPEED']
        )
        
        # 创建最大速度图
        max_speed = np.full(
            (height, width),
            self.config['motion']['MAX_SPEED']
        )
        
        # 创建速度标准差图（可以基于地形复杂度调整）
        speed_stddev = np.full(
            (height, width),
            self.config['motion']['SPEED_STDDEV']
        )
        
        # 对每个像素计算实际速度限制
        for row in range(height):
            for col in range(width):
                attrs = self.terrain_loader.get_terrain_attributes(row, col)
                
                # 坡度影响
                if 'slope' in attrs:
                    slope_factor = 1.0 - (
                        attrs['slope'] / 
                        self.config['motion']['MAX_SLOPE_DEGREES']
                    )
                    typical_speed[row, col] *= max(0.1, slope_factor)
                    max_speed[row, col] *= max(0.1, slope_factor)
                    
                # 土地覆盖类型影响
                if ('landcover' in attrs and 
                    attrs['landcover'] in self.config['terrain']['SPEED_FACTORS']):
                    landcover_factor = self.config['terrain']['SPEED_FACTORS'][
                        attrs['landcover']
                    ]
                    typical_speed[row, col] *= landcover_factor
                    max_speed[row, col] *= landcover_factor
                    
                # 确保速度不低于最小值
                typical_speed[row, col] = max(
                    typical_speed[row, col],
                    self.config['motion']['MIN_SPEED']
                )
                max_speed[row, col] = max(
                    max_speed[row, col],
                    self.config['motion']['MIN_SPEED']
                )
                
        return EnvironmentMaps(
            typical_speed=typical_speed,
            max_speed=max_speed,
            speed_stddev=speed_stddev,
            slope_magnitude=self.terrain_loader.slope_data,
            slope_aspect=self.terrain_loader.aspect_data,
            landcover=self.terrain_loader.landcover_data
        )
        
    def generate_trajectory(
            self,
            start_point: Tuple[float, float],
            end_point: Tuple[float, float],
            output_file: Optional[Path] = None
        ) -> List[TrajectoryPoint]:
        """
        生成轨迹
        
        Args:
            start_point: 起点坐标 (经度, 纬度)
            end_point: 终点坐标 (经度, 纬度)
            output_file: 输出文件路径
            
        Returns:
            List[TrajectoryPoint]: 轨迹点列表
        """
        logger.info(f"开始生成轨迹: {start_point} -> {end_point}")
        
        # 将经纬度转换为UTM坐标
        start_utm = self.terrain_loader.lonlat_to_utm(*start_point)
        end_utm = self.terrain_loader.lonlat_to_utm(*end_point)
        
        # 将UTM坐标转换为像素坐标
        start_pixel = self.terrain_loader.coord_to_pixel(*start_utm)
        end_pixel = self.terrain_loader.coord_to_pixel(*end_utm)
        
        # 使用A*规划路径
        logger.info("使用A*算法规划路径...")
        path = self.path_planner.find_path(start_pixel, end_pixel)
        
        # 使用运动模拟器生成轨迹
        logger.info("使用运动模拟器生成轨迹...")
        trajectory = self.motion_simulator.simulate_motion(path)
        
        # 保存轨迹
        if output_file:
            self._save_trajectory(trajectory, output_file)
            
        return trajectory
        
    def _save_trajectory(
            self,
            trajectory: List[TrajectoryPoint],
            output_file: Path
        ) -> None:
        """
        保存轨迹到文件
        
        Args:
            trajectory: 轨迹点列表
            output_file: 输出文件路径
        """
        # 创建输出目录
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换轨迹点为字典列表
        trajectory_data = []
        for point in trajectory:
            # 将像素坐标转换为UTM坐标
            utm_coord = self.terrain_loader.pixel_to_coord(
                point.row, point.col
            )
            # 将UTM坐标转换为经纬度
            lon, lat = self.terrain_loader.utm_to_lonlat(*utm_coord)
            
            trajectory_data.append({
                'timestamp': point.timestamp,
                'longitude': lon,
                'latitude': lat,
                'speed': point.speed,
                'heading': point.heading
            })
            
        # 保存为JSON文件
        with open(output_file, 'w') as f:
            json.dump({
                'trajectory': trajectory_data,
                'metadata': {
                    'total_time': trajectory[-1].timestamp,
                    'total_distance': sum(
                        point.speed * 0.1  # dt = 0.1
                        for point in trajectory[:-1]
                    ),
                    'average_speed': np.mean([
                        point.speed for point in trajectory
                    ])
                }
            }, f, indent=2) 