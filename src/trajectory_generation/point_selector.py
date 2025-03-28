"""
起终点选择器
用于批量选择满足约束条件的起终点对
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from ..data_processing import TerrainLoader

logger = logging.getLogger(__name__)

class PointSelector:
    """起终点选择器"""
    
    def __init__(
            self,
            terrain_loader: TerrainLoader,
            config: Dict
        ):
        """
        初始化起终点选择器
        
        Args:
            terrain_loader: 地形数据加载器实例
            config: 配置参数字典，包含：
                - NUM_TRAJECTORIES_TO_GENERATE: 要生成的轨迹总数
                - NUM_END_POINTS: 要选择的固定终点数量
                - MIN_START_END_DISTANCE_METERS: 起终点最小直线距离(米)
                - URBAN_LANDCOVER_CODES: 代表城市/建成区的地物编码列表
                - IMPASSABLE_LANDCOVER_CODES: 代表不可通行的地物编码列表
        """
        self.terrain_loader = terrain_loader
        self.config = config
        
        # 验证配置参数
        self._validate_config()
        
        # 初始化结果列表
        self.selected_end_points = []  # [(row, col), ...]
        self.generation_pairs = []     # [(start_point, end_point), ...]
    
    def _validate_config(self) -> None:
        """验证配置参数"""
        required_params = [
            'NUM_TRAJECTORIES_TO_GENERATE',
            'NUM_END_POINTS',
            'MIN_START_END_DISTANCE_METERS',
            'URBAN_LANDCOVER_CODES',
            'IMPASSABLE_LANDCOVER_CODES'
        ]
        
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"缺少必要的配置参数: {param}")
    
    def select_points(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        选择满足约束条件的起终点对
        
        Returns:
            List[Tuple[Tuple[float, float], Tuple[float, float]]]:
            起终点对列表，每个元素为((start_lon, start_lat), (end_lon, end_lat))
        """
        logger.info("开始选择起终点对...")
        
        # 选择终点
        self._select_end_points()
        if not self.selected_end_points:
            raise RuntimeError("未能找到合适的终点")
        
        # 为每个终点选择起点
        trajectories_per_end = self.config['NUM_TRAJECTORIES_TO_GENERATE'] // len(self.selected_end_points)
        remaining = self.config['NUM_TRAJECTORIES_TO_GENERATE'] % len(self.selected_end_points)
        
        for i, end_point in enumerate(self.selected_end_points):
            # 确定当前终点需要的起点数量
            n_starts = trajectories_per_end + (1 if i < remaining else 0)
            
            # 选择起点
            start_points = self._select_start_points(end_point, n_starts)
            
            # 转换为地理坐标
            end_coord = self.terrain_loader.transform_pixel_to_coord(*end_point)
            start_coords = [
                self.terrain_loader.transform_pixel_to_coord(*p)
                for p in start_points
            ]
            
            # 添加到生成对列表
            for start_coord in start_coords:
                self.generation_pairs.append((start_coord, end_coord))
        
        logger.info(f"已选择 {len(self.generation_pairs)} 对起终点")
        return self.generation_pairs
    
    def _select_end_points(self) -> None:
        """选择合适的终点"""
        # 获取城市区域掩码
        urban_mask = np.isin(
            self.terrain_loader.landcover_data,
            self.config['URBAN_LANDCOVER_CODES']
        )
        
        # 获取可通行区域掩码
        passable_mask = ~np.isin(
            self.terrain_loader.landcover_data,
            self.config['IMPASSABLE_LANDCOVER_CODES']
        )
        
        # 获取有效终点候选区域
        valid_mask = urban_mask & passable_mask
        valid_points = np.argwhere(valid_mask)
        
        if len(valid_points) < self.config['NUM_END_POINTS']:
            raise RuntimeError(
                f"可用的终点候选数量({len(valid_points)})少于需要的数量"
                f"({self.config['NUM_END_POINTS']})"
            )
        
        # 随机选择终点
        selected_indices = np.random.choice(
            len(valid_points),
            size=self.config['NUM_END_POINTS'],
            replace=False
        )
        
        self.selected_end_points = [
            tuple(valid_points[i])
            for i in selected_indices
        ]
        
        logger.info(f"已选择 {len(self.selected_end_points)} 个终点")
    
    def _select_start_points(
            self,
            end_point: Tuple[int, int],
            n_points: int
        ) -> List[Tuple[int, int]]:
        """
        为指定终点选择合适的起点
        
        Args:
            end_point: 终点像素坐标 (row, col)
            n_points: 需要选择的起点数量
            
        Returns:
            List[Tuple[int, int]]: 起点列表
        """
        # 获取不可通行区域掩码
        impassable_mask = np.isin(
            self.terrain_loader.landcover_data,
            self.config['IMPASSABLE_LANDCOVER_CODES']
        )
        
        # 计算与终点的距离（米）
        rows, cols = np.indices(self.terrain_loader.dem_data.shape)
        distances = self._calculate_distances(
            rows, cols,
            end_point[0], end_point[1]
        )
        
        # 创建有效起点掩码
        valid_mask = (
            ~impassable_mask &  # 可通行
            (distances >= self.config['MIN_START_END_DISTANCE_METERS'])  # 满足最小距离
        )
        
        valid_points = np.argwhere(valid_mask)
        if len(valid_points) < n_points:
            raise RuntimeError(
                f"可用的起点候选数量({len(valid_points)})少于需要的数量({n_points})"
            )
        
        # 随机选择起点
        selected_indices = np.random.choice(
            len(valid_points),
            size=n_points,
            replace=False
        )
        
        return [
            tuple(valid_points[i])
            for i in selected_indices
        ]
    
    def _calculate_distances(
            self,
            rows: np.ndarray,
            cols: np.ndarray,
            end_row: int,
            end_col: int
        ) -> np.ndarray:
        """
        计算栅格中所有点到终点的距离（米）
        
        Args:
            rows: 行索引数组
            cols: 列索引数组
            end_row: 终点行索引
            end_col: 终点列索引
            
        Returns:
            np.ndarray: 距离数组（米）
        """
        # 将像素距离转换为地理距离（米）
        pixel_distances = np.sqrt(
            (rows - end_row)**2 +
            (cols - end_col)**2
        )
        
        return pixel_distances * self.terrain_loader.resolution
    
    def visualize_points(self, output_file: Optional[str] = None) -> None:
        """
        可视化选择的起终点
        
        Args:
            output_file: 输出文件路径，可选
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # 绘制地形
        plt.imshow(self.terrain_loader.dem_data, cmap='terrain')
        plt.colorbar(label='Elevation (m)')
        
        # 绘制终点
        end_points = np.array(self.selected_end_points)
        plt.scatter(
            end_points[:, 1],
            end_points[:, 0],
            c='red',
            marker='^',
            s=100,
            label='End Points'
        )
        
        # 绘制起点
        start_points = []
        for start_coord, _ in self.generation_pairs:
            row, col = self.terrain_loader.transform_coordinates(
                start_coord[0],
                start_coord[1]
            )
            start_points.append((row, col))
        
        start_points = np.array(start_points)
        plt.scatter(
            start_points[:, 1],
            start_points[:, 0],
            c='blue',
            marker='o',
            s=50,
            label='Start Points'
        )
        
        plt.title('Selected Start and End Points')
        plt.legend()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"已保存可视化结果到: {output_file}")
        else:
            plt.show()
        
        plt.close() 