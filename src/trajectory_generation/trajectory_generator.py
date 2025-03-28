"""
轨迹生成器
实现轨迹插值和速度规划功能
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist

from ..data_processing import TerrainLoader, EnvironmentMapper

logger = logging.getLogger(__name__)

class TrajectoryGenerator:
    """轨迹生成器基类"""
    
    def __init__(
            self,
            terrain_loader: TerrainLoader,
            environment_mapper: EnvironmentMapper,
            config: Dict
        ):
        """
        初始化轨迹生成器
        
        Args:
            terrain_loader: 地形数据加载器实例
            environment_mapper: 环境地图生成器实例
            config: 配置参数字典，包含：
                - dt: 时间步长（秒）
                - MAX_ACCELERATION: 最大加速度（米/秒²）
                - MAX_DECELERATION: 最大减速度（米/秒²）
                - MAX_SPEED: 最大速度（米/秒）
                - MIN_SPEED: 最小速度（米/秒）
        """
        self.terrain_loader = terrain_loader
        self.environment_mapper = environment_mapper
        self.config = config
        
        # 获取环境地图
        self.maps = environment_mapper.get_maps()
        
        # 验证配置参数
        self._validate_config()
    
    def _validate_config(self) -> None:
        """验证配置参数"""
        required_params = [
            'dt',
            'MAX_ACCELERATION',
            'MAX_DECELERATION',
            'MAX_SPEED',
            'MIN_SPEED'
        ]
        
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"缺少必要的配置参数: {param}")
    
    def generate_trajectory(
            self,
            path_points: List[Tuple[int, int]]
        ) -> Dict:
        """
        生成轨迹
        
        Args:
            path_points: 路径点列表，每个点为 (row, col)
            
        Returns:
            Dict: 轨迹数据，包含：
                - timestamps: 时间戳列表（秒）
                - positions: 位置列表，每个元素为 (row, col)
                - speeds: 速度列表（米/秒）
                - headings: 朝向列表（度）
                - coordinates: 地理坐标列表，每个元素为 (lon, lat)
        """
        # 1. 路径插值
        interpolated_path = self._interpolate_path(path_points)
        
        # 2. 速度规划
        speeds = self._plan_speeds(interpolated_path)
        
        # 3. 时间规划
        timestamps = self._plan_timestamps(interpolated_path, speeds)
        
        # 4. 计算朝向
        headings = self._calculate_headings(interpolated_path)
        
        # 5. 转换为地理坐标
        coordinates = [
            self.terrain_loader.transform_pixel_to_coord(row, col)
            for row, col in interpolated_path
        ]
        
        return {
            'timestamps': timestamps,
            'positions': interpolated_path,
            'speeds': speeds,
            'headings': headings,
            'coordinates': coordinates
        }
    
    def _interpolate_path(
            self,
            path_points: List[Tuple[int, int]]
        ) -> List[Tuple[int, int]]:
        """
        使用三次样条插值平滑路径
        
        Args:
            path_points: 原始路径点列表
            
        Returns:
            List[Tuple[int, int]]: 插值后的路径点列表
        """
        if len(path_points) < 2:
            return path_points
        
        # 转换为数组
        points = np.array(path_points)
        
        # 计算路径长度参数
        t = np.zeros(len(points))
        for i in range(1, len(points)):
            t[i] = t[i-1] + np.sqrt(
                np.sum((points[i] - points[i-1])**2)
            )
        
        # 创建更密集的参数点
        num_points = int(t[-1] / self.config['dt'])
        t_new = np.linspace(0, t[-1], num_points)
        
        # 对行和列分别进行插值
        cs_row = CubicSpline(t, points[:, 0])
        cs_col = CubicSpline(t, points[:, 1])
        
        # 生成插值点
        rows = cs_row(t_new)
        cols = cs_col(t_new)
        
        # 转换为整数坐标
        interpolated = list(zip(
            np.round(rows).astype(int),
            np.round(cols).astype(int)
        ))
        
        # 去除重复点
        return list(dict.fromkeys(interpolated))
    
    def _plan_speeds(
            self,
            path_points: List[Tuple[int, int]]
        ) -> List[float]:
        """
        规划速度曲线
        
        Args:
            path_points: 插值后的路径点列表
            
        Returns:
            List[float]: 速度列表（米/秒）
        """
        if len(path_points) < 2:
            return [0.0] * len(path_points)
        
        # 获取每个点的最大允许速度
        max_speeds = [
            min(
                self.maps['max_speed_map'][row, col],
                self.config['MAX_SPEED']
            )
            for row, col in path_points
        ]
        
        # 初始化速度列表
        speeds = [0.0] * len(path_points)
        speeds[0] = self.config['MIN_SPEED']  # 起点速度
        
        # 前向传播：考虑加速度限制
        for i in range(1, len(path_points)):
            # 计算两点间距离
            dist = np.sqrt(
                sum((a - b)**2 for a, b in
                    zip(path_points[i], path_points[i-1]))
            ) * self.terrain_loader.resolution
            
            # 计算可能的最大速度（考虑加速度限制）
            v_prev = speeds[i-1]
            v_max_acc = np.sqrt(
                v_prev**2 +
                2 * self.config['MAX_ACCELERATION'] * dist
            )
            
            # 取较小值作为当前速度
            speeds[i] = min(v_max_acc, max_speeds[i])
        
        # 后向传播：考虑减速度限制
        for i in range(len(path_points)-2, -1, -1):
            # 计算两点间距离
            dist = np.sqrt(
                sum((a - b)**2 for a, b in
                    zip(path_points[i+1], path_points[i]))
            ) * self.terrain_loader.resolution
            
            # 计算为了安全减速需要的速度
            v_next = speeds[i+1]
            v_max_dec = np.sqrt(
                v_next**2 +
                2 * self.config['MAX_DECELERATION'] * dist
            )
            
            # 更新速度
            speeds[i] = min(speeds[i], v_max_dec)
        
        return speeds
    
    def _plan_timestamps(
            self,
            path_points: List[Tuple[int, int]],
            speeds: List[float]
        ) -> List[float]:
        """
        规划时间戳
        
        Args:
            path_points: 路径点列表
            speeds: 速度列表
            
        Returns:
            List[float]: 时间戳列表（秒）
        """
        timestamps = [0.0]  # 起点时间戳
        
        for i in range(1, len(path_points)):
            # 计算两点间距离
            dist = np.sqrt(
                sum((a - b)**2 for a, b in
                    zip(path_points[i], path_points[i-1]))
            ) * self.terrain_loader.resolution
            
            # 使用平均速度计算时间增量
            avg_speed = (speeds[i] + speeds[i-1]) / 2
            dt = dist / max(avg_speed, self.config['MIN_SPEED'])
            
            # 添加时间戳
            timestamps.append(timestamps[-1] + dt)
        
        return timestamps
    
    def _calculate_headings(
            self,
            path_points: List[Tuple[int, int]]
        ) -> List[float]:
        """
        计算路径点的朝向角度
        
        Args:
            path_points: 路径点列表
            
        Returns:
            List[float]: 朝向角度列表（度，北为0，顺时针为正）
        """
        if len(path_points) < 2:
            return [0.0] * len(path_points)
        
        headings = []
        
        # 计算第一个点的朝向（使用下一个点）
        dx = path_points[1][1] - path_points[0][1]
        dy = path_points[1][0] - path_points[0][0]
        heading = np.degrees(np.arctan2(dx, -dy)) % 360
        headings.append(heading)
        
        # 计算中间点的朝向（使用前后点的平均）
        for i in range(1, len(path_points)-1):
            dx = path_points[i+1][1] - path_points[i-1][1]
            dy = path_points[i+1][0] - path_points[i-1][0]
            heading = np.degrees(np.arctan2(dx, -dy)) % 360
            headings.append(heading)
        
        # 计算最后一个点的朝向（使用前一个点）
        dx = path_points[-1][1] - path_points[-2][1]
        dy = path_points[-1][0] - path_points[-2][0]
        heading = np.degrees(np.arctan2(dx, -dy)) % 360
        headings.append(heading)
        
        return headings 