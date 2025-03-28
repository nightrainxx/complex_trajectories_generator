"""
路径规划器模块
负责使用A*算法进行智能路径规划
"""

import heapq
import logging
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import rasterio
from pathlib import Path
from dataclasses import dataclass, field
from scipy.interpolate import splprep, splev

from ..data_processing import GISDataLoader
from src.generator.config import (
    LANDCOVER_COST_FACTORS,
    IMPASSABLE_LANDCOVER_CODES,
    MAX_SLOPE_THRESHOLD
)

# 配置日志
logger = logging.getLogger(__name__)

@dataclass(order=True)
class PathNode:
    """路径节点类，用于A*算法"""
    priority: float = field(compare=True)  # f = g + h
    position: Tuple[int, int] = field(compare=False)  # (row, col)
    g_cost: float = field(compare=False)  # 从起点到当前点的实际代价
    h_cost: float = field(compare=False)  # 从当前点到终点的估计代价
    parent: Optional['PathNode'] = field(compare=False, default=None)  # 父节点

class PathPlanner:
    """路径规划器类"""
    
    def __init__(
        self,
        cost_map_path: str,
        smoothness_weight: float = 0.3,
        heuristic_weight: float = 1.1,
        interpolation_points: int = 100
    ):
        """初始化路径规划器
        
        Args:
            cost_map_path: 成本地图文件路径
            smoothness_weight: 平滑度权重，控制路径的平滑程度
            heuristic_weight: 启发式权重，控制A*算法的搜索倾向
            interpolation_points: 插值点数量，控制平滑后路径的精度
        """
        # 检查文件是否存在
        if not Path(cost_map_path).exists():
            raise FileNotFoundError(f"找不到文件: {cost_map_path}")
        
        # 读取成本地图
        with rasterio.open(cost_map_path) as src:
            self.cost_map = src.read(1)
            self.transform = src.transform
            self.height = src.height
            self.width = src.width
        
        # 初始化参数
        self.smoothness_weight = smoothness_weight
        self.heuristic_weight = heuristic_weight
        self.interpolation_points = interpolation_points
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        
        # 定义8个方向的偏移量（上、右上、右、右下、下、左下、左、左上）
        self.directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        # 计算对角线方向的代价系数（√2）
        self.direction_costs = [
            1.0, np.sqrt(2), 1.0, np.sqrt(2),
            1.0, np.sqrt(2), 1.0, np.sqrt(2)
        ]
    
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """检查位置是否有效
        
        Args:
            position: 位置坐标 (row, col)
            
        Returns:
            bool: 位置是否有效
        """
        row, col = position
        return (0 <= row < self.height and 
                0 <= col < self.width and 
                self.cost_map[row, col] < float('inf'))
    
    def calculate_heuristic(
        self,
        position: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> float:
        """计算启发式值（使用欧几里得距离）
        
        Args:
            position: 当前位置
            goal: 目标位置
            
        Returns:
            float: 启发式值
        """
        return np.sqrt(
            (position[0] - goal[0])**2 +
            (position[1] - goal[1])**2
        )
    
    def calculate_turn_cost(
        self,
        current: Tuple[int, int],
        next_pos: Tuple[int, int],
        parent: Optional[Tuple[int, int]] = None
    ) -> float:
        """计算转弯代价
        
        Args:
            current: 当前位置
            next_pos: 下一个位置
            parent: 父节点位置
            
        Returns:
            float: 转弯代价
        """
        if parent is None:
            return 0.0
        
        # 计算两个方向向量
        v1 = np.array([current[0] - parent[0], current[1] - parent[1]], dtype=np.float64)
        v2 = np.array([next_pos[0] - current[0], next_pos[1] - current[1]], dtype=np.float64)
        
        # 计算向量的模
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        # 如果任一向量为零向量，说明路径重叠
        if n1 < 1e-6 or n2 < 1e-6:
            return self.smoothness_weight * np.pi
        
        # 计算夹角的余弦值
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # 将余弦值转换为角度（0-180度）
        angle = np.arccos(cos_angle)
        
        # 返回转弯代价（角度越大，代价越高）
        return angle * self.smoothness_weight
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """使用A*算法寻找最优路径
        
        Args:
            start: 起点坐标 (row, col)
            goal: 终点坐标 (row, col)
            
        Returns:
            List[Tuple[int, int]]: 路径点列表
        """
        # 检查起终点的有效性
        if not self.is_valid_position(start):
            raise ValueError(f"无效的起点坐标: {start}")
        if not self.is_valid_position(goal):
            raise ValueError(f"无效的终点坐标: {goal}")
        
        # 初始化开放列表和关闭列表
        open_list = []
        closed_set = set()
        
        # 创建起点节点
        start_node = PathNode(
            priority=0,
            position=start,
            g_cost=0,
            h_cost=self.calculate_heuristic(start, goal)
        )
        
        # 将起点加入开放列表
        heapq.heappush(open_list, start_node)
        
        # 开始搜索
        while open_list:
            # 获取f值最小的节点
            current = heapq.heappop(open_list)
            
            # 如果到达目标，构建并返回路径
            if current.position == goal:
                path = []
                while current:
                    path.append(current.position)
                    current = current.parent
                return list(reversed(path))
            
            # 将当前节点加入关闭列表
            closed_set.add(current.position)
            
            # 检查所有相邻节点
            for i, (dy, dx) in enumerate(self.directions):
                next_pos = (
                    current.position[0] + dy,
                    current.position[1] + dx
                )
                
                # 跳过无效或已访问的节点
                if (not self.is_valid_position(next_pos) or
                    next_pos in closed_set):
                    continue
                
                # 计算新的g值（考虑方向代价和转弯代价）
                new_g = (current.g_cost +
                        self.direction_costs[i] * self.cost_map[next_pos] +
                        self.calculate_turn_cost(
                            current.position,
                            next_pos,
                            current.parent.position if current.parent else None
                        ))
                
                # 计算h值
                h = self.calculate_heuristic(next_pos, goal)
                
                # 创建新节点
                neighbor = PathNode(
                    priority=new_g + self.heuristic_weight * h,
                    position=next_pos,
                    g_cost=new_g,
                    h_cost=h,
                    parent=current
                )
                
                # 将新节点加入开放列表
                heapq.heappush(open_list, neighbor)
        
        # 如果没有找到路径，返回空列表
        self.logger.warning(f"未能找到从{start}到{goal}的路径")
        return []
    
    def smooth_path(
        self,
        path: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """使用B样条插值平滑路径
        
        Args:
            path: 原始路径点列表
            
        Returns:
            List[Tuple[int, int]]: 平滑后的路径点列表
        """
        if len(path) < 3:
            return path
        
        # 提取路径点的坐标
        points = np.array(path)
        x = points[:, 0]
        y = points[:, 1]
        
        # 计算路径点之间的累积距离作为参数化变量
        t = np.zeros(len(path))
        for i in range(1, len(path)):
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            t[i] = t[i-1] + np.sqrt(dx*dx + dy*dy)
        
        # 归一化参数化变量
        if t[-1] > 0:
            t = t / t[-1]
        
        # 为了保证起点和终点不变，在两端添加重复点
        x = np.concatenate([[x[0]], x, [x[-1]]])
        y = np.concatenate([[y[0]], y, [y[-1]]])
        t = np.concatenate([[0], t, [1]])
        
        try:
            # 使用B样条插值，设置较小的平滑因子以保持路径形状
            tck, _ = splprep([x, y], u=t, s=len(path) * 0.1, k=min(3, len(path)-1))
            
            # 生成更密集的点
            u = np.linspace(0, 1, self.interpolation_points)
            smooth_points = np.array(splev(u, tck)).T
            
            # 保证起点和终点不变
            smooth_points[0] = points[0]
            smooth_points[-1] = points[-1]
            
        except Exception as e:
            self.logger.warning(f"路径平滑失败: {e}，返回原始路径")
            return path
        
        # 将插值点转换为整数坐标，使用更精确的四舍五入
        smooth_path = []
        prev_point = None
        for p in smooth_points:
            # 对坐标进行四舍五入
            point = (int(round(p[0])), int(round(p[1])))
            
            # 避免重复点
            if point != prev_point:
                smooth_path.append(point)
                prev_point = point
        
        # 验证平滑路径的可行性
        valid_path = []
        for point in smooth_path:
            if self.is_valid_position(point):
                valid_path.append(point)
            else:
                # 如果遇到无效点，尝试找到最近的有效点
                found_valid = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        new_point = (point[0] + dy, point[1] + dx)
                        if self.is_valid_position(new_point):
                            valid_path.append(new_point)
                            found_valid = True
                            break
                    if found_valid:
                        break
                if not found_valid:
                    self.logger.warning(f"平滑路径点{point}无效，已跳过")
        
        # 确保起点和终点正确
        if valid_path and valid_path[0] != path[0]:
            valid_path[0] = path[0]
        if valid_path and valid_path[-1] != path[-1]:
            valid_path[-1] = path[-1]
        
        # 如果平滑路径无效，返回原始路径
        if not valid_path:
            self.logger.warning("平滑路径无效，返回原始路径")
            return path
        
        return valid_path
    
    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        smooth: bool = True
    ) -> List[Tuple[int, int]]:
        """规划路径的主函数
        
        Args:
            start: 起点坐标 (row, col)
            goal: 终点坐标 (row, col)
            smooth: 是否对路径进行平滑处理
            
        Returns:
            List[Tuple[int, int]]: 路径点列表
        """
        # 使用A*算法寻找路径
        path = self.find_path(start, goal)
        
        if not path:
            return []
        
        # 如果需要，对路径进行平滑处理
        if smooth and len(path) > 2:
            path = self.smooth_path(path)
        
        return path 