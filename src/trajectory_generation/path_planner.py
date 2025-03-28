"""
路径规划器
使用A*算法在环境地图上规划最优路径
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
import heapq
import numpy as np

from ..data_processing import TerrainLoader, EnvironmentMapper

logger = logging.getLogger(__name__)

class PathPlanner:
    """路径规划器"""
    
    def __init__(
            self,
            terrain_loader: TerrainLoader,
            environment_mapper: EnvironmentMapper,
            config: Dict
        ):
        """
        初始化路径规划器
        
        Args:
            terrain_loader: 地形数据加载器实例
            environment_mapper: 环境地图生成器实例
            config: 配置参数字典
        """
        self.terrain_loader = terrain_loader
        self.environment_mapper = environment_mapper
        self.config = config
        
        # 获取成本图和其他环境地图
        self.maps = environment_mapper.get_maps()
        self.cost_map = self.maps['cost_map']
        
        # 定义移动方向（8个方向）
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # 对角线移动的距离系数
        self.move_costs = [
            np.sqrt(2), 1, np.sqrt(2),
            1,              1,
            np.sqrt(2), 1, np.sqrt(2)
        ]
    
    def plan_path(
            self,
            start: Tuple[int, int],
            end: Tuple[int, int]
        ) -> Optional[List[Tuple[int, int]]]:
        """
        使用A*算法规划路径
        
        Args:
            start: 起点坐标 (row, col)
            end: 终点坐标 (row, col)
            
        Returns:
            Optional[List[Tuple[int, int]]]: 路径点列表，如果找不到路径则返回None
        """
        logger.info(f"开始规划路径: 从{start}到{end}")
        
        # 验证起终点
        if not self._is_valid_point(start) or not self._is_valid_point(end):
            logger.error("起点或终点无效")
            return None
        
        # A*算法的数据结构
        open_set = []  # 优先队列
        closed_set = set()  # 已访问的节点
        came_from = {}  # 路径追踪
        g_score = {start: 0}  # 从起点到当前点的实际代价
        f_score = {start: self._heuristic(start, end)}  # 估计的总代价
        
        # 将起点加入开放列表
        heapq.heappush(open_set, (f_score[start], start))
        
        while open_set:
            # 获取f值最小的节点
            current = heapq.heappop(open_set)[1]
            
            # 到达终点
            if current == end:
                logger.info("找到路径")
                return self._reconstruct_path(came_from, end)
            
            # 将当前节点加入关闭列表
            closed_set.add(current)
            
            # 检查相邻节点
            for i, (dr, dc) in enumerate(self.directions):
                neighbor = (current[0] + dr, current[1] + dc)
                
                # 检查节点是否有效
                if not self._is_valid_point(neighbor):
                    continue
                
                # 如果节点已在关闭列表中，跳过
                if neighbor in closed_set:
                    continue
                
                # 计算经过当前节点到邻居节点的代价
                move_cost = self.move_costs[i]
                tentative_g_score = (
                    g_score[current] +
                    move_cost * self.cost_map[neighbor]
                )
                
                # 如果找到更好的路径或是新节点
                if (neighbor not in g_score or
                    tentative_g_score < g_score[neighbor]):
                    # 更新路径
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = (
                        g_score[neighbor] +
                        self._heuristic(neighbor, end)
                    )
                    
                    # 将邻居节点加入开放列表
                    heapq.heappush(
                        open_set,
                        (f_score[neighbor], neighbor)
                    )
        
        logger.warning("未找到有效路径")
        return None
    
    def _is_valid_point(self, point: Tuple[int, int]) -> bool:
        """
        检查点是否有效
        
        Args:
            point: 坐标点 (row, col)
            
        Returns:
            bool: 是否有效
        """
        row, col = point
        shape = self.cost_map.shape
        
        # 检查边界
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            return False
        
        # 检查是否可通行
        if np.isinf(self.cost_map[row, col]):
            return False
        
        return True
    
    def _heuristic(
            self,
            point: Tuple[int, int],
            goal: Tuple[int, int]
        ) -> float:
        """
        计算启发式值（使用欧几里得距离）
        
        Args:
            point: 当前点
            goal: 目标点
            
        Returns:
            float: 启发式值
        """
        return np.sqrt(
            (point[0] - goal[0])**2 +
            (point[1] - goal[1])**2
        )
    
    def _reconstruct_path(
            self,
            came_from: Dict[Tuple[int, int], Tuple[int, int]],
            end: Tuple[int, int]
        ) -> List[Tuple[int, int]]:
        """
        重建路径
        
        Args:
            came_from: 路径追踪字典
            end: 终点
            
        Returns:
            List[Tuple[int, int]]: 路径点列表
        """
        path = [end]
        current = end
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path
    
    def smooth_path(
            self,
            path: List[Tuple[int, int]],
            smoothing_factor: float = 0.5
        ) -> List[Tuple[int, int]]:
        """
        使用路径平滑算法优化路径
        
        Args:
            path: 原始路径
            smoothing_factor: 平滑因子 (0-1)
            
        Returns:
            List[Tuple[int, int]]: 平滑后的路径
        """
        if len(path) <= 2:
            return path
        
        smoothed = np.array(path, dtype=float)
        change = True
        while change:
            change = False
            for i in range(1, len(path) - 1):
                old_point = smoothed[i].copy()
                
                # 向前后点移动
                smoothed[i] = (smoothed[i] +
                    smoothing_factor * (
                        smoothed[i-1] +
                        smoothed[i+1] -
                        2 * smoothed[i]
                    )
                )
                
                # 检查新位置是否有效
                new_point = tuple(map(int, smoothed[i]))
                if not self._is_valid_point(new_point):
                    smoothed[i] = old_point
                    continue
                
                # 如果点有显著移动，继续迭代
                if not np.allclose(old_point, smoothed[i]):
                    change = True
        
        return [tuple(map(int, p)) for p in smoothed] 