"""
路径规划器模块
使用A*算法在成本图上规划最优路径
"""

import numpy as np
import logging
from typing import List, Tuple, Set, Dict, Optional
from queue import PriorityQueue
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Node:
    """A*搜索节点"""
    row: int
    col: int
    g_cost: float  # 从起点到当前节点的实际成本
    h_cost: float  # 从当前节点到终点的启发式估计成本
    parent: Optional['Node'] = None
    
    @property
    def f_cost(self) -> float:
        """总成本 f(n) = g(n) + h(n)"""
        return self.g_cost + self.h_cost
    
    def __lt__(self, other: 'Node') -> bool:
        """优先队列比较函数"""
        return self.f_cost < other.f_cost

class PathPlanner:
    """A*路径规划器"""
    
    def __init__(self, cost_map: np.ndarray):
        """
        初始化路径规划器
        
        Args:
            cost_map: 成本图，表示每个像素的通行成本
        """
        self.cost_map = cost_map
        self.height, self.width = cost_map.shape
        
    def heuristic(self, node: Node, end: Tuple[int, int]) -> float:
        """
        计算启发式成本（使用欧几里得距离）
        
        Args:
            node: 当前节点
            end: 终点坐标 (row, col)
            
        Returns:
            float: 启发式估计成本
        """
        # 使用欧几里得距离乘以成本图中的最小成本作为启发式
        min_cost = np.min(self.cost_map[self.cost_map > 0])
        return min_cost * np.sqrt(
            (node.row - end[0])**2 + (node.col - end[1])**2
        )
        
    def get_neighbors(self, node: Node) -> List[Tuple[int, int]]:
        """
        获取节点的相邻节点坐标
        
        Args:
            node: 当前节点
            
        Returns:
            List[Tuple[int, int]]: 相邻节点坐标列表
        """
        neighbors = []
        # 8个方向：上、下、左、右、左上、右上、左下、右下
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        
        for dr, dc in directions:
            new_row = node.row + dr
            new_col = node.col + dc
            
            # 检查边界
            if (0 <= new_row < self.height and 
                0 <= new_col < self.width):
                # 检查是否可通行（成本不为无穷大）
                if self.cost_map[new_row, new_col] < float('inf'):
                    neighbors.append((new_row, new_col))
                    
        return neighbors
        
    def find_path(
            self,
            start: Tuple[int, int],
            end: Tuple[int, int]
        ) -> List[Tuple[int, int]]:
        """
        使用A*算法寻找最优路径
        
        Args:
            start: 起点坐标 (row, col)
            end: 终点坐标 (row, col)
            
        Returns:
            List[Tuple[int, int]]: 路径点列表，从起点到终点
        """
        # 检查起点和终点是否有效
        if (not (0 <= start[0] < self.height and 
                0 <= start[1] < self.width)):
            raise ValueError(f"无效的起点坐标: {start}")
        if (not (0 <= end[0] < self.height and 
                0 <= end[1] < self.width)):
            raise ValueError(f"无效的终点坐标: {end}")
            
        # 检查起点和终点是否可通行
        if self.cost_map[start] == float('inf'):
            raise ValueError(f"起点不可通行: {start}")
        if self.cost_map[end] == float('inf'):
            raise ValueError(f"终点不可通行: {end}")
            
        # 初始化开放列表（优先队列）和关闭列表
        open_list = PriorityQueue()
        closed_set: Set[Tuple[int, int]] = set()
        
        # 创建起点节点
        start_node = Node(
            row=start[0],
            col=start[1],
            g_cost=0,
            h_cost=self.heuristic(Node(start[0], start[1], 0, 0), end)
        )
        
        # 将起点加入开放列表
        open_list.put(start_node)
        node_dict: Dict[Tuple[int, int], Node] = {
            (start[0], start[1]): start_node
        }
        
        # 开始搜索
        while not open_list.empty():
            current = open_list.get()
            
            # 如果到达终点
            if (current.row, current.col) == end:
                # 重建路径
                path = []
                while current:
                    path.append((current.row, current.col))
                    current = current.parent
                return path[::-1]  # 反转路径，从起点到终点
                
            # 将当前节点加入关闭列表
            closed_set.add((current.row, current.col))
            
            # 检查所有相邻节点
            for neighbor_pos in self.get_neighbors(current):
                # 如果节点已在关闭列表中，跳过
                if neighbor_pos in closed_set:
                    continue
                    
                # 计算从起点经过当前节点到相邻节点的成本
                # 对角线移动的成本为√2倍
                is_diagonal = (abs(neighbor_pos[0] - current.row) + 
                             abs(neighbor_pos[1] - current.col)) == 2
                movement_cost = (
                    np.sqrt(2) if is_diagonal else 1
                ) * self.cost_map[neighbor_pos]
                
                new_g_cost = current.g_cost + movement_cost
                
                # 如果是新节点或找到更好的路径
                if (neighbor_pos not in node_dict or 
                    new_g_cost < node_dict[neighbor_pos].g_cost):
                    # 创建新的相邻节点
                    neighbor = Node(
                        row=neighbor_pos[0],
                        col=neighbor_pos[1],
                        g_cost=new_g_cost,
                        h_cost=self.heuristic(
                            Node(neighbor_pos[0], neighbor_pos[1], 0, 0),
                            end
                        ),
                        parent=current
                    )
                    
                    # 更新或添加到开放列表
                    node_dict[neighbor_pos] = neighbor
                    open_list.put(neighbor)
                    
        # 如果没有找到路径
        raise ValueError("未找到可行路径") 