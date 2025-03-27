"""
路径规划器模块
负责使用A*算法进行智能路径规划
"""

import heapq
import logging
import numpy as np
from typing import List, Tuple, Dict, Set

from ..data_processing import GISDataLoader

# 配置日志
logger = logging.getLogger(__name__)

class Node:
    """A*算法中的节点类"""
    def __init__(self, position: Tuple[float, float], g_cost: float = float('inf'),
                 h_cost: float = 0.0, parent=None):
        self.position = position  # (lon, lat)
        self.g_cost = g_cost     # 从起点到当前节点的实际代价
        self.h_cost = h_cost     # 从当前节点到终点的估计代价
        self.parent = parent     # 父节点

    @property
    def f_cost(self) -> float:
        """总代价 = 实际代价 + 估计代价"""
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        """用于优先队列的比较"""
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        """用于比较两个节点是否相同"""
        if not isinstance(other, Node):
            return False
        return self.position == other.position

    def __hash__(self):
        """用于在集合中使用节点"""
        return hash(self.position)

class PathPlanner:
    """路径规划器类，使用A*算法进行路径规划"""

    def __init__(self, gis_loader, grid_size: float = 0.005):
        """初始化路径规划器

        Args:
            gis_loader: GIS数据加载器实例
            grid_size: 网格大小（度），默认0.005度（约500米）
        """
        self.gis_data_loader = gis_loader
        self.step_size = grid_size
        self.max_slope = 30.0   # 最大允许坡度
        self.directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # 上右下左
            (1, 1), (1, -1), (-1, -1), (-1, 1)  # 对角线
        ]
        self.obstacle_cost = 1000.0  # 障碍物代价
        self.slope_cost_factor = 10.0  # 坡度代价因子

    def plan_path(self, start_point: Tuple[float, float],
                  end_point: Tuple[float, float]) -> List[Tuple[float, float]]:
        """使用A*算法规划从起点到终点的路径

        Args:
            start_point: 起点坐标 (lon, lat)
            end_point: 终点坐标 (lon, lat)

        Returns:
            路径点列表 [(lon, lat), ...]
        """
        # 初始化起点和终点节点
        start_node = Node(start_point, g_cost=0.0,
                         h_cost=self._calculate_distance(start_point, end_point))
        end_node = Node(end_point)

        # 初始化开放列表和关闭列表
        open_list = []
        closed_set = set()
        node_dict = {}  # 用于快速查找节点

        # 将起点加入开放列表
        heapq.heappush(open_list, start_node)
        node_dict[start_point] = start_node

        while open_list:
            # 获取f值最小的节点
            current_node = heapq.heappop(open_list)
            current_pos = current_node.position

            # 如果到达终点附近
            if self._calculate_distance(current_pos, end_point) < self.step_size:
                # 确保最后一个点是精确的终点
                path = self._reconstruct_path(current_node)
                if path[-1] != end_point:
                    path.append(end_point)
                return path

            # 将当前节点加入关闭列表
            closed_set.add(current_pos)

            # 获取相邻节点
            neighbors = self._get_neighbors(current_pos)
            for next_pos in neighbors:
                # 如果该位置已在关闭列表中，跳过
                if next_pos in closed_set:
                    continue

                # 计算移动代价
                movement_cost = self._calculate_movement_cost(current_pos, next_pos)
                if movement_cost is None:  # 如果位置完全不可通行
                    continue

                # 计算从起点经过当前节点到相邻节点的代价
                tentative_g_cost = current_node.g_cost + movement_cost

                # 获取或创建相邻节点
                if next_pos in node_dict:
                    next_node = node_dict[next_pos]
                    if tentative_g_cost >= next_node.g_cost:
                        continue
                else:
                    next_node = Node(next_pos)
                    node_dict[next_pos] = next_node

                # 更新节点信息
                next_node.parent = current_node
                next_node.g_cost = tentative_g_cost
                next_node.h_cost = self._calculate_distance(next_pos, end_point)

                # 将相邻节点加入开放列表
                if next_node not in open_list:
                    heapq.heappush(open_list, next_node)

        # 如果没有找到路径，返回直线路径
        logger.warning("未找到最优路径，使用直线路径")
        return self._generate_straight_path(start_point, end_point)

    def _get_neighbors(self, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """获取给定位置的相邻节点

        Args:
            position: 当前位置 (lon, lat)

        Returns:
            相邻位置列表
        """
        neighbors = []
        lon, lat = position
        for dx, dy in self.directions:
            next_pos = (
                lon + dx * self.step_size,
                lat + dy * self.step_size
            )
            neighbors.append(next_pos)
        return neighbors

    def _calculate_movement_cost(self, from_pos: Tuple[float, float],
                               to_pos: Tuple[float, float]) -> float:
        """计算从一个位置移动到另一个位置的代价

        Args:
            from_pos: 起始位置 (lon, lat)
            to_pos: 目标位置 (lon, lat)

        Returns:
            移动代价，如果完全不可通行则返回None
        """
        # 基础移动代价（欧氏距离）
        base_cost = self._calculate_distance(from_pos, to_pos)

        # 检查目标位置的地形条件
        slope = self.gis_data_loader.get_slope(*to_pos)
        landcover = self.gis_data_loader.get_landcover(*to_pos)

        # 如果是水体或建筑物，直接返回None表示不可通行
        if landcover in [1, 2]:
            return None

        # 如果坡度过大，直接返回None表示不可通行
        if slope > self.max_slope:
            return None

        # 根据坡度增加移动代价
        slope_cost = base_cost * (1 + (slope / self.max_slope) * self.slope_cost_factor)

        return slope_cost

    def _is_valid_position(self, position: Tuple[float, float]) -> bool:
        """检查位置是否可通行

        Args:
            position: 位置坐标 (lon, lat)

        Returns:
            位置是否可通行
        """
        lon, lat = position
        
        # 检查坡度
        slope = self.gis_data_loader.get_slope(lon, lat)
        if slope > self.max_slope:
            return False

        # 检查土地覆盖类型
        landcover = self.gis_data_loader.get_landcover(lon, lat)
        if landcover in [1, 2]:  # 水体和建筑物
            return False

        return True

    def _calculate_distance(self, pos1: Tuple[float, float],
                          pos2: Tuple[float, float]) -> float:
        """计算两点之间的欧氏距离

        Args:
            pos1: 第一个点的坐标 (lon, lat)
            pos2: 第二个点的坐标 (lon, lat)

        Returns:
            两点之间的距离
        """
        return np.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2
        )

    def _reconstruct_path(self, end_node: Node) -> List[Tuple[float, float]]:
        """从终点节点重建完整路径

        Args:
            end_node: 终点节点

        Returns:
            路径点列表
        """
        path = []
        current = end_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # 反转列表，使其从起点开始

    def _generate_straight_path(self, start_point: Tuple[float, float],
                              end_point: Tuple[float, float]) -> List[Tuple[float, float]]:
        """生成两点之间的直线路径

        Args:
            start_point: 起点坐标 (lon, lat)
            end_point: 终点坐标 (lon, lat)

        Returns:
            路径点列表
        """
        distance = self._calculate_distance(start_point, end_point)
        num_points = max(2, int(distance / self.step_size))
        
        path = []
        for i in range(num_points):
            t = i / (num_points - 1)
            lon = start_point[0] + t * (end_point[0] - start_point[0])
            lat = start_point[1] + t * (end_point[1] - start_point[1])
            
            # 检查点是否可通行
            point = (lon, lat)
            if self._is_valid_position(point):
                path.append(point)
            else:
                # 如果点不可通行，尝试在周围找一个可通行的点
                found_valid = False
                for dx, dy in self.directions:
                    new_point = (
                        lon + dx * self.step_size * 0.5,
                        lat + dy * self.step_size * 0.5
                    )
                    if self._is_valid_position(new_point):
                        path.append(new_point)
                        found_valid = True
                        break
                
                if not found_valid:
                    logger.warning(f"在位置 {point} 附近未找到可通行点")
                    continue
        
        # 确保路径至少包含起点和终点
        if not path:
            path = [start_point, end_point]
        elif path[0] != start_point:
            path.insert(0, start_point)
        elif path[-1] != end_point:
            path.append(end_point)
            
        return path 