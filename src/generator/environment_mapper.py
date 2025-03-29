"""环境地图生成器模块

此模块负责生成环境地图，包括：
1. 最大速度地图：基于坡度大小和土地覆盖类型
2. 典型速度地图：考虑坡度方向对速度的影响
3. 速度标准差地图：反映速度的变化程度
4. 成本地图：用于路径规划

输入:
    - 土地覆盖栅格文件 (.tif)
    - 坡度大小栅格文件 (.tif)
    - 坡度方向栅格文件 (.tif)
    - 环境组学习结果文件 (.json) (可选，使用学习结果生成地图)

输出:
    - 最大速度地图 (max_speed_map.tif)
    - 典型速度地图 (typical_speed_map.tif)
    - 速度标准差地图 (speed_stddev_map.tif)
    - 成本地图 (cost_map.tif)
"""

import numpy as np
import rasterio
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, Any
import os
import json

# 从统一配置文件导入配置
from config import (
    SLOPE_BINS, MAX_SLOPE_THRESHOLD, 
    LANDCOVER_SPEED_FACTORS, LANDCOVER_COST_FACTORS,
    TYPICAL_SPEED_FACTOR, UP_SLOPE_FACTOR, DOWN_SLOPE_FACTOR, CROSS_SLOPE_FACTOR,
    BASE_SPEED_STDDEV_FACTOR, SLOPE_STDDEV_FACTOR, COMPLEX_TERRAIN_STDDEV_FACTOR,
    COMPLEX_TERRAIN_CODES, IMPASSABLE_LANDCOVER_CODES,
    MAX_SPEED, DEFAULT_SPEED
)

class EnvironmentMapper:
    """环境地图生成器类"""
    
    def __init__(
        self, 
        landcover_path: str, 
        slope_magnitude_path: str, 
        slope_aspect_path: str,
        environment_groups_path: Optional[str] = None
    ):
        """初始化环境地图生成器
        
        Args:
            landcover_path: 土地覆盖栅格文件路径
            slope_magnitude_path: 坡度大小栅格文件路径
            slope_aspect_path: 坡度方向栅格文件路径
            environment_groups_path: 环境组学习结果文件路径(可选)
        """
        # 检查文件是否存在
        for path in [landcover_path, slope_magnitude_path, slope_aspect_path]:
            if not Path(path).exists():
                raise FileNotFoundError(f"找不到文件: {path}")
        
        # 读取栅格数据
        with rasterio.open(landcover_path) as src:
            self.landcover_data = src.read(1)
            self.transform = src.transform
            self.meta = src.meta.copy()
            self.height = src.height
            self.width = src.width
        
        with rasterio.open(slope_magnitude_path) as src:
            self.slope_magnitude_data = src.read(1)
        
        with rasterio.open(slope_aspect_path) as src:
            self.slope_aspect_data = src.read(1)
        
        # 验证数据形状一致
        if not (self.landcover_data.shape == self.slope_magnitude_data.shape == self.slope_aspect_data.shape):
            raise ValueError("输入数据形状不一致")
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        
        # 加载环境组数据（如果提供）
        self.environment_groups = {}
        if environment_groups_path and Path(environment_groups_path).exists():
            try:
                with open(environment_groups_path, 'r') as f:
                    self.environment_groups = json.load(f)
                self.logger.info(f"已加载环境组数据: {len(self.environment_groups)}个组")
                
                # 记录学习到的环境组信息
                for group_label, group_data in self.environment_groups.items():
                    self.logger.debug(f"环境组 {group_label}: 典型速度={group_data.get('typical_speed', 'N/A')}, "
                                    f"最大速度={group_data.get('max_speed', 'N/A')}, "
                                    f"速度标准差={group_data.get('speed_stddev', 'N/A')}")
            except Exception as e:
                self.logger.error(f"加载环境组数据失败: {e}")
                self.environment_groups = {}
    
    def _get_slope_bin(self, slope_magnitude: float) -> int:
        """获取坡度等级
        
        Args:
            slope_magnitude: 坡度大小(度)
            
        Returns:
            int: 坡度等级
        """
        return np.digitize(slope_magnitude, SLOPE_BINS) - 1
    
    def _get_group_label(self, landcover_code: int, slope_magnitude: float) -> str:
        """获取环境组标签
        
        Args:
            landcover_code: 土地覆盖类型代码
            slope_magnitude: 坡度大小(度)
            
        Returns:
            str: 环境组标签
        """
        slope_bin = self._get_slope_bin(slope_magnitude)
        return f"LC{landcover_code}_S{slope_bin}"
    
    def _get_environment_values(self, landcover_code: int, slope_magnitude: float) -> Dict[str, float]:
        """获取环境组的值
        
        根据环境组标签获取对应的环境参数值，如果找不到则使用默认规则计算
        
        Args:
            landcover_code: 土地覆盖类型代码
            slope_magnitude: 坡度大小(度)
            
        Returns:
            Dict[str, float]: 包含typical_speed, max_speed, speed_stddev的字典
        """
        # 生成环境组标签
        group_label = self._get_group_label(landcover_code, slope_magnitude)
        
        # 检查是否在学习到的环境组中
        if group_label in self.environment_groups:
            group_data = self.environment_groups[group_label]
            return {
                'typical_speed': group_data.get('typical_speed', DEFAULT_SPEED),
                'max_speed': group_data.get('max_speed', MAX_SPEED),
                'speed_stddev': group_data.get('speed_stddev', DEFAULT_SPEED * BASE_SPEED_STDDEV_FACTOR)
            }
        else:
            # 使用默认规则计算
            return self._calculate_default_values(landcover_code, slope_magnitude)
    
    def _calculate_default_values(self, landcover_code: int, slope_magnitude: float) -> Dict[str, float]:
        """使用默认规则计算环境参数值
        
        Args:
            landcover_code: 土地覆盖类型代码
            slope_magnitude: 坡度大小(度)
            
        Returns:
            Dict[str, float]: 包含typical_speed, max_speed, speed_stddev的字典
        """
        # 计算最大速度
        # 1. 检查是否可通行
        if landcover_code in IMPASSABLE_LANDCOVER_CODES or slope_magnitude > MAX_SLOPE_THRESHOLD:
            return {
                'typical_speed': 0.0,
                'max_speed': 0.0,
                'speed_stddev': 0.0
            }
        
        # 2. 应用坡度影响
        slope_factor = np.clip(1 - SLOPE_SPEED_FACTOR * slope_magnitude, 0.1, 1.0)
        
        # 3. 应用土地覆盖影响
        lc_factor = LANDCOVER_SPEED_FACTORS.get(landcover_code, 1.0)
        
        # 计算最大速度
        max_speed = MAX_SPEED * slope_factor * lc_factor
        
        # 计算典型速度
        typical_speed = max_speed * TYPICAL_SPEED_FACTOR
        
        # 计算速度标准差
        stddev_factor = 1.0
        if landcover_code in COMPLEX_TERRAIN_CODES:
            stddev_factor *= COMPLEX_TERRAIN_STDDEV_FACTOR
        
        slope_stddev_factor = np.clip(
            1 + SLOPE_STDDEV_FACTOR * (slope_magnitude / MAX_SLOPE_THRESHOLD),
            1.0,
            2.0
        )
        stddev_factor *= slope_stddev_factor
        
        speed_stddev = typical_speed * BASE_SPEED_STDDEV_FACTOR * stddev_factor
        
        return {
            'typical_speed': typical_speed,
            'max_speed': max_speed,
            'speed_stddev': speed_stddev
        }
    
    def calculate_max_speed_map(self) -> np.ndarray:
        """计算最大速度地图
        
        基于坡度大小和土地覆盖类型计算每个像素的最大可能速度。
        如果有学习结果，则优先使用学习结果。
        
        Returns:
            np.ndarray: 最大速度地图（米/秒）
        """
        # 初始化最大速度地图
        max_speed = np.zeros(self.landcover_data.shape, dtype=np.float32)
        
        # 使用学习结果或默认规则计算每个像素的最大速度
        for i in range(self.height):
            for j in range(self.width):
                landcover_code = int(self.landcover_data[i, j])
                slope_magnitude = self.slope_magnitude_data[i, j]
                
                env_values = self._get_environment_values(landcover_code, slope_magnitude)
                max_speed[i, j] = env_values['max_speed']
        
        # 记录速度图统计信息
        self.logger.info(f"最大速度图统计:")
        valid_speeds = max_speed[max_speed > 0]
        if len(valid_speeds) > 0:
            self.logger.info(f"  最大值: {np.max(valid_speeds):.2f} m/s")
            self.logger.info(f"  最小值: {np.min(valid_speeds):.2f} m/s")
            self.logger.info(f"  平均值: {np.mean(valid_speeds):.2f} m/s")
            self.logger.info(f"  标准差: {np.std(valid_speeds):.2f} m/s")
            self.logger.info(f"  不同值数量: {len(np.unique(valid_speeds))}")
        else:
            self.logger.warning("最大速度图中没有有效的速度值")
        
        return max_speed
    
    def calculate_typical_speed_map(self) -> np.ndarray:
        """计算典型速度地图
        
        基于学习结果或默认规则计算每个像素的典型速度。
        
        Returns:
            np.ndarray: 典型速度地图（米/秒）
        """
        # 初始化典型速度地图
        typical_speed = np.zeros(self.landcover_data.shape, dtype=np.float32)
        
        # 使用学习结果或默认规则计算每个像素的典型速度
        for i in range(self.height):
            for j in range(self.width):
                landcover_code = int(self.landcover_data[i, j])
                slope_magnitude = self.slope_magnitude_data[i, j]
                
                env_values = self._get_environment_values(landcover_code, slope_magnitude)
                typical_speed[i, j] = env_values['typical_speed']
        
        # 记录速度图统计信息
        self.logger.info(f"典型速度图统计:")
        valid_speeds = typical_speed[typical_speed > 0]
        if len(valid_speeds) > 0:
            self.logger.info(f"  最大值: {np.max(valid_speeds):.2f} m/s")
            self.logger.info(f"  最小值: {np.min(valid_speeds):.2f} m/s")
            self.logger.info(f"  平均值: {np.mean(valid_speeds):.2f} m/s")
            self.logger.info(f"  标准差: {np.std(valid_speeds):.2f} m/s")
            self.logger.info(f"  不同值数量: {len(np.unique(valid_speeds))}")
            
            # 如果速度图异常（如恒定值），发出警告
            if len(np.unique(valid_speeds)) < 10:
                self.logger.warning("典型速度图变化很小，可能导致模拟轨迹速度恒定")
        else:
            self.logger.warning("典型速度图中没有有效的速度值")
        
        return typical_speed
    
    def calculate_speed_stddev_map(self) -> np.ndarray:
        """计算速度标准差地图
        
        基于学习结果或默认规则计算每个像素的速度标准差。
        
        Returns:
            np.ndarray: 速度标准差地图（米/秒）
        """
        # 初始化速度标准差地图
        speed_stddev = np.zeros(self.landcover_data.shape, dtype=np.float32)
        
        # 使用学习结果或默认规则计算每个像素的速度标准差
        for i in range(self.height):
            for j in range(self.width):
                landcover_code = int(self.landcover_data[i, j])
                slope_magnitude = self.slope_magnitude_data[i, j]
                
                env_values = self._get_environment_values(landcover_code, slope_magnitude)
                speed_stddev[i, j] = env_values['speed_stddev']
        
        # 记录速度标准差图统计信息
        self.logger.info(f"速度标准差图统计:")
        valid_stddevs = speed_stddev[speed_stddev > 0]
        if len(valid_stddevs) > 0:
            self.logger.info(f"  最大值: {np.max(valid_stddevs):.2f} m/s")
            self.logger.info(f"  最小值: {np.min(valid_stddevs):.2f} m/s")
            self.logger.info(f"  平均值: {np.mean(valid_stddevs):.2f} m/s")
            self.logger.info(f"  标准差: {np.std(valid_stddevs):.2f} m/s")
        else:
            self.logger.warning("速度标准差图中没有有效的值")
        
        return speed_stddev
    
    def calculate_cost_map(self) -> np.ndarray:
        """计算成本地图
        
        基于典型速度和土地覆盖类型计算通行成本。
        不可通行区域的成本设为无穷大。
        
        Returns:
            np.ndarray: 成本地图（秒/米）
        """
        # 获取典型速度
        typical_speed = self.calculate_typical_speed_map()
        
        # 初始化成本地图
        cost = np.zeros_like(typical_speed)
        
        # 处理不可通行区域
        impassable_mask = np.isin(self.landcover_data, IMPASSABLE_LANDCOVER_CODES)
        steep_mask = self.slope_magnitude_data > MAX_SLOPE_THRESHOLD
        cost[impassable_mask | steep_mask] = np.inf
        
        # 计算可通行区域的成本
        passable_mask = ~(impassable_mask | steep_mask)
        # 确保不除以零
        valid_speed_mask = (typical_speed > 0) & passable_mask
        cost[valid_speed_mask] = 1 / typical_speed[valid_speed_mask]  # 基础成本：单位距离所需时间
        
        # 应用土地覆盖成本因子
        for code, factor in LANDCOVER_COST_FACTORS.items():
            landcover_mask = self.landcover_data == code
            cost[landcover_mask & valid_speed_mask] *= factor
        
        return cost
    
    def generate_environment_maps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """生成所有环境地图
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (最大速度地图, 典型速度地图, 速度标准差地图, 成本地图)
        """
        self.logger.info("开始生成环境地图...")
        
        # 生成各类地图
        max_speed_map = self.calculate_max_speed_map()
        self.logger.info("最大速度地图生成完成")
        
        typical_speed_map = self.calculate_typical_speed_map()
        self.logger.info("典型速度地图生成完成")
        
        speed_stddev_map = self.calculate_speed_stddev_map()
        self.logger.info("速度标准差地图生成完成")
        
        cost_map = self.calculate_cost_map()
        self.logger.info("成本地图生成完成")
        
        return max_speed_map, typical_speed_map, speed_stddev_map, cost_map
    
    def save_environment_maps(
        self,
        output_dir: str,
        max_speed_map: np.ndarray,
        typical_speed_map: np.ndarray,
        speed_stddev_map: np.ndarray,
        cost_map: np.ndarray
    ) -> None:
        """保存环境地图
        
        Args:
            output_dir: 输出目录路径
            max_speed_map: 最大速度地图
            typical_speed_map: 典型速度地图
            speed_stddev_map: 速度标准差地图
            cost_map: 成本地图
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备元数据
        meta = self.meta.copy()
        meta.update(dtype=np.float32)
        
        # 保存最大速度地图
        with rasterio.open(os.path.join(output_dir, "max_speed_map.tif"), 'w', **meta) as dst:
            dst.write(max_speed_map.astype(np.float32), 1)
        
        # 保存典型速度地图
        with rasterio.open(os.path.join(output_dir, "typical_speed_map.tif"), 'w', **meta) as dst:
            dst.write(typical_speed_map.astype(np.float32), 1)
        
        # 保存速度标准差地图
        with rasterio.open(os.path.join(output_dir, "speed_stddev_map.tif"), 'w', **meta) as dst:
            dst.write(speed_stddev_map.astype(np.float32), 1)
        
        # 保存成本地图
        with rasterio.open(os.path.join(output_dir, "cost_map.tif"), 'w', **meta) as dst:
            dst.write(cost_map.astype(np.float32), 1)
        
        self.logger.info(f"环境地图已保存到: {output_dir}")
    
    def get_environment_params(self, row: int, col: int) -> dict:
        """获取指定位置的环境参数
        
        Args:
            row: 像素行号（从0开始）
            col: 像素列号（从0开始）
            
        Returns:
            包含环境参数的字典：
            {
                'max_speed': 最大速度 (m/s),
                'typical_speed': 典型速度 (m/s),
                'speed_stddev': 速度标准差 (m/s),
                'cost': 移动成本 (s/m),
                'landcover': 土地覆盖类型代码,
                'slope_magnitude': 坡度大小 (度),
                'slope_aspect': 坡向 (度),
                'is_passable': 是否可通行
            }
        
        Raises:
            ValueError: 如果位置超出范围
        """
        # 检查位置是否在有效范围内
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"位置 ({row}, {col}) 超出范围")
        
        # 获取土地覆盖和坡度信息
        landcover = self.landcover_data[row, col]
        slope_magnitude = self.slope_magnitude_data[row, col]
        slope_aspect = self.slope_aspect_data[row, col]
        
        # 判断是否可通行
        is_passable = (
            landcover not in IMPASSABLE_LANDCOVER_CODES and
            slope_magnitude <= MAX_SLOPE_THRESHOLD
        )
        
        # 获取环境参数
        max_speed = self.calculate_max_speed_map()[row, col]
        typical_speed = self.calculate_typical_speed_map()[row, col]
        speed_stddev = self.calculate_speed_stddev_map()[row, col]
        cost = self.calculate_cost_map()[row, col]
        
        return {
            'max_speed': float(max_speed),
            'typical_speed': float(typical_speed),
            'speed_stddev': float(speed_stddev),
            'cost': float(cost),
            'landcover': int(landcover),
            'slope_magnitude': float(slope_magnitude),
            'slope_aspect': float(slope_aspect),
            'is_passable': bool(is_passable)
        } 