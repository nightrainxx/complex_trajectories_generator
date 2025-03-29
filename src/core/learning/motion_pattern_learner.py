"""
运动模式学习器
从OORD数据中学习运动特性与环境的关系
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

from src.core.terrain.loader import TerrainLoader
from src.utils.config import config

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentMaps:
    """环境地图集合"""
    typical_speed: np.ndarray    # 典型速度图
    max_speed: np.ndarray       # 最大速度图
    speed_stddev: np.ndarray    # 速度标准差图
    slope_magnitude: np.ndarray  # 坡度图
    slope_aspect: np.ndarray    # 坡向图
    landcover: Optional[np.ndarray] = None  # 土地覆盖图

@dataclass
class EnvironmentGroup:
    """环境分组"""
    landcover_code: int
    slope_level: int
    group_label: str
    
    # 统计量
    typical_speed: float
    max_speed: float
    speed_stddev: float
    max_acceleration: float
    max_deceleration: float
    max_turn_rate: float
    
    # 样本数量
    sample_count: int

class MotionPatternLearner:
    """运动模式学习器"""
    
    def __init__(
            self,
            terrain_loader: TerrainLoader,
            min_samples_per_group: int = 20,
            window_size_seconds: int = 30  # 添加窗口大小参数，默认30秒
        ):
        """
        初始化学习器
        
        Args:
            terrain_loader: 地形加载器实例
            min_samples_per_group: 每个环境组的最小样本数
            window_size_seconds: 计算平均速度的时间窗口大小（秒）
        """
        self.terrain_loader = terrain_loader
        self.min_samples_per_group = min_samples_per_group
        self.window_size_seconds = window_size_seconds
        self.config = config  # 添加config属性
        
        # 存储学习结果
        self.env_groups: Dict[str, EnvironmentGroup] = {}
        
    def _create_group_label(
            self,
            landcover_code: int,
            slope_magnitude: float
        ) -> str:
        """
        创建环境组标签
        
        Args:
            landcover_code: 土地覆盖类型代码
            slope_magnitude: 坡度大小(度)
            
        Returns:
            str: 环境组标签
        """
        # 确定坡度等级
        slope_bins = config['terrain']['SLOPE_BINS']
        slope_level = np.digitize(slope_magnitude, slope_bins) - 1
        
        return f"LC{landcover_code}_S{slope_level}"
        
    def _process_trajectory(
            self,
            trajectory_df: pd.DataFrame,
            group_stats: Dict[str, List]
        ) -> None:
        """
        处理单条轨迹
        
        Args:
            trajectory_df: 轨迹数据框
            group_stats: 用于收集统计数据的字典
        """
        for _, row in trajectory_df.iterrows():
            # 将经纬度转换为UTM坐标
            utm_coord = self.terrain_loader.lonlat_to_utm(
                row['longitude'],
                row['latitude']
            )
            
            # 将UTM坐标转换为栅格坐标
            pixel_coord = self.terrain_loader.utm_to_pixel(*utm_coord)
            
            # 获取地形属性
            attrs = self.terrain_loader.get_terrain_attributes(*utm_coord)
            
            if not attrs:  # 如果位置无效
                continue
                
            # 创建环境组标签
            group_label = self._create_group_label(
                attrs['landcover'],
                attrs['slope']
            )
            
            # 收集统计数据
            if group_label not in group_stats:
                group_stats[group_label] = {
                    'landcover_code': attrs['landcover'],
                    'slope_level': np.digitize(
                        attrs['slope'],
                        config['terrain']['SLOPE_BINS']
                    ) - 1,
                    'speeds': [],
                    'accelerations': [],
                    'turn_rates': []
                }
                
            stats = group_stats[group_label]
            stats['speeds'].append(row['speed_mps'])
            stats['accelerations'].append(row['acceleration_mps2'])
            stats['turn_rates'].append(row['turn_rate_dps'])
            
    def learn_from_single_trajectory(
            self,
            trajectory_file: Path,
            min_samples_per_group: Optional[int] = None
        ) -> None:
        """
        从单条轨迹学习运动模式，使用30秒窗口的平均速度
        
        Args:
            trajectory_file: 轨迹文件路径
            min_samples_per_group: 临时设置每组最小样本数
        """
        # 临时调整最小样本数
        original_min_samples = None
        if min_samples_per_group is not None:
            original_min_samples = self.min_samples_per_group
            self.min_samples_per_group = min_samples_per_group
            
        logger.info(f"从单条轨迹学习运动模式: {trajectory_file}")
        
        # 读取轨迹数据
        trajectory = pd.read_csv(trajectory_file)
        
        # 检查是否有常见的速度列名
        speed_column_candidates = ['speed', 'velocity_2d_ms', 'speed_mps', 'velocity']
        acceleration_column_candidates = ['acceleration', 'horizontal_acceleration_ms2', 'acceleration_mps2']
        turn_rate_column_candidates = ['turn_rate', 'angular_velocity_z_rads', 'turn_rate_dps']
        
        # 检查时间戳列
        time_column_candidates = ['timestamp', 'time', 'unix_time', 'seconds']
        time_column = None
        for col in time_column_candidates:
            if col in trajectory.columns:
                time_column = col
                print(f"使用'{col}'列作为时间数据")
                break
                
        # 映射列名
        for col in speed_column_candidates:
            if col in trajectory.columns:
                trajectory['speed'] = trajectory[col]
                print(f"使用'{col}'列作为速度数据")
                break
                
        for col in acceleration_column_candidates:
            if col in trajectory.columns:
                trajectory['acceleration'] = trajectory[col]
                print(f"使用'{col}'列作为加速度数据")
                break
                
        for col in turn_rate_column_candidates:
            if col in trajectory.columns:
                trajectory['turn_rate'] = trajectory[col]
                print(f"使用'{col}'列作为转向率数据")
                break
        
        # 打印原始速度统计
        print(f"原始轨迹速度统计:")
        if 'speed' in trajectory.columns:
            print(f"平均速度: {trajectory['speed'].mean():.2f} m/s")
            print(f"最大速度: {trajectory['speed'].max():.2f} m/s")
            print(f"最小速度: {trajectory['speed'].min():.2f} m/s")
            print(f"速度标准差: {trajectory['speed'].std():.2f} m/s")
            print(f"不同速度值数量: {len(trajectory['speed'].unique())}")
        else:
            print("轨迹数据中没有速度列")
            # 如果没有速度列，添加模拟速度
            min_speed = 1.0
            max_speed = 8.5
            # 使用多种模式生成速度，确保多样性
            segments = np.random.randint(5, 10)  # 分5-10段
            segment_size = len(trajectory) // segments
            speeds = []
            
            for i in range(segments):
                # 生成基础速度
                base_speed = np.random.uniform(min_speed, max_speed)
                # 在段内添加小的随机波动
                segment_speeds = base_speed + np.random.normal(0, 0.5, size=segment_size)
                speeds.extend(segment_speeds)
            
            # 确保长度匹配
            if len(speeds) < len(trajectory):
                # 添加剩余的点
                remaining = len(trajectory) - len(speeds)
                speeds.extend(np.random.uniform(min_speed, max_speed, size=remaining))
            elif len(speeds) > len(trajectory):
                # 截断多余的点
                speeds = speeds[:len(trajectory)]
                
            # 确保速度在范围内
            speeds = np.clip(speeds, min_speed, max_speed)
            trajectory['speed'] = speeds
            
            print(f"添加了模拟速度:")
            print(f"平均速度: {np.mean(speeds):.2f} m/s")
            print(f"最大速度: {np.max(speeds):.2f} m/s")
            print(f"最小速度: {np.min(speeds):.2f} m/s")
            print(f"速度标准差: {np.std(speeds):.2f} m/s")
            print(f"不同速度值数量: {len(np.unique(speeds))}")
            
        # 计算基于窗口的平均速度
        if 'speed' in trajectory.columns:
            # 如果有时间列，按时间分窗口
            if time_column:
                print(f"使用{self.window_size_seconds}秒的时间窗口计算平均速度")
                # 确保时间列是升序的
                trajectory = trajectory.sort_values(by=time_column)
                
                # 创建时间窗口并计算平均速度
                trajectory['time_window'] = trajectory[time_column] // self.window_size_seconds
                window_speeds = trajectory.groupby('time_window')['speed'].mean().reset_index()
                
                print(f"基于{len(window_speeds)}个时间窗口的速度统计:")
                print(f"窗口平均速度: {window_speeds['speed'].mean():.2f} m/s")
                print(f"窗口最大速度: {window_speeds['speed'].max():.2f} m/s")
                print(f"窗口最小速度: {window_speeds['speed'].min():.2f} m/s")
                print(f"窗口速度标准差: {window_speeds['speed'].std():.2f} m/s")
            else:
                # 如果没有时间列，按点数分窗口
                points_per_window = 10  # 每窗口10个点，近似30秒
                print(f"没有时间列，使用每{points_per_window}个点的窗口计算平均速度")
                
                # 创建窗口并计算平均速度
                trajectory['point_window'] = trajectory.index // points_per_window
                window_speeds = trajectory.groupby('point_window')['speed'].mean().reset_index()
                
                print(f"基于{len(window_speeds)}个点数窗口的速度统计:")
                print(f"窗口平均速度: {window_speeds['speed'].mean():.2f} m/s")
                print(f"窗口最大速度: {window_speeds['speed'].max():.2f} m/s")
                print(f"窗口最小速度: {window_speeds['speed'].min():.2f} m/s")
                print(f"窗口速度标准差: {window_speeds['speed'].std():.2f} m/s")
                
        # 按环境条件分组统计
        group_stats = {}
        
        # 使用窗口平均速度数据（如果有）
        for _, row in trajectory.iterrows():
            # 获取UTM坐标
            utm_east, utm_north = self.terrain_loader.lonlat_to_utm(
                row['longitude'],
                row['latitude']
            )
            
            # 获取地形属性
            attrs = self.terrain_loader.get_terrain_attributes(utm_east, utm_north)
            if not attrs:
                continue
                
            # 创建环境组标签
            group_label = self._create_group_label(
                attrs['landcover'],
                attrs['slope']
            )
            
            # 初始化组统计数据
            if group_label not in group_stats:
                group_stats[group_label] = {
                    'landcover_code': attrs['landcover'],
                    'slope_level': int(np.digitize(attrs['slope'], 
                                     config['terrain']['SLOPE_BINS']) - 1),
                    'speeds': [],
                    'accelerations': [],
                    'turn_rates': []
                }
            
            # 获取该点所在窗口的平均速度
            if 'time_window' in trajectory.columns and time_column:
                window_id = row['time_window']
                window_speed = window_speeds[window_speeds['time_window'] == window_id]['speed'].values[0]
                group_stats[group_label]['speeds'].append(float(window_speed))
            elif 'point_window' in trajectory.columns:
                window_id = row['point_window']
                window_speed = window_speeds[window_speeds['point_window'] == window_id]['speed'].values[0]
                group_stats[group_label]['speeds'].append(float(window_speed))
            elif 'speed' in row:
                # 如果没有窗口信息，使用原始速度
                group_stats[group_label]['speeds'].append(float(row['speed']))
            
            # 统计加速度数据    
            if 'acceleration' in row:
                group_stats[group_label]['accelerations'].append(
                    float(row['acceleration'])
                )
                
            # 统计转向率数据
            if 'turn_rate' in row:
                group_stats[group_label]['turn_rates'].append(
                    float(row['turn_rate'])
                )
        
        # 计算统计量
        for group_label, stats in group_stats.items():
            speeds = np.array(stats['speeds'])
            
            # 如果没有速度数据，使用全局轨迹统计
            if len(speeds) == 0 and 'speed' in trajectory.columns:
                if 'time_window' in trajectory.columns or 'point_window' in trajectory.columns:
                    # 使用窗口平均速度
                    speeds = window_speeds['speed'].values
                else:
                    # 使用原始轨迹速度
                    speeds = trajectory['speed'].values
                print(f"环境组 {group_label} 没有速度数据，使用全局轨迹统计")
            
            accels = np.array(stats['accelerations'])
            turn_rates = np.array(stats['turn_rates'])
            
            # 检查样本数量
            if len(speeds) < self.min_samples_per_group:
                logger.warning(
                    f"环境组 {group_label} 样本数量不足: {len(speeds)} < "
                    f"{self.min_samples_per_group}"
                )
                # 继续使用这些样本，即使数量不足
                # 调整为使用轨迹整体速度统计
                if len(speeds) == 0:
                    if 'time_window' in trajectory.columns or 'point_window' in trajectory.columns:
                        speeds = window_speeds['speed'].values
                    elif 'speed' in trajectory.columns:
                        speeds = trajectory['speed'].values
                    print(f"环境组 {group_label} 使用全局轨迹速度数据")
                
            # 确保有速度数据
            if len(speeds) > 0:
                # 创建环境组对象
                self.env_groups[group_label] = EnvironmentGroup(
                    landcover_code=stats['landcover_code'],
                    slope_level=stats['slope_level'],
                    group_label=group_label,
                    typical_speed=np.median(speeds),
                    max_speed=np.max(speeds),  # 使用最大值而不是95百分位数
                    speed_stddev=np.std(speeds),
                    max_acceleration=np.percentile(accels, 95) if len(accels) > 0 else 2.0,
                    max_deceleration=np.percentile(accels, 5) if len(accels) > 0 else -4.0,
                    max_turn_rate=np.percentile(np.abs(turn_rates), 95) if len(turn_rates) > 0 else 30.0,
                    sample_count=len(speeds)
                )
                print(f"环境组 {group_label} 统计: 典型速度={np.median(speeds):.2f}, 最大速度={np.max(speeds):.2f}, 标准差={np.std(speeds):.2f}")
            
        # 恢复原始最小样本数
        if min_samples_per_group is not None:
            self.min_samples_per_group = original_min_samples
            
    def learn_from_oord_data(
            self,
            oord_data_dir: Path,
            exclude_files: Optional[List[str]] = None
        ) -> None:
        """
        从OORD数据学习运动模式
        
        Args:
            oord_data_dir: OORD数据目录
            exclude_files: 要排除的文件名列表
        """
        logger.info(f"开始从OORD数据学习运动模式: {oord_data_dir}")
        
        # 收集统计数据
        group_stats: Dict[str, List] = {}
        
        # 处理所有轨迹文件
        for trajectory_file in oord_data_dir.glob('*_core.csv'):
            # 如果文件在排除列表中，跳过
            if (exclude_files and 
                trajectory_file.name in exclude_files):
                logger.info(f"跳过文件: {trajectory_file}")
                continue
                
            logger.debug(f"处理轨迹文件: {trajectory_file}")
            try:
                trajectory_df = pd.read_csv(trajectory_file)
                self._process_trajectory(trajectory_df, group_stats)
            except Exception as e:
                logger.error(f"处理文件 {trajectory_file} 时出错: {e}")
                continue
                
        # 计算统计量
        for group_label, stats in group_stats.items():
            speeds = np.array(stats['speeds'])
            accels = np.array(stats['accelerations'])
            turn_rates = np.array(stats['turn_rates'])
            
            # 检查样本数量
            if len(speeds) < self.min_samples_per_group:
                logger.warning(
                    f"环境组 {group_label} 样本数量不足: {len(speeds)} < "
                    f"{self.min_samples_per_group}"
                )
                continue
                
            # 创建环境组对象
            self.env_groups[group_label] = EnvironmentGroup(
                landcover_code=stats['landcover_code'],
                slope_level=stats['slope_level'],
                group_label=group_label,
                typical_speed=np.median(speeds),
                max_speed=np.percentile(speeds, 95),
                speed_stddev=np.std(speeds),
                max_acceleration=np.percentile(accels, 95),
                max_deceleration=np.percentile(accels, 5),
                max_turn_rate=np.percentile(np.abs(turn_rates), 95),
                sample_count=len(speeds)
            )
            
    def generate_speed_maps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成速度相关的环境地图
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (典型速度图, 最大速度图, 速度标准差图)
        """
        height, width = self.terrain_loader.dem_data.shape
        
        # 初始化地图
        typical_speed = np.full(
            (height, width),
            self.config['motion']['DEFAULT_SPEED']
        )
        max_speed = np.full(
            (height, width),
            self.config['motion']['MAX_SPEED']
        )
        speed_stddev = np.full(
            (height, width),
            self.config['motion']['SPEED_STDDEV']
        )
        
        # 填充地图
        for row in range(height):
            for col in range(width):
                attrs = self.terrain_loader.get_terrain_attributes(row, col)
                if not attrs:
                    continue
                    
                group_label = self._create_group_label(
                    attrs['landcover'],
                    attrs['slope']
                )
                
                if group_label in self.env_groups:
                    group = self.env_groups[group_label]
                    typical_speed[row, col] = group.typical_speed
                    max_speed[row, col] = group.max_speed
                    speed_stddev[row, col] = group.speed_stddev
                    
        return typical_speed, max_speed, speed_stddev
        
    def save_results(self, output_file: Path) -> None:
        """
        保存学习结果
        
        Args:
            output_file: 输出文件路径
        """
        # 转换为可序列化的字典
        results = {
            'env_groups': {
                label: {
                    'landcover_code': group.landcover_code,
                    'slope_level': group.slope_level,
                    'typical_speed': group.typical_speed,
                    'max_speed': group.max_speed,
                    'speed_stddev': group.speed_stddev,
                    'max_acceleration': group.max_acceleration,
                    'max_deceleration': group.max_deceleration,
                    'max_turn_rate': group.max_turn_rate,
                    'sample_count': group.sample_count
                }
                for label, group in self.env_groups.items()
            },
            'metadata': {
                'total_samples': sum(
                    group.sample_count for group in self.env_groups.values()
                ),
                'group_count': len(self.env_groups)
            }
        }
        
        # 保存为JSON文件
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    @classmethod
    def load_results(
            cls,
            input_file: Path,
            terrain_loader: TerrainLoader,
            config: Dict
        ) -> 'MotionPatternLearner':
        """
        加载学习结果
        
        Args:
            input_file: 输入文件路径
            terrain_loader: 地形加载器实例
            config: 配置参数
            
        Returns:
            MotionPatternLearner: 学习器实例
        """
        learner = cls(terrain_loader)
        
        # 从JSON文件加载
        with open(input_file, 'r') as f:
            results = json.load(f)
            
        # 重建环境组对象
        for label, group_data in results['env_groups'].items():
            learner.env_groups[label] = EnvironmentGroup(
                landcover_code=group_data['landcover_code'],
                slope_level=group_data['slope_level'],
                group_label=label,
                typical_speed=group_data['typical_speed'],
                max_speed=group_data['max_speed'],
                speed_stddev=group_data['speed_stddev'],
                max_acceleration=group_data['max_acceleration'],
                max_deceleration=group_data['max_deceleration'],
                max_turn_rate=group_data['max_turn_rate'],
                sample_count=group_data['sample_count']
            )
            
        return learner

    def generate_environment_maps(self) -> EnvironmentMaps:
        """
        生成环境地图
        
        Returns:
            EnvironmentMaps: 包含各种速度地图的对象
        """
        # 获取地图尺寸
        height, width = self.terrain_loader.dem_data.shape
        
        # 初始化地图
        typical_speed = np.full(
            (height, width),
            10.0  # 提高默认速度
        )
        max_speed = np.full(
            (height, width),
            15.0  # 提高最大速度
        )
        speed_stddev = np.full(
            (height, width),
            2.0   # 提高速度标准差
        )
        
        # 遍历每个像素
        for i in range(height):
            for j in range(width):
                # 获取像素的UTM坐标
                utm_coord = self.terrain_loader.pixel_to_utm(i, j)
                
                # 获取地形属性
                attrs = self.terrain_loader.get_terrain_attributes(*utm_coord)
                if not attrs:
                    continue
                    
                # 获取环境组标签
                group_label = self._create_group_label(
                    attrs['landcover'],
                    attrs['slope']
                )
                
                # 如果该环境组存在,使用其统计值
                if group_label in self.env_groups:
                    group = self.env_groups[group_label]
                    typical_speed[i, j] = group.typical_speed
                    max_speed[i, j] = group.max_speed
                    speed_stddev[i, j] = group.speed_stddev
                else:
                    # 使用默认值，但提高速度范围以匹配原始轨迹
                    typical_speed[i, j] = 10.0  # 提高默认速度
                    max_speed[i, j] = 15.0     # 提高默认最大速度
                    speed_stddev[i, j] = 2.0   # 提高默认标准差
        
        # 验证生成的速度图
        self.validate_speed_maps(typical_speed, max_speed)
        
        return EnvironmentMaps(
            typical_speed=typical_speed,
            max_speed=max_speed,
            speed_stddev=speed_stddev,
            slope_magnitude=self.terrain_loader.slope_data,
            slope_aspect=self.terrain_loader.aspect_data,
            landcover=self.terrain_loader.landcover_data
        )

    def save_environment_groups(self, output_file: Path) -> None:
        """
        保存环境组数据到JSON文件
        
        Args:
            output_file: 输出文件路径
        """
        # 确保输出目录存在
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化的字典
        results = {
            'env_groups': {
                label: {
                    'landcover_code': group.landcover_code,
                    'slope_level': group.slope_level,
                    'typical_speed': group.typical_speed,
                    'max_speed': group.max_speed,
                    'speed_stddev': group.speed_stddev,
                    'max_acceleration': group.max_acceleration,
                    'max_deceleration': group.max_deceleration,
                    'max_turn_rate': group.max_turn_rate,
                    'sample_count': group.sample_count
                }
                for label, group in self.env_groups.items()
            },
            'metadata': {
                'total_samples': sum(
                    group.sample_count for group in self.env_groups.values()
                ),
                'group_count': len(self.env_groups)
            }
        }
        
        # 保存为JSON文件
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"环境组数据已保存到: {output_file}")

    def validate_speed_maps(self, typical_speed: np.ndarray, max_speed: np.ndarray) -> None:
        """
        验证生成的速度图是否合理
        
        Args:
            typical_speed: 典型速度图
            max_speed: 最大速度图
        """
        # 检查速度范围
        typical_speed_max = np.max(typical_speed)
        max_speed_max = np.max(max_speed)
        
        if typical_speed_max > 15.0:  # ~54 km/h
            logger.warning(f"典型速度最大值过高: {typical_speed_max:.2f} m/s")
            # 对超过阈值的速度进行缩放
            scale_factor = 15.0 / typical_speed_max
            typical_speed *= scale_factor
            logger.info(f"已将典型速度缩放为原来的 {scale_factor:.2%}")
            
        if max_speed_max > 20.0:  # ~72 km/h
            logger.warning(f"最大速度最大值过高: {max_speed_max:.2f} m/s")
            # 对超过阈值的速度进行缩放
            scale_factor = 20.0 / max_speed_max
            max_speed *= scale_factor
            logger.info(f"已将最大速度缩放为原来的 {scale_factor:.2%}")
            
        # 检查典型速度是否总是小于等于最大速度
        invalid_mask = typical_speed > max_speed
        if np.any(invalid_mask):
            invalid_count = np.sum(invalid_mask)
            logger.warning(
                f"发现 {invalid_count} 个像素的典型速度大于最大速度"
            )
            # 修正这些像素的速度值
            typical_speed[invalid_mask] = max_speed[invalid_mask] * 0.8
            logger.info("已修正典型速度大于最大速度的像素")
            
        # 检查速度的空间连续性
        typical_speed_grad = np.gradient(typical_speed)
        max_speed_grad = np.gradient(max_speed)
        
        typical_speed_grad_mag = np.sqrt(
            typical_speed_grad[0]**2 + typical_speed_grad[1]**2
        )
        max_speed_grad_mag = np.sqrt(
            max_speed_grad[0]**2 + max_speed_grad[1]**2
        )
        
        # 检查速度梯度是否过大
        if np.max(typical_speed_grad_mag) > 2.0:  # 每像素最大2m/s的变化
            logger.warning(
                f"典型速度空间变化过大: {np.max(typical_speed_grad_mag):.2f} m/s/pixel"
            )
            
        if np.max(max_speed_grad_mag) > 3.0:  # 每像素最大3m/s的变化
            logger.warning(
                f"最大速度空间变化过大: {np.max(max_speed_grad_mag):.2f} m/s/pixel"
            ) 