"""
运动模式学习器
从OORD数据中学习目标在不同环境下的运动特性
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .terrain_analyzer import TerrainAnalyzer
from .terrain_loader import TerrainLoader

logger = logging.getLogger(__name__)

class MotionPatternLearner:
    """运动模式学习器"""
    
    def __init__(self, terrain_loader: TerrainLoader):
        """
        初始化运动模式学习器
        
        Args:
            terrain_loader: 地形数据加载器实例
        """
        self.terrain_loader = terrain_loader
        self.terrain_analyzer = TerrainAnalyzer(terrain_loader)
        
        # 学习结果
        self.learned_patterns = {
            'slope_speed_model': None,          # 坡度-速度关系模型
            'slope_direction_model': None,      # 坡向-速度关系模型
            'landcover_speed_stats': None,      # 地表类型-速度统计
            'turn_rate_stats': None,            # 转向率统计
            'acceleration_stats': None,         # 加速度统计
            'environment_clusters': None        # 环境分类结果
        }
        
        # 环境分组参数
        self.env_params = {
            'slope_bins': [0, 5, 15, 30, 90],   # 坡度分组边界
            'direction_bins': [-180, -135, -90, -45, 0, 45, 90, 135, 180],  # 相对坡向分组
            'min_samples': 100                   # 每组最小样本数
        }
    
    def learn_from_trajectories(self, trajectories: List[pd.DataFrame]) -> None:
        """
        从轨迹数据中学习运动模式
        
        Args:
            trajectories: OORD轨迹数据列表
        """
        logger.info("开始从%d条轨迹中学习运动模式", len(trajectories))
        
        # 合并所有轨迹数据
        combined_data = self._preprocess_trajectories(trajectories)
        
        # 学习各种运动模式
        self._learn_slope_speed_relation(combined_data)
        self._learn_slope_direction_effect(combined_data)
        self._learn_landcover_speed_relation(combined_data)
        self._learn_turn_rate_patterns(combined_data)
        self._learn_acceleration_patterns(combined_data)
        self._cluster_environments(combined_data)
        
        logger.info("运动模式学习完成")
    
    def _preprocess_trajectories(self, trajectories: List[pd.DataFrame]) -> pd.DataFrame:
        """
        预处理轨迹数据，添加环境特征
        
        Args:
            trajectories: 轨迹数据列表
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        processed_data = []
        
        for traj in trajectories:
            # 计算基本运动特征
            traj = traj.copy()
            traj['speed'] = np.sqrt(
                traj['velocity_north_ms']**2 + 
                traj['velocity_east_ms']**2
            )
            traj['acceleration'] = np.sqrt(
                traj['acceleration_x_ms2']**2 + 
                traj['acceleration_y_ms2']**2 + 
                traj['acceleration_z_ms2']**2
            )
            traj['turn_rate'] = np.sqrt(
                traj['angular_velocity_x_rads']**2 + 
                traj['angular_velocity_y_rads']**2 + 
                traj['angular_velocity_z_rads']**2
            )
            
            # 计算行进方向（航向角）
            traj['heading'] = np.degrees(np.arctan2(
                traj['velocity_east_ms'],
                traj['velocity_north_ms']
            ))
            traj['heading'] = np.where(traj['heading'] < 0, 
                                     traj['heading'] + 360,
                                     traj['heading'])
            
            # 添加环境特征
            for idx, row in traj.iterrows():
                terrain_attrs = self.terrain_analyzer.get_terrain_attributes(
                    row['longitude'],
                    row['latitude']
                )
                traj.loc[idx, 'slope_magnitude'] = terrain_attrs['slope_magnitude']
                traj.loc[idx, 'slope_aspect'] = terrain_attrs['slope_aspect']
                traj.loc[idx, 'landcover'] = self.terrain_loader.get_landcover(
                    row['longitude'],
                    row['latitude']
                )
            
            # 计算相对坡向（行进方向与坡向的夹角）
            traj['relative_aspect'] = traj['heading'] - traj['slope_aspect']
            traj['relative_aspect'] = np.where(
                traj['relative_aspect'] > 180,
                traj['relative_aspect'] - 360,
                traj['relative_aspect']
            )
            traj['relative_aspect'] = np.where(
                traj['relative_aspect'] < -180,
                traj['relative_aspect'] + 360,
                traj['relative_aspect']
            )
            
            processed_data.append(traj)
        
        return pd.concat(processed_data, ignore_index=True)
    
    def _learn_slope_speed_relation(self, data: pd.DataFrame) -> None:
        """学习坡度与速度的关系"""
        # 按坡度分组统计速度
        slope_speed = data.groupby(pd.cut(
            data['slope_magnitude'],
            bins=self.env_params['slope_bins']
        ))['speed'].agg(['mean', 'std', 'count'])
        
        # 过滤掉样本数不足的组
        slope_speed = slope_speed[slope_speed['count'] >= self.env_params['min_samples']]
        
        # 计算速度因子（相对于平地速度的比例）
        flat_speed = slope_speed.iloc[0]['mean']  # 第一组（0-5度）作为基准
        slope_speed['speed_factor'] = slope_speed['mean'] / flat_speed
        
        self.learned_patterns['slope_speed_model'] = slope_speed
    
    def _learn_slope_direction_effect(self, data: pd.DataFrame) -> None:
        """学习坡向对速度的影响"""
        # 按相对坡向和坡度大小分组
        direction_groups = data.groupby([
            pd.cut(data['relative_aspect'], 
                  bins=self.env_params['direction_bins']),
            pd.cut(data['slope_magnitude'],
                  bins=self.env_params['slope_bins'])
        ])
        
        # 统计每组的速度特征
        direction_speed = direction_groups['speed'].agg(['mean', 'std', 'count'])
        
        # 过滤样本数不足的组
        direction_speed = direction_speed[
            direction_speed['count'] >= self.env_params['min_samples']
        ]
        
        # 计算速度影响因子
        flat_forward_speed = direction_speed.xs(
            (slice(-45, 45), slice(0, 5)),  # 平地前向组
            level=[0, 1]
        )['mean'].mean()
        
        direction_speed['speed_factor'] = direction_speed['mean'] / flat_forward_speed
        
        self.learned_patterns['slope_direction_model'] = direction_speed
    
    def _learn_landcover_speed_relation(self, data: pd.DataFrame) -> None:
        """学习地表类型与速度的关系"""
        # 按地表类型分组统计
        landcover_speed = data.groupby('landcover')['speed'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ])
        
        # 过滤样本数不足的组
        landcover_speed = landcover_speed[
            landcover_speed['count'] >= self.env_params['min_samples']
        ]
        
        # 计算速度因子
        road_speed = landcover_speed.loc[1]['mean']  # 假设1为道路类型
        landcover_speed['speed_factor'] = landcover_speed['mean'] / road_speed
        
        self.learned_patterns['landcover_speed_stats'] = landcover_speed
    
    def _learn_turn_rate_patterns(self, data: pd.DataFrame) -> None:
        """学习转向率模式"""
        # 计算转向率统计特征
        turn_rate_stats = {
            'mean': float(data['turn_rate'].mean()),
            'std': float(data['turn_rate'].std()),
            'percentiles': {
                '50': float(data['turn_rate'].quantile(0.5)),
                '75': float(data['turn_rate'].quantile(0.75)),
                '90': float(data['turn_rate'].quantile(0.9)),
                '95': float(data['turn_rate'].quantile(0.95)),
                '99': float(data['turn_rate'].quantile(0.99))
            }
        }
        
        self.learned_patterns['turn_rate_stats'] = turn_rate_stats
    
    def _learn_acceleration_patterns(self, data: pd.DataFrame) -> None:
        """学习加速度模式"""
        # 计算加速度统计特征
        acceleration_stats = {
            'mean': float(data['acceleration'].mean()),
            'std': float(data['acceleration'].std()),
            'percentiles': {
                '50': float(data['acceleration'].quantile(0.5)),
                '75': float(data['acceleration'].quantile(0.75)),
                '90': float(data['acceleration'].quantile(0.9)),
                '95': float(data['acceleration'].quantile(0.95)),
                '99': float(data['acceleration'].quantile(0.99))
            }
        }
        
        self.learned_patterns['acceleration_stats'] = acceleration_stats
    
    def _cluster_environments(self, data: pd.DataFrame) -> None:
        """
        对环境特征进行聚类分析
        用于发现典型的环境组合模式
        """
        # 准备特征
        features = ['slope_magnitude', 'relative_aspect', 'landcover']
        X = data[features].copy()
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means聚类
        n_clusters = 5  # 可以根据需要调整
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # 分析每个簇的特征
        cluster_stats = []
        for i in range(n_clusters):
            cluster_data = data[clusters == i]
            stats = {
                'size': len(cluster_data),
                'slope_mean': float(cluster_data['slope_magnitude'].mean()),
                'slope_std': float(cluster_data['slope_magnitude'].std()),
                'relative_aspect_mean': float(cluster_data['relative_aspect'].mean()),
                'relative_aspect_std': float(cluster_data['relative_aspect'].std()),
                'landcover_mode': int(cluster_data['landcover'].mode().iloc[0]),
                'speed_mean': float(cluster_data['speed'].mean()),
                'speed_std': float(cluster_data['speed'].std())
            }
            cluster_stats.append(stats)
        
        self.learned_patterns['environment_clusters'] = {
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'feature_names': features,
            'scaler': scaler,
            'cluster_stats': cluster_stats
        }
    
    def get_learned_patterns(self) -> Dict:
        """
        获取学习到的运动模式
        
        Returns:
            Dict: 学习结果字典
        """
        return self.learned_patterns
    
    def save_patterns(self, filepath: str) -> None:
        """
        保存学习结果到文件
        
        Args:
            filepath: 保存路径
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.learned_patterns, f)
        logger.info("学习结果已保存到: %s", filepath)
    
    def load_patterns(self, filepath: str) -> None:
        """
        从文件加载学习结果
        
        Args:
            filepath: 文件路径
        """
        import pickle
        with open(filepath, 'rb') as f:
            self.learned_patterns = pickle.load(f)
        logger.info("已从%s加载学习结果", filepath) 