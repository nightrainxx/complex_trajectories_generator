"""
环境分析模块
负责分析OORD轨迹在不同环境条件（坡度、土地覆盖）下的运动特征
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer

from ..config import SLOPE_BINS, SLOPE_LABELS
from ..data_processing import TerrainLoader, OORDProcessor

# 配置日志
logger = logging.getLogger(__name__)

class EnvironmentAnalyzer:
    """环境分析器，用于学习轨迹与环境的关系"""
    
    def __init__(self, gis_loader: TerrainLoader):
        """
        初始化环境分析器
        
        Args:
            gis_loader: 已加载GIS数据的GISDataLoader实例
        """
        self.gis_loader = gis_loader
        self.environment_stats: Dict[str, Dict] = {}  # 存储环境-运动特征统计
        self.speed_models: Dict[str, Dict] = {}  # 存储环境组的速度模型
    
    def analyze_trajectory(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """
        分析单条轨迹的环境特征
        
        Args:
            trajectory_df: 预处理后的轨迹DataFrame
            
        Returns:
            enriched_df: 添加了环境信息的DataFrame
        """
        df = trajectory_df.copy()
        
        # 获取轨迹点的像素坐标
        pixel_coords = np.array([
            self.gis_loader.get_pixel_coords(lon, lat)
            for lon, lat in zip(df['longitude'], df['latitude'])
        ])
        
        # 添加环境信息
        df['elevation'] = [
            self.gis_loader.get_elevation(row, col)
            for row, col in pixel_coords
        ]
        
        df['slope'] = [
            self.gis_loader.get_slope(row, col)
            for row, col in pixel_coords
        ]
        
        df['landcover'] = [
            self.gis_loader.get_landcover(row, col)
            for row, col in pixel_coords
        ]
        
        # 添加坡度等级
        df['slope_class'] = pd.cut(
            df['slope'],
            bins=SLOPE_BINS,
            labels=SLOPE_LABELS,
            include_lowest=True
        ).astype(str).str.replace('S', '')  # 移除'S'前缀，只保留数字
        
        # 创建环境组标签
        df['environment_group'] = df.apply(
            lambda x: f"LC{int(x['landcover'])}_SS{x['slope_class']}", 
            axis=1
        )
        
        return df
    
    def analyze_all_trajectories(self, processed_trajectories: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        分析所有轨迹的环境特征
        
        Args:
            processed_trajectories: 预处理后的轨迹字典
            
        Returns:
            enriched_trajectories: 添加了环境信息的轨迹字典
        """
        enriched_trajectories = {}
        for traj_id, df in processed_trajectories.items():
            try:
                enriched_df = self.analyze_trajectory(df)
                enriched_trajectories[traj_id] = enriched_df
                logger.info(f"完成轨迹 {traj_id} 的环境分析")
            except Exception as e:
                logger.error(f"分析轨迹 {traj_id} 时出错: {str(e)}")
                continue
        
        return enriched_trajectories
    
    def compute_environment_statistics(self, enriched_trajectories: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        计算不同环境条件下的运动特征统计
        
        Args:
            enriched_trajectories: 添加了环境信息的轨迹字典
            
        Returns:
            environment_stats: 环境-运动特征统计字典
        """
        # 合并所有轨迹数据
        all_data = pd.concat(enriched_trajectories.values(), ignore_index=True)
        
        # 按环境组进行分组统计
        grouped = all_data.groupby('environment_group')
        
        for group_name, group_data in grouped:
            # 计算速度统计量
            speed_stats = {
                'mean': group_data['speed'].mean(),
                'std': group_data['speed'].std(),
                'median': group_data['speed'].median(),
                'q25': group_data['speed'].quantile(0.25),
                'q75': group_data['speed'].quantile(0.75),
                'max': group_data['speed'].quantile(0.95),  # 使用95分位数作为最大速度
                'min': group_data['speed'].quantile(0.05)   # 使用5分位数作为最小速度
            }
            
            # 计算转向率统计量
            turn_rate_stats = {
                'mean': group_data['turn_rate'].mean(),
                'std': group_data['turn_rate'].std(),
                'median': group_data['turn_rate'].median(),
                'max': abs(group_data['turn_rate']).quantile(0.95)
            }
            
            # 计算加速度统计量
            accel_stats = {
                'mean': group_data['acceleration'].mean(),
                'std': group_data['acceleration'].std(),
                'max_accel': group_data['acceleration'].quantile(0.95),
                'max_decel': abs(group_data['acceleration'].quantile(0.05))
            }
            
            # 存储该环境组的统计结果
            self.environment_stats[group_name] = {
                'speed': speed_stats,
                'turn_rate': turn_rate_stats,
                'acceleration': accel_stats,
                'sample_size': len(group_data)
            }
        
        logger.info(f"完成 {len(self.environment_stats)} 个环境组的统计分析")
        return self.environment_stats
    
    def fit_speed_models(self, enriched_trajectories: Dict[str, pd.DataFrame], min_samples: int = 10) -> Dict[str, Dict]:
        """
        为每个环境组拟合速度分布模型
        
        Args:
            enriched_trajectories: 添加了环境信息的轨迹字典
            min_samples: 拟合模型所需的最小样本数量
            
        Returns:
            speed_models: 环境组的速度模型字典
        """
        # 合并所有轨迹数据
        all_data = pd.concat(enriched_trajectories.values(), ignore_index=True)
        
        # 按环境组拟合速度分布
        for group_name, group_data in all_data.groupby('environment_group'):
            speeds = group_data['speed'].dropna().values
            
            if len(speeds) < min_samples:  # 样本太少，跳过拟合
                logger.warning(f"环境组 {group_name} 样本数量不足 ({len(speeds)})")
                continue
            
            try:
                # 尝试拟合多个分布，选择最佳拟合
                distributions = ['norm', 'gamma', 'lognorm']
                best_dist = None
                best_params = None
                best_kstest = float('inf')
                
                for dist_name in distributions:
                    # 拟合分布
                    params = getattr(stats, dist_name).fit(speeds)
                    # 进行KS检验
                    ks_statistic, _ = stats.kstest(speeds, dist_name, params)
                    
                    if ks_statistic < best_kstest:
                        best_dist = dist_name
                        best_params = params
                        best_kstest = ks_statistic
                
                self.speed_models[group_name] = {
                    'distribution': best_dist,
                    'parameters': best_params,
                    'ks_statistic': best_kstest
                }
                
                logger.info(f"环境组 {group_name} 速度分布拟合完成，使用 {best_dist} 分布")
                
            except Exception as e:
                logger.error(f"拟合环境组 {group_name} 的速度分布时出错: {str(e)}")
                continue
        
        return self.speed_models
    
    def get_environment_group_stats(self, landcover: int, slope: float) -> Optional[Dict]:
        """
        获取指定环境条件下的统计信息
        
        Args:
            landcover: 土地覆盖类型
            slope: 坡度值
            
        Returns:
            stats: 该环境组的统计信息，如果没有找到则返回None
        """
        # 将坡度值转换为坡度等级
        slope_class = pd.cut(
            [slope], 
            bins=SLOPE_BINS, 
            labels=SLOPE_LABELS, 
            include_lowest=True
        )[0].replace('S', '')  # 移除'S'前缀，只保留数字
        
        # 构建环境组标签
        group_name = f"LC{int(landcover)}_SS{slope_class}"
        
        if group_name not in self.environment_stats:
            logger.warning(f"环境组 {group_name} 没有统计数据")
            return None
            
        return self.environment_stats[group_name]
    
    def sample_speed(self, landcover: int, slope: float) -> float:
        """
        根据环境条件采样合理的速度值
        
        Args:
            landcover: 土地覆盖类型编码
            slope: 坡度值（度）
            
        Returns:
            speed: 采样的速度值（米/秒）
        """
        # 确定坡度等级
        slope_class = pd.cut(
            [slope], 
            bins=SLOPE_BINS, 
            labels=SLOPE_LABELS, 
            include_lowest=True
        )[0].replace('S', '')  # 移除'S'前缀，只保留数字
        
        # 构建环境组标签
        group_name = f"LC{int(landcover)}_SS{slope_class}"
        
        if group_name in self.speed_models:
            model = self.speed_models[group_name]
            dist = getattr(stats, model['distribution'])
            speed = dist.rvs(*model['parameters'])
            
            # 确保速度在合理范围内
            stats = self.environment_stats[group_name]['speed']
            speed = np.clip(speed, stats['min'], stats['max'])
            
            return float(speed)
        else:
            # 如果没有该环境组的模型，返回一个基于统计的速度
            if group_name in self.environment_stats:
                stats = self.environment_stats[group_name]['speed']
                return float(np.random.normal(stats['mean'], stats['std']))
            else:
                logger.warning(f"环境组 {group_name} 没有速度模型和统计数据")
                return 5.0  # 返回一个默认的合理速度值 