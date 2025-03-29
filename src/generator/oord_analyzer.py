"""OORD数据分析器模块

此模块负责分析OORD轨迹数据中的运动特性与环境交互，功能包括：
1. 关联环境信息（高程、坡度、坡向、土地覆盖）
2. 定义环境分组
3. 分组统计分析
4. 建立环境-运动规则模型

输入:
    - 预处理后的OORD轨迹数据
    - 环境数据（DEM、坡度、坡向、土地覆盖）

输出:
    - 环境-运动规则模型
    - 统计分析结果
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from sklearn.preprocessing import KBinsDiscretizer

# 从统一配置文件导入配置
from config import config

@dataclass
class EnvironmentGroup:
    """环境组数据类"""
    group_label: str  # 环境组标签
    landcover_code: int  # 土地覆盖编码
    slope_bin: int  # 坡度等级
    count: int  # 样本数量
    max_speed: float  # 最大速度
    typical_speed: float  # 典型速度
    speed_stddev: float  # 速度标准差
    max_turn_rate: float  # 最大转向率
    typical_turn_rate: float  # 典型转向率
    max_acceleration: float  # 最大加速度
    typical_acceleration: float  # 典型加速度

class OORDAnalyzer:
    """OORD数据分析器类"""
    
    def __init__(self, slope_bins: List[float] = None, min_samples_per_group: int = 100):
        """初始化分析器
        
        Args:
            slope_bins: 坡度分组边界值列表
            min_samples_per_group: 每个环境组的最小样本数
        """
        self.slope_bins = slope_bins or config['terrain']['SLOPE_BINS']
        self.min_samples_per_group = min_samples_per_group
        self.logger = logging.getLogger(__name__)
        self.environment_groups: Dict[str, EnvironmentGroup] = {}
    
    def add_environment_info(self, df: pd.DataFrame,
                           dem_data: np.ndarray,
                           slope_data: np.ndarray,
                           aspect_data: np.ndarray,
                           landcover_data: np.ndarray) -> pd.DataFrame:
        """添加环境信息到轨迹数据
        
        Args:
            df: 预处理后的轨迹数据
            dem_data: DEM数据
            slope_data: 坡度数据
            aspect_data: 坡向数据
            landcover_data: 土地覆盖数据
            
        Returns:
            pd.DataFrame: 添加环境信息后的数据
            
        Raises:
            ValueError: 当数据缺少必要的列或数据形状不匹配时
        """
        # 检查必要的列
        required_cols = ['row', 'col']
        if 'heading_degrees' in df.columns:
            required_cols.append('heading_degrees')
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要的列: {missing_cols}")
            
        # 检查数据形状是否一致
        shape = dem_data.shape
        if (slope_data.shape != shape or
            aspect_data.shape != shape or
            landcover_data.shape != shape):
            raise ValueError(f"所有环境数据的形状必须一致。DEM形状: {shape}, "
                          f"坡度形状: {slope_data.shape}, "
                          f"坡向形状: {aspect_data.shape}, "
                          f"土地覆盖形状: {landcover_data.shape}")
        
        # 检查像素坐标是否在范围内
        if (df['row'].max() >= shape[0] or
            df['col'].max() >= shape[1] or
            df['row'].min() < 0 or
            df['col'].min() < 0):
            raise ValueError(f"像素坐标超出范围。数据形状: {shape}, "
                          f"行范围: [{df['row'].min()}, {df['row'].max()}], "
                          f"列范围: [{df['col'].min()}, {df['col'].max()}]")
        
        # 使用像素坐标获取环境数据
        df['elevation'] = dem_data[df['row'].values, df['col'].values]
        df['slope_magnitude'] = slope_data[df['row'].values, df['col'].values]
        df['slope_aspect'] = aspect_data[df['row'].values, df['col'].values]
        df['landcover'] = landcover_data[df['row'].values, df['col'].values]
        
        # 如果有朝向数据，计算方向性坡度
        if 'heading_degrees' in df.columns:
            # 计算行驶方向与坡向的关系
            df['delta_angle'] = df['heading_degrees'] - df['slope_aspect']
            # 处理角度环绕
            df.loc[df['delta_angle'] > 180, 'delta_angle'] -= 360
            df.loc[df['delta_angle'] < -180, 'delta_angle'] += 360
            
            # 计算方向性坡度
            df['slope_along_path'] = df['slope_magnitude'] * np.cos(np.radians(df['delta_angle']))
            df['cross_slope'] = df['slope_magnitude'] * np.abs(np.sin(np.radians(df['delta_angle'])))
        
        # 添加坡度等级
        df['slope_bin'] = pd.cut(df['slope_magnitude'],
                               bins=self.slope_bins,
                               labels=range(len(self.slope_bins)-1))
        
        # 创建环境组标签
        df['group_label'] = df.apply(
            lambda x: f"LC{x['landcover']}_S{x['slope_bin']}",
            axis=1
        )
        
        return df
    
    def analyze_groups(self, df: pd.DataFrame) -> None:
        """对每个环境组进行统计分析
        
        Args:
            df: 包含环境信息的轨迹数据
        """
        # 检查必要的列
        required_cols = ['speed_mps', 'turn_rate_dps', 'acceleration_mps2']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"数据缺少必要的列: {required_cols}")
        
        # 按环境组分组
        groups = df.groupby('group_label')
        
        # 分析每个组
        for group_label, group_data in groups:
            # 检查样本数量
            if len(group_data) < self.min_samples_per_group:
                self.logger.warning(f"环境组 {group_label} 样本数量不足: {len(group_data)} < {self.min_samples_per_group}")
                # 即使样本不足，也尝试使用
                
            # 提取landcover和slope_bin
            landcover_code = int(group_data['landcover'].iloc[0])  # 转换为Python int
            slope_bin = int(group_data['slope_bin'].iloc[0])  # 转换为Python int
            
            # 创建环境组对象
            self.environment_groups[group_label] = EnvironmentGroup(
                group_label=group_label,
                landcover_code=landcover_code,
                slope_bin=slope_bin,
                count=int(len(group_data)),  # 转换为Python int
                max_speed=float(np.percentile(group_data['speed_mps'], 95)),
                typical_speed=float(group_data['speed_mps'].median()),
                speed_stddev=float(group_data['speed_mps'].std()),
                max_turn_rate=float(np.percentile(abs(group_data['turn_rate_dps']), 95)),
                typical_turn_rate=float(group_data['turn_rate_dps'].median()),
                max_acceleration=float(np.percentile(abs(group_data['acceleration_mps2']), 95)),
                typical_acceleration=float(group_data['acceleration_mps2'].median())
            )
            
            # 记录环境组统计信息
            self.logger.info(f"环境组 {group_label}: "
                          f"样本数={len(group_data)}, "
                          f"典型速度={self.environment_groups[group_label].typical_speed:.2f}m/s, "
                          f"最大速度={self.environment_groups[group_label].max_speed:.2f}m/s, "
                          f"速度标准差={self.environment_groups[group_label].speed_stddev:.2f}m/s")
    
    def analyze_slope_direction_effect(self, df: pd.DataFrame) -> Dict:
        """分析坡向对速度的影响
        
        Args:
            df: 包含环境信息的轨迹数据
            
        Returns:
            Dict: 坡向影响系数
        """
        # 检查必要的列
        required_cols = ['slope_along_path', 'cross_slope', 'speed_mps']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"数据缺少必要的列: {required_cols}")
        
        # 初始化结果字典
        effect_params = {}
        
        # 按landcover分组分析
        for landcover in df['landcover'].unique():
            lc_data = df[df['landcover'] == landcover]
            
            # 上坡影响（slope_along_path > 0）
            uphill_data = lc_data[lc_data['slope_along_path'] > 0]
            if len(uphill_data) > self.min_samples_per_group:
                # 使用线性回归分析上坡减速效应
                from sklearn.linear_model import LinearRegression
                X = uphill_data['slope_along_path'].values.reshape(-1, 1)
                y = uphill_data['speed_mps'].values
                reg = LinearRegression().fit(X, y)
                k_uphill = max(0.1, abs(reg.coef_[0]))  # 确保上坡减速系数为正
            else:
                k_uphill = 0.1  # 默认值
            
            # 横坡影响
            # 使用二次回归分析横坡减速效应
            from sklearn.preprocessing import PolynomialFeatures
            cross_slope_data = lc_data[lc_data['cross_slope'] > 0]
            if len(cross_slope_data) > self.min_samples_per_group:
                X = cross_slope_data['cross_slope'].values.reshape(-1, 1)
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                y = cross_slope_data['speed_mps'].values
                reg = LinearRegression().fit(X_poly, y)
                k_cross = max(0.05, abs(reg.coef_[2]))  # 确保横坡减速系数为正
            else:
                k_cross = 0.05  # 默认值
            
            effect_params[str(int(landcover))] = {  # 转换landcover为字符串
                'k_uphill': float(k_uphill),  # 转换为Python float
                'k_cross': float(k_cross),  # 转换为Python float
                'max_cross_slope_degrees': 30.0  # 默认值
            }
        
        return effect_params
    
    def save_analysis_results(self, output_path: str, effect_params: Optional[Dict] = None) -> None:
        """保存分析结果
        
        Args:
            output_path: 输出文件路径
            effect_params: 可选的坡向影响系数，如果为None则只保存环境组数据
        """
        # 确保输出目录存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 将环境组数据转换为字典
        groups_dict = {
            label: {
                'landcover_code': group.landcover_code,
                'slope_bin': group.slope_bin,
                'count': group.count,
                'max_speed': group.max_speed,
                'typical_speed': group.typical_speed,
                'speed_stddev': group.speed_stddev,
                'max_turn_rate': group.max_turn_rate,
                'typical_turn_rate': group.typical_turn_rate,
                'max_acceleration': group.max_acceleration,
                'typical_acceleration': group.typical_acceleration
            }
            for label, group in self.environment_groups.items()
        }
        
        # 合并环境组数据和坡向影响系数（如果提供）
        result_dict = {
            'environment_groups': groups_dict
        }
        
        if effect_params:
            result_dict['slope_direction_effects'] = effect_params
        
        # 添加元数据
        result_dict['metadata'] = {
            'num_groups': len(groups_dict),
            'min_samples_per_group': self.min_samples_per_group,
            'slope_bins': self.slope_bins
        }
        
        # 保存为JSON文件
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"分析结果已保存至: {output_path}")
        self.logger.info(f"共保存了 {len(groups_dict)} 个环境组的数据")
        
        # 输出每个环境组的基本信息 (前5个)
        top_groups = list(groups_dict.items())[:5]
        for label, data in top_groups:
            self.logger.info(f"环境组 {label}: 典型速度={data['typical_speed']:.2f}m/s, "
                          f"最大速度={data['max_speed']:.2f}m/s, "
                          f"速度标准差={data['speed_stddev']:.2f}m/s")
    
    def save_environment_groups(self, output_path: str) -> None:
        """仅保存环境组数据
        
        这是一个便捷的包装方法，用于快速保存环境组数据
        
        Args:
            output_path: 输出文件路径
        """
        self.save_analysis_results(output_path) 