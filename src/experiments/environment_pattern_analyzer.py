"""
环境模式分析模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import json
from sklearn.preprocessing import KBinsDiscretizer

logger = logging.getLogger(__name__)

class EnvironmentPatternAnalyzer:
    """环境模式分析器"""
    
    def __init__(self, n_slope_bins: int = 5, n_cross_slope_bins: int = 3):
        """初始化分析器
        
        Args:
            n_slope_bins: 坡度分箱数量
            n_cross_slope_bins: 横坡分箱数量
        """
        self.n_slope_bins = n_slope_bins
        self.n_cross_slope_bins = n_cross_slope_bins
        
        # 初始化分箱器
        self.slope_binner = KBinsDiscretizer(
            n_bins=n_slope_bins,
            encode='ordinal',
            strategy='quantile'
        )
        
        self.cross_slope_binner = KBinsDiscretizer(
            n_bins=n_cross_slope_bins,
            encode='ordinal',
            strategy='quantile'
        )
        
        # 地表类型映射
        self.landcover_mapping = {
            1: 'urban_road',
            2: 'rural_road',
            3: 'trail',
            4: 'rough_terrain',
            0: 'unknown'
        }
        
        logger.info(f"初始化环境模式分析器，坡度分箱: {n_slope_bins}，横坡分箱: {n_cross_slope_bins}")
        
    def fit_binners(self, window_features: pd.DataFrame) -> None:
        """拟合分箱器
        
        Args:
            window_features: 窗口特征DataFrame
        """
        self.slope_binner.fit(window_features[['avg_slope']])
        self.cross_slope_binner.fit(window_features[['avg_cross_slope']])
        
        # 获取分箱边界
        self.slope_bins = self.slope_binner.bin_edges_[0]
        self.cross_slope_bins = self.cross_slope_binner.bin_edges_[0]
        
        logger.info(f"坡度分箱边界: {self.slope_bins}")
        logger.info(f"横坡分箱边界: {self.cross_slope_bins}")
        
    def create_environment_group(self, slope: float, cross_slope: float,
                               landcover: int) -> str:
        """创建环境特征组合标签
        
        Args:
            slope: 平均坡度
            cross_slope: 平均横坡
            landcover: 地表类型
            
        Returns:
            环境组合标签
        """
        # 获取坡度和横坡的分箱
        slope_bin = np.digitize(slope, self.slope_bins) - 1
        cross_slope_bin = np.digitize(abs(cross_slope), self.cross_slope_bins) - 1
        
        # 获取地表类型标签
        landcover_label = self.landcover_mapping.get(landcover, 'unknown')
        
        return f"LC{landcover_label}_S{slope_bin}_CS{cross_slope_bin}"
        
    def analyze_patterns(self, window_features: pd.DataFrame) -> Dict:
        """分析环境-速度关系模式
        
        Args:
            window_features: 窗口特征DataFrame
            
        Returns:
            环境模式字典
        """
        # 拟合分箱器
        self.fit_binners(window_features)
        
        # 创建环境组标签
        env_groups = []
        for _, row in window_features.iterrows():
            group = self.create_environment_group(
                slope=row['avg_slope'],
                cross_slope=row['avg_cross_slope'],
                landcover=row['dominant_landcover']
            )
            env_groups.append(group)
            
        window_features['env_group'] = env_groups
        
        # 分析每个环境组合的速度特征
        patterns = {}
        for group in window_features['env_group'].unique():
            group_data = window_features[window_features['env_group'] == group]
            
            if len(group_data) < 5:  # 样本太少的组合跳过
                logger.warning(f"环境组 {group} 样本数量不足: {len(group_data)}")
                continue
                
            patterns[group] = {
                'mean_speed': float(group_data['avg_speed'].mean()),
                'std_speed': float(group_data['avg_speed'].std()),
                'min_speed': float(group_data['avg_speed'].min()),
                'max_speed': float(group_data['avg_speed'].max()),
                'sample_count': int(len(group_data)),
                
                # 环境特征统计
                'slope_stats': {
                    'mean': float(group_data['avg_slope'].mean()),
                    'std': float(group_data['avg_slope'].std()),
                    'min': float(group_data['avg_slope'].min()),
                    'max': float(group_data['avg_slope'].max())
                },
                'cross_slope_stats': {
                    'mean': float(group_data['avg_cross_slope'].mean()),
                    'std': float(group_data['avg_cross_slope'].std()),
                    'min': float(group_data['avg_cross_slope'].min()),
                    'max': float(group_data['avg_cross_slope'].max())
                },
                
                # 速度-坡度相关性
                'speed_slope_correlation': float(
                    group_data[['avg_speed', 'avg_slope']].corr().iloc[0,1]
                )
            }
            
        logger.info(f"分析了 {len(patterns)} 个有效环境组合")
        return patterns
        
    def save_patterns(self, patterns: Dict, output_file: str) -> None:
        """保存环境模式
        
        Args:
            patterns: 环境模式字典
            output_file: 输出文件路径
        """
        with open(output_file, 'w') as f:
            json.dump(patterns, f, indent=2)
        logger.info(f"环境模式已保存到: {output_file}")
        
    def load_patterns(self, input_file: str) -> Dict:
        """加载环境模式
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            环境模式字典
        """
        with open(input_file, 'r') as f:
            patterns = json.load(f)
        logger.info(f"从 {input_file} 加载了 {len(patterns)} 个环境模式")
        return patterns 