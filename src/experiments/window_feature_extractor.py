"""
基于30秒窗口的特征提取模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class WindowFeatureExtractor:
    """30秒窗口特征提取器"""
    
    def __init__(self, window_size_seconds: int = 30):
        """初始化特征提取器
        
        Args:
            window_size_seconds: 窗口大小（秒）
        """
        self.window_size = window_size_seconds
        logger.info(f"初始化窗口特征提取器，窗口大小: {window_size_seconds}秒")
        
    def extract_window_features(self, trajectory_df: pd.DataFrame, 
                              env_features: pd.DataFrame) -> pd.DataFrame:
        """提取30秒窗口的平均特征
        
        Args:
            trajectory_df: 包含轨迹数据的DataFrame
            env_features: 环境特征DataFrame
            
        Returns:
            包含窗口化特征的DataFrame
        """
        window_features = []
        
        # 确保时间戳是datetime格式
        timestamps = pd.to_datetime(trajectory_df['timestamp_ms'], unit='ms')
        start_time = timestamps.iloc[0]
        end_time = timestamps.iloc[-1]
        
        # 按30秒窗口滑动
        current_time = start_time
        window_count = 0
        
        while current_time + pd.Timedelta(seconds=self.window_size) <= end_time:
            window_end = current_time + pd.Timedelta(seconds=self.window_size)
            
            # 获取当前窗口的数据
            window_mask = (timestamps >= current_time) & (timestamps < window_end)
            window_data = trajectory_df[window_mask]
            window_env = env_features[window_mask]
            
            if len(window_data) > 0:
                # 计算窗口内的平均特征
                feature_dict = {
                    'window_id': window_count,
                    'start_time': current_time,
                    'end_time': window_end,
                    
                    # 位置特征
                    'center_lat': window_data['latitude'].mean(),
                    'center_lon': window_data['longitude'].mean(),
                    'start_lat': window_data['latitude'].iloc[0],
                    'start_lon': window_data['longitude'].iloc[0],
                    'end_lat': window_data['latitude'].iloc[-1],
                    'end_lon': window_data['longitude'].iloc[-1],
                    
                    # 运动特征
                    'avg_speed': window_env['speed_mps'].mean(),
                    'speed_std': window_env['speed_mps'].std(),
                    'min_speed': window_env['speed_mps'].min(),
                    'max_speed': window_env['speed_mps'].max(),
                    'distance': window_data['distance_m'].sum(),
                    
                    # 环境特征
                    'avg_slope': window_env['slope_along_path'].mean(),
                    'slope_std': window_env['slope_along_path'].std(),
                    'max_slope': window_env['slope_along_path'].max(),
                    'min_slope': window_env['slope_along_path'].min(),
                    
                    'avg_cross_slope': window_env['cross_slope'].mean(),
                    'cross_slope_std': window_env['cross_slope'].std(),
                    'max_cross_slope': window_env['cross_slope'].max(),
                    'min_cross_slope': window_env['cross_slope'].min(),
                    
                    # 使用众数作为窗口的地表类型
                    'dominant_landcover': window_env['landcover'].mode().iloc[0],
                    
                    # 样本数量
                    'sample_count': len(window_data)
                }
                
                window_features.append(feature_dict)
                window_count += 1
            
            # 滑动窗口
            current_time += pd.Timedelta(seconds=self.window_size)
            
        result_df = pd.DataFrame(window_features)
        logger.info(f"提取了 {len(result_df)} 个窗口特征")
        
        return result_df
        
    def extract_validation_windows(self, trajectory_df: pd.DataFrame,
                                 env_features: pd.DataFrame,
                                 predicted_speeds: np.ndarray) -> pd.DataFrame:
        """提取用于验证的窗口特征
        
        Args:
            trajectory_df: 轨迹数据
            env_features: 环境特征
            predicted_speeds: 预测的速度数组
            
        Returns:
            包含验证信息的DataFrame
        """
        # 提取窗口特征
        window_features = self.extract_window_features(trajectory_df, env_features)
        
        # 添加预测速度
        window_features['predicted_speed'] = predicted_speeds
        
        # 计算误差
        window_features['speed_error'] = (
            window_features['predicted_speed'] - window_features['avg_speed']
        )
        window_features['speed_error_percent'] = (
            window_features['speed_error'] / window_features['avg_speed'] * 100
        )
        
        return window_features 