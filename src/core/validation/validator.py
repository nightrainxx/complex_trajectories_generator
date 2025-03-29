"""
轨迹验证器模块
实现真实轨迹和模拟轨迹的对比分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.interpolate import interp1d

from src.core.motion.simulator import TrajectoryPoint
from src.utils.visualization import plot_trajectory_on_map
from src.core.terrain.loader import convert_to_pixel_coords, TerrainLoader

@dataclass
class ValidationMetrics:
    """验证指标"""
    # 速度相关
    speed_rmse: float
    speed_correlation: float
    speed_ks_statistic: float
    speed_ks_pvalue: float
    
    # 加速度相关
    accel_rmse: float
    accel_correlation: float
    accel_ks_statistic: float
    accel_ks_pvalue: float
    
    # 转向率相关
    turn_rate_rmse: float
    turn_rate_correlation: float
    turn_rate_ks_statistic: float
    turn_rate_ks_pvalue: float
    
    # 全局指标
    total_time_diff: float  # 总时间差异（秒）
    total_distance_diff: float  # 总距离差异（米）
    avg_speed_diff: float  # 平均速度差异（米/秒）

class TrajectoryValidator:
    """轨迹验证器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化验证器
        
        Args:
            output_dir: 输出目录，用于保存验证结果和图表
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _interpolate_trajectory(
            self,
            trajectory: pd.DataFrame,
            target_timestamps: np.ndarray
        ) -> pd.DataFrame:
        """
        将轨迹插值到目标时间点
        
        Args:
            trajectory: 轨迹数据框
            target_timestamps: 目标时间点数组
            
        Returns:
            pd.DataFrame: 插值后的轨迹数据框
        """
        # 创建插值器
        interp_speed = interp1d(
            trajectory['timestamp'],
            trajectory['speed_mps'],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        interp_accel = interp1d(
            trajectory['timestamp'],
            trajectory['acceleration_mps2'],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        interp_turn_rate = interp1d(
            trajectory['timestamp'],
            trajectory['turn_rate_dps'],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # 执行插值
        return pd.DataFrame({
            'timestamp': target_timestamps,
            'speed_mps': interp_speed(target_timestamps),
            'acceleration_mps2': interp_accel(target_timestamps),
            'turn_rate_dps': interp_turn_rate(target_timestamps)
        })
        
    def _calculate_metrics(
            self,
            real_df: pd.DataFrame,
            sim_df: pd.DataFrame
        ) -> ValidationMetrics:
        """
        计算验证指标
        
        Args:
            real_df: 真实轨迹数据框
            sim_df: 模拟轨迹数据框
            
        Returns:
            ValidationMetrics: 验证指标
        """
        # 速度指标
        speed_rmse = np.sqrt(np.mean(
            (real_df['speed_mps'] - sim_df['speed_mps'])**2
        ))
        speed_corr = np.corrcoef(
            real_df['speed_mps'],
            sim_df['speed_mps']
        )[0, 1]
        speed_ks = stats.ks_2samp(
            real_df['speed_mps'],
            sim_df['speed_mps']
        )
        
        # 加速度指标
        accel_rmse = np.sqrt(np.mean(
            (real_df['acceleration_mps2'] - 
             sim_df['acceleration_mps2'])**2
        ))
        accel_corr = np.corrcoef(
            real_df['acceleration_mps2'],
            sim_df['acceleration_mps2']
        )[0, 1]
        accel_ks = stats.ks_2samp(
            real_df['acceleration_mps2'],
            sim_df['acceleration_mps2']
        )
        
        # 转向率指标
        turn_rate_rmse = np.sqrt(np.mean(
            (real_df['turn_rate_dps'] - 
             sim_df['turn_rate_dps'])**2
        ))
        turn_rate_corr = np.corrcoef(
            real_df['turn_rate_dps'],
            sim_df['turn_rate_dps']
        )[0, 1]
        turn_rate_ks = stats.ks_2samp(
            real_df['turn_rate_dps'],
            sim_df['turn_rate_dps']
        )
        
        # 全局指标
        total_time_diff = (
            sim_df['timestamp'].max() - 
            real_df['timestamp'].max()
        )
        
        total_distance_diff = (
            np.sum(sim_df['speed_mps'] * 0.1) -  # dt = 0.1
            np.sum(real_df['speed_mps'] * 0.1)
        )
        
        avg_speed_diff = (
            sim_df['speed_mps'].mean() - 
            real_df['speed_mps'].mean()
        )
        
        return ValidationMetrics(
            speed_rmse=speed_rmse,
            speed_correlation=speed_corr,
            speed_ks_statistic=speed_ks.statistic,
            speed_ks_pvalue=speed_ks.pvalue,
            
            accel_rmse=accel_rmse,
            accel_correlation=accel_corr,
            accel_ks_statistic=accel_ks.statistic,
            accel_ks_pvalue=accel_ks.pvalue,
            
            turn_rate_rmse=turn_rate_rmse,
            turn_rate_correlation=turn_rate_corr,
            turn_rate_ks_statistic=turn_rate_ks.statistic,
            turn_rate_ks_pvalue=turn_rate_ks.pvalue,
            
            total_time_diff=total_time_diff,
            total_distance_diff=total_distance_diff,
            avg_speed_diff=avg_speed_diff
        )
        
    def _plot_time_series(
            self,
            real_df: pd.DataFrame,
            sim_df: pd.DataFrame,
            output_prefix: str
        ) -> None:
        """
        绘制时间序列对比图
        
        Args:
            real_df: 真实轨迹数据框
            sim_df: 模拟轨迹数据框
            output_prefix: 输出文件前缀
        """
        # 速度对比图
        plt.figure(figsize=(12, 6))
        plt.plot(
            real_df['timestamp'],
            real_df['speed_mps'],
            'b-',
            label='真实轨迹'
        )
        plt.plot(
            sim_df['timestamp'],
            sim_df['speed_mps'],
            'r--',
            label='模拟轨迹'
        )
        plt.xlabel('时间 (秒)')
        plt.ylabel('速度 (米/秒)')
        plt.title('速度随时间变化对比')
        plt.legend()
        plt.grid(True)
        plt.savefig(
            self.output_dir / f'{output_prefix}_speed.png'
        )
        plt.close()
        
        # 加速度对比图
        plt.figure(figsize=(12, 6))
        plt.plot(
            real_df['timestamp'],
            real_df['acceleration_mps2'],
            'b-',
            label='真实轨迹'
        )
        plt.plot(
            sim_df['timestamp'],
            sim_df['acceleration_mps2'],
            'r--',
            label='模拟轨迹'
        )
        plt.xlabel('时间 (秒)')
        plt.ylabel('加速度 (米/秒²)')
        plt.title('加速度随时间变化对比')
        plt.legend()
        plt.grid(True)
        plt.savefig(
            self.output_dir / f'{output_prefix}_acceleration.png'
        )
        plt.close()
        
        # 转向率对比图
        plt.figure(figsize=(12, 6))
        plt.plot(
            real_df['timestamp'],
            real_df['turn_rate_dps'],
            'b-',
            label='真实轨迹'
        )
        plt.plot(
            sim_df['timestamp'],
            sim_df['turn_rate_dps'],
            'r--',
            label='模拟轨迹'
        )
        plt.xlabel('时间 (秒)')
        plt.ylabel('转向率 (度/秒)')
        plt.title('转向率随时间变化对比')
        plt.legend()
        plt.grid(True)
        plt.savefig(
            self.output_dir / f'{output_prefix}_turn_rate.png'
        )
        plt.close()
        
    def _plot_distributions(
            self,
            real_df: pd.DataFrame,
            sim_df: pd.DataFrame,
            output_prefix: str
        ) -> None:
        """
        绘制分布对比图
        
        Args:
            real_df: 真实轨迹数据框
            sim_df: 模拟轨迹数据框
            output_prefix: 输出文件前缀
        """
        # 速度分布
        plt.figure(figsize=(10, 6))
        plt.hist(
            real_df['speed_mps'],
            bins=30,
            alpha=0.5,
            density=True,
            label='真实轨迹'
        )
        plt.hist(
            sim_df['speed_mps'],
            bins=30,
            alpha=0.5,
            density=True,
            label='模拟轨迹'
        )
        plt.xlabel('速度 (米/秒)')
        plt.ylabel('密度')
        plt.title('速度分布对比')
        plt.legend()
        plt.grid(True)
        plt.savefig(
            self.output_dir / f'{output_prefix}_speed_dist.png'
        )
        plt.close()
        
        # 加速度分布
        plt.figure(figsize=(10, 6))
        plt.hist(
            real_df['acceleration_mps2'],
            bins=30,
            alpha=0.5,
            density=True,
            label='真实轨迹'
        )
        plt.hist(
            sim_df['acceleration_mps2'],
            bins=30,
            alpha=0.5,
            density=True,
            label='模拟轨迹'
        )
        plt.xlabel('加速度 (米/秒²)')
        plt.ylabel('密度')
        plt.title('加速度分布对比')
        plt.legend()
        plt.grid(True)
        plt.savefig(
            self.output_dir / f'{output_prefix}_acceleration_dist.png'
        )
        plt.close()
        
        # 转向率分布
        plt.figure(figsize=(10, 6))
        plt.hist(
            real_df['turn_rate_dps'],
            bins=30,
            alpha=0.5,
            density=True,
            label='真实轨迹'
        )
        plt.hist(
            sim_df['turn_rate_dps'],
            bins=30,
            alpha=0.5,
            density=True,
            label='模拟轨迹'
        )
        plt.xlabel('转向率 (度/秒)')
        plt.ylabel('密度')
        plt.title('转向率分布对比')
        plt.legend()
        plt.grid(True)
        plt.savefig(
            self.output_dir / f'{output_prefix}_turn_rate_dist.png'
        )
        plt.close()
        
    def validate_trajectory(
            self,
            real_trajectory: pd.DataFrame,
            sim_trajectory: List[TrajectoryPoint],
            dem_data: Optional[TerrainLoader] = None,
            output_prefix: str = 'validation'
        ) -> ValidationMetrics:
        """验证模拟轨迹与真实轨迹的差异。

        Args:
            real_trajectory: 真实轨迹数据，包含timestamp、longitude、latitude、speed_mps等列
            sim_trajectory: 模拟轨迹点列表
            dem_data: 地形加载器对象，用于绘制轨迹叠加图
            output_prefix: 输出文件前缀

        Returns:
            ValidationMetrics: 验证指标
        """
        # 将模拟轨迹转换为DataFrame
        sim_df = pd.DataFrame([
            {
                'timestamp': p.timestamp,
                'longitude': p.lon,
                'latitude': p.lat,
                'speed_mps': p.speed,
                'acceleration_mps2': p.acceleration,
                'turn_rate_dps': p.turn_rate
            }
            for p in sim_trajectory
        ])

        # 将经纬度转换为行列坐标
        if dem_data is not None:
            real_coords = convert_to_pixel_coords(
                real_trajectory[['longitude', 'latitude']].values,
                dem_data.transform
            )
            sim_coords = convert_to_pixel_coords(
                sim_df[['longitude', 'latitude']].values,
                dem_data.transform
            )
            real_trajectory['row'] = real_coords[:, 0]
            real_trajectory['col'] = real_coords[:, 1]
            sim_df['row'] = sim_coords[:, 0]
            sim_df['col'] = sim_coords[:, 1]

        # 对齐时间戳
        target_timestamps = np.union1d(
            real_trajectory['timestamp'].values,
            sim_df['timestamp'].values
        )
        real_interp = self._interpolate_trajectory(real_trajectory, target_timestamps)
        sim_interp = self._interpolate_trajectory(sim_df, target_timestamps)
        
        # 计算验证指标
        metrics = self._calculate_metrics(real_interp, sim_interp)
        
        # 绘制对比图
        self._plot_time_series(real_interp, sim_interp, output_prefix)
        self._plot_distributions(real_interp, sim_interp, output_prefix)
        
        # 如果提供了DEM数据，绘制轨迹叠加图
        if dem_data is not None:
            plot_trajectory_on_map(
                dem_data.dem_data,
                real_trajectory[['row', 'col']].values,
                sim_df[['row', 'col']].values,
                self.output_dir / f'{output_prefix}_map.png'
            )
            
        # 保存验证指标
        metrics_dict = {
            'speed': {
                'rmse': metrics.speed_rmse,
                'correlation': metrics.speed_correlation,
                'ks_statistic': metrics.speed_ks_statistic,
                'ks_pvalue': metrics.speed_ks_pvalue
            },
            'acceleration': {
                'rmse': metrics.accel_rmse,
                'correlation': metrics.accel_correlation,
                'ks_statistic': metrics.accel_ks_statistic,
                'ks_pvalue': metrics.accel_ks_pvalue
            },
            'turn_rate': {
                'rmse': metrics.turn_rate_rmse,
                'correlation': metrics.turn_rate_correlation,
                'ks_statistic': metrics.turn_rate_ks_statistic,
                'ks_pvalue': metrics.turn_rate_ks_pvalue
            },
            'global': {
                'total_time_diff': metrics.total_time_diff,
                'total_distance_diff': metrics.total_distance_diff,
                'avg_speed_diff': metrics.avg_speed_diff
            }
        }
        
        with open(
            self.output_dir / f'{output_prefix}_metrics.json', 'w'
        ) as f:
            import json
            json.dump(metrics_dict, f, indent=2)
            
        return metrics 