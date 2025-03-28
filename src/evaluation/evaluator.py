"""
轨迹评估器
用于评估生成轨迹的质量和真实性
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class Evaluator:
    """轨迹评估器"""
    
    def __init__(
            self,
            oord_data: pd.DataFrame,
            output_dir: str
        ):
        """
        初始化评估器
        
        Args:
            oord_data: OORD轨迹数据，包含：
                - timestamp_ms: 时间戳（毫秒）
                - speed_mps: 速度（米/秒）
                - heading_degrees: 朝向（度）
                - turn_rate_dps: 转向率（度/秒）
                - acceleration_mps2: 加速度（米/秒²）
                - group_label: 环境组标签
            output_dir: 评估结果输出目录
        """
        self.oord_data = oord_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算OORD数据的统计特征
        self._calculate_oord_statistics()
    
    def _calculate_oord_statistics(self) -> None:
        """计算OORD数据的统计特征"""
        # 全局统计
        self.oord_stats = {
            'speed': {
                'mean': self.oord_data['speed_mps'].mean(),
                'std': self.oord_data['speed_mps'].std(),
                'min': self.oord_data['speed_mps'].min(),
                'max': self.oord_data['speed_mps'].max(),
                'percentiles': np.percentile(
                    self.oord_data['speed_mps'],
                    [25, 50, 75]
                )
            },
            'acceleration': {
                'mean': self.oord_data['acceleration_mps2'].mean(),
                'std': self.oord_data['acceleration_mps2'].std(),
                'min': self.oord_data['acceleration_mps2'].min(),
                'max': self.oord_data['acceleration_mps2'].max(),
                'percentiles': np.percentile(
                    self.oord_data['acceleration_mps2'],
                    [25, 50, 75]
                )
            },
            'turn_rate': {
                'mean': self.oord_data['turn_rate_dps'].mean(),
                'std': self.oord_data['turn_rate_dps'].std(),
                'min': self.oord_data['turn_rate_dps'].min(),
                'max': self.oord_data['turn_rate_dps'].max(),
                'percentiles': np.percentile(
                    self.oord_data['turn_rate_dps'],
                    [25, 50, 75]
                )
            }
        }
        
        # 按环境组统计
        self.oord_group_stats = {}
        for group in self.oord_data['group_label'].unique():
            group_data = self.oord_data[
                self.oord_data['group_label'] == group
            ]
            self.oord_group_stats[group] = {
                'speed': {
                    'mean': group_data['speed_mps'].mean(),
                    'std': group_data['speed_mps'].std()
                }
            }
    
    def evaluate_trajectory(
            self,
            trajectory: Dict,
            group_labels: Optional[List[str]] = None
        ) -> Dict:
        """
        评估单条轨迹
        
        Args:
            trajectory: 轨迹数据字典，包含：
                - timestamps: 时间戳列表（秒）
                - speeds: 速度列表（米/秒）
                - headings: 朝向列表（度）
            group_labels: 轨迹点的环境组标签列表，可选
            
        Returns:
            Dict: 评估结果
        """
        # 转换为DataFrame
        traj_df = pd.DataFrame({
            'timestamp': trajectory['timestamps'],
            'speed': trajectory['speeds'],
            'heading': trajectory['headings']
        })
        
        # 计算转向率和加速度
        traj_df['turn_rate'] = np.gradient(
            traj_df['heading'],
            traj_df['timestamp']
        )
        traj_df['acceleration'] = np.gradient(
            traj_df['speed'],
            traj_df['timestamp']
        )
        
        # 计算统计特征
        stats_result = {
            'speed': {
                'mean': traj_df['speed'].mean(),
                'std': traj_df['speed'].std(),
                'min': traj_df['speed'].min(),
                'max': traj_df['speed'].max(),
                'percentiles': np.percentile(
                    traj_df['speed'],
                    [25, 50, 75]
                )
            },
            'acceleration': {
                'mean': traj_df['acceleration'].mean(),
                'std': traj_df['acceleration'].std(),
                'min': traj_df['acceleration'].min(),
                'max': traj_df['acceleration'].max(),
                'percentiles': np.percentile(
                    traj_df['acceleration'],
                    [25, 50, 75]
                )
            },
            'turn_rate': {
                'mean': traj_df['turn_rate'].mean(),
                'std': traj_df['turn_rate'].std(),
                'min': traj_df['turn_rate'].min(),
                'max': traj_df['turn_rate'].max(),
                'percentiles': np.percentile(
                    traj_df['turn_rate'],
                    [25, 50, 75]
                )
            }
        }
        
        # 进行KS检验
        ks_results = {
            'speed': stats.ks_2samp(
                traj_df['speed'],
                self.oord_data['speed_mps']
            ),
            'acceleration': stats.ks_2samp(
                traj_df['acceleration'],
                self.oord_data['acceleration_mps2']
            ),
            'turn_rate': stats.ks_2samp(
                traj_df['turn_rate'],
                self.oord_data['turn_rate_dps']
            )
        }
        
        # 如果提供了环境组标签，进行分组评估
        group_results = {}
        if group_labels is not None:
            traj_df['group_label'] = group_labels
            for group in traj_df['group_label'].unique():
                group_data = traj_df[
                    traj_df['group_label'] == group
                ]
                oord_group_data = self.oord_data[
                    self.oord_data['group_label'] == group
                ]
                
                if len(group_data) > 0 and len(oord_group_data) > 0:
                    group_results[group] = {
                        'speed': {
                            'mean': group_data['speed'].mean(),
                            'std': group_data['speed'].std(),
                            'ks_test': stats.ks_2samp(
                                group_data['speed'],
                                oord_group_data['speed_mps']
                            )
                        }
                    }
        
        return {
            'statistics': stats_result,
            'ks_tests': ks_results,
            'group_results': group_results
        }
    
    def evaluate_batch(
            self,
            trajectories: List[Dict],
            group_labels_list: Optional[List[List[str]]] = None
        ) -> Dict:
        """
        评估一批轨迹
        
        Args:
            trajectories: 轨迹数据字典列表
            group_labels_list: 轨迹点的环境组标签列表的列表，可选
            
        Returns:
            Dict: 评估结果
        """
        # 评估每条轨迹
        results = []
        for i, traj in enumerate(trajectories):
            group_labels = (
                group_labels_list[i]
                if group_labels_list is not None
                else None
            )
            results.append(
                self.evaluate_trajectory(traj, group_labels)
            )
        
        # 汇总统计结果
        speed_means = [r['statistics']['speed']['mean'] for r in results]
        accel_means = [
            r['statistics']['acceleration']['mean']
            for r in results
        ]
        turn_means = [
            r['statistics']['turn_rate']['mean']
            for r in results
        ]
        
        summary = {
            'speed': {
                'mean_of_means': np.mean(speed_means),
                'std_of_means': np.std(speed_means)
            },
            'acceleration': {
                'mean_of_means': np.mean(accel_means),
                'std_of_means': np.std(accel_means)
            },
            'turn_rate': {
                'mean_of_means': np.mean(turn_means),
                'std_of_means': np.std(turn_means)
            }
        }
        
        return {
            'individual_results': results,
            'summary': summary
        }
    
    def plot_distributions(
            self,
            trajectories: List[Dict],
            prefix: str = ''
        ) -> None:
        """
        绘制分布对比图
        
        Args:
            trajectories: 轨迹数据字典列表
            prefix: 输出文件名前缀
        """
        # 合并所有轨迹数据
        all_speeds = []
        all_accels = []
        all_turns = []
        
        for traj in trajectories:
            all_speeds.extend(traj['speeds'])
            
            # 计算加速度和转向率
            times = np.array(traj['timestamps'])
            speeds = np.array(traj['speeds'])
            headings = np.array(traj['headings'])
            
            accels = np.gradient(speeds, times)
            turns = np.gradient(headings, times)
            
            all_accels.extend(accels)
            all_turns.extend(turns)
        
        # 绘制速度分布
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=self.oord_data['speed_mps'],
            label='OORD',
            color='blue'
        )
        sns.kdeplot(
            data=all_speeds,
            label='Generated',
            color='red'
        )
        plt.title('Speed Distribution Comparison')
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(
            self.output_dir / f'{prefix}speed_distribution.png'
        )
        plt.close()
        
        # 绘制加速度分布
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=self.oord_data['acceleration_mps2'],
            label='OORD',
            color='blue'
        )
        sns.kdeplot(
            data=all_accels,
            label='Generated',
            color='red'
        )
        plt.title('Acceleration Distribution Comparison')
        plt.xlabel('Acceleration (m/s²)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(
            self.output_dir / f'{prefix}acceleration_distribution.png'
        )
        plt.close()
        
        # 绘制转向率分布
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=self.oord_data['turn_rate_dps'],
            label='OORD',
            color='blue'
        )
        sns.kdeplot(
            data=all_turns,
            label='Generated',
            color='red'
        )
        plt.title('Turn Rate Distribution Comparison')
        plt.xlabel('Turn Rate (deg/s)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(
            self.output_dir / f'{prefix}turn_rate_distribution.png'
        )
        plt.close()
    
    def generate_report(
            self,
            batch_results: Dict,
            output_file: str
        ) -> None:
        """
        生成评估报告
        
        Args:
            batch_results: 批量评估结果
            output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 轨迹评估报告\n\n")
            
            # 写入总体统计
            f.write("## 总体统计\n\n")
            f.write("### 速度统计\n")
            f.write(f"- OORD平均速度: {self.oord_stats['speed']['mean']:.2f} m/s\n")
            f.write(f"- 生成轨迹平均速度: {batch_results['summary']['speed']['mean_of_means']:.2f} m/s\n")
            f.write(f"- OORD速度标准差: {self.oord_stats['speed']['std']:.2f} m/s\n")
            f.write(f"- 生成轨迹速度标准差: {batch_results['summary']['speed']['std_of_means']:.2f} m/s\n\n")
            
            f.write("### 加速度统计\n")
            f.write(f"- OORD平均加速度: {self.oord_stats['acceleration']['mean']:.2f} m/s²\n")
            f.write(f"- 生成轨迹平均加速度: {batch_results['summary']['acceleration']['mean_of_means']:.2f} m/s²\n")
            f.write(f"- OORD加速度标准差: {self.oord_stats['acceleration']['std']:.2f} m/s²\n")
            f.write(f"- 生成轨迹加速度标准差: {batch_results['summary']['acceleration']['std_of_means']:.2f} m/s²\n\n")
            
            f.write("### 转向率统计\n")
            f.write(f"- OORD平均转向率: {self.oord_stats['turn_rate']['mean']:.2f} deg/s\n")
            f.write(f"- 生成轨迹平均转向率: {batch_results['summary']['turn_rate']['mean_of_means']:.2f} deg/s\n")
            f.write(f"- OORD转向率标准差: {self.oord_stats['turn_rate']['std']:.2f} deg/s\n")
            f.write(f"- 生成轨迹转向率标准差: {batch_results['summary']['turn_rate']['std_of_means']:.2f} deg/s\n\n")
            
            # 写入KS检验结果
            f.write("## KS检验结果\n\n")
            for i, result in enumerate(batch_results['individual_results']):
                f.write(f"### 轨迹 {i+1}\n")
                f.write("- 速度分布检验:\n")
                f.write(f"  - 统计量: {result['ks_tests']['speed'].statistic:.4f}\n")
                f.write(f"  - p值: {result['ks_tests']['speed'].pvalue:.4f}\n")
                f.write("- 加速度分布检验:\n")
                f.write(f"  - 统计量: {result['ks_tests']['acceleration'].statistic:.4f}\n")
                f.write(f"  - p值: {result['ks_tests']['acceleration'].pvalue:.4f}\n")
                f.write("- 转向率分布检验:\n")
                f.write(f"  - 统计量: {result['ks_tests']['turn_rate'].statistic:.4f}\n")
                f.write(f"  - p值: {result['ks_tests']['turn_rate'].pvalue:.4f}\n\n")
            
            # 写入环境组分析结果
            if any(r['group_results'] for r in batch_results['individual_results']):
                f.write("## 环境组分析\n\n")
                for i, result in enumerate(batch_results['individual_results']):
                    if result['group_results']:
                        f.write(f"### 轨迹 {i+1}\n")
                        for group, stats in result['group_results'].items():
                            f.write(f"#### {group}\n")
                            f.write(f"- 平均速度: {stats['speed']['mean']:.2f} m/s\n")
                            f.write(f"- 速度标准差: {stats['speed']['std']:.2f} m/s\n")
                            f.write("- KS检验结果:\n")
                            f.write(f"  - 统计量: {stats['speed']['ks_test'].statistic:.4f}\n")
                            f.write(f"  - p值: {stats['speed']['ks_test'].pvalue:.4f}\n\n") 