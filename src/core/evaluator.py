"""
轨迹评估器模块
用于评估生成轨迹的质量和真实性
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import config

logger = logging.getLogger(__name__)

class Evaluator:
    """轨迹评估器"""
    
    def __init__(
            self,
            output_dir: Path = config.paths.EVALUATION_DIR
        ):
        """
        初始化评估器
        
        Args:
            output_dir: 评估结果输出目录
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储数据
        self.oord_data: Optional[pd.DataFrame] = None
        self.synthetic_data: Optional[pd.DataFrame] = None
        
    def load_data(
            self,
            oord_file: Path,
            synthetic_dir: Path
        ) -> None:
        """
        加载OORD数据和合成轨迹数据
        
        Args:
            oord_file: OORD数据文件路径
            synthetic_dir: 合成轨迹目录路径
        """
        logger.info("加载数据...")
        
        # 加载OORD数据
        self.oord_data = pd.read_csv(oord_file)
        logger.info(f"已加载{len(self.oord_data)}条OORD记录")
        
        # 加载合成轨迹
        synthetic_trajectories = []
        for file_path in synthetic_dir.glob('trajectory_*.json'):
            with open(file_path) as f:
                trajectory = json.load(f)
                
            # 转换为DataFrame格式
            df = pd.DataFrame({
                'timestamp': trajectory['timestamp'],
                'x': trajectory['x'],
                'y': trajectory['y'],
                'speed': trajectory['speed'],
                'orientation': trajectory['orientation']
            })
            
            # 添加轨迹ID
            df['trajectory_id'] = trajectory['metadata']['index']
            
            synthetic_trajectories.append(df)
            
        if not synthetic_trajectories:
            raise ValueError(f"未在{synthetic_dir}找到轨迹文件")
            
        self.synthetic_data = pd.concat(
            synthetic_trajectories,
            ignore_index=True
        )
        logger.info(f"已加载{len(synthetic_trajectories)}条合成轨迹")
        
    def evaluate(self) -> Dict[str, float]:
        """
        执行评估
        
        Returns:
            Dict[str, float]: 评估指标
        """
        if self.oord_data is None or self.synthetic_data is None:
            raise ValueError("请先加载数据")
            
        logger.info("开始评估...")
        
        # 计算统计指标
        metrics = {}
        
        # 速度分布比较
        metrics.update(self._compare_speed_distributions())
        
        # 加速度分布比较
        metrics.update(self._compare_acceleration_distributions())
        
        # 转向率分布比较
        metrics.update(self._compare_turn_rate_distributions())
        
        # 环境交互比较
        if 'group_label' in self.oord_data.columns:
            metrics.update(self._compare_environment_interaction())
            
        # 生成评估报告
        self._generate_report(metrics)
        
        logger.info("评估完成")
        return metrics
        
    def _compare_speed_distributions(self) -> Dict[str, float]:
        """
        比较速度分布
        
        Returns:
            Dict[str, float]: 速度相关指标
        """
        oord_speeds = self.oord_data['speed']
        synthetic_speeds = self.synthetic_data['speed']
        
        # 计算KS检验
        ks_stat, p_value = stats.ks_2samp(
            oord_speeds,
            synthetic_speeds
        )
        
        # 计算基本统计量
        metrics = {
            'speed_ks_stat': ks_stat,
            'speed_ks_p_value': p_value,
            'speed_mean_diff': abs(
                oord_speeds.mean() - synthetic_speeds.mean()
            ),
            'speed_std_diff': abs(
                oord_speeds.std() - synthetic_speeds.std()
            )
        }
        
        # 绘制分布对比图
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=oord_speeds,
            label='OORD',
            color='blue',
            alpha=0.5
        )
        sns.kdeplot(
            data=synthetic_speeds,
            label='Synthetic',
            color='red',
            alpha=0.5
        )
        plt.title('速度分布对比')
        plt.xlabel('速度 (m/s)')
        plt.ylabel('密度')
        plt.legend()
        plt.savefig(self.output_dir / 'speed_distribution.png')
        plt.close()
        
        return metrics
        
    def _compare_acceleration_distributions(self) -> Dict[str, float]:
        """
        比较加速度分布
        
        Returns:
            Dict[str, float]: 加速度相关指标
        """
        # 计算加速度
        oord_acc = np.diff(self.oord_data['speed']) / \
                   np.diff(self.oord_data['timestamp'])
        synthetic_acc = np.diff(self.synthetic_data['speed']) / \
                       np.diff(self.synthetic_data['timestamp'])
        
        # 计算KS检验
        ks_stat, p_value = stats.ks_2samp(
            oord_acc,
            synthetic_acc
        )
        
        # 计算基本统计量
        metrics = {
            'acceleration_ks_stat': ks_stat,
            'acceleration_ks_p_value': p_value,
            'acceleration_mean_diff': abs(
                np.mean(oord_acc) - np.mean(synthetic_acc)
            ),
            'acceleration_std_diff': abs(
                np.std(oord_acc) - np.std(synthetic_acc)
            )
        }
        
        # 绘制分布对比图
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=oord_acc,
            label='OORD',
            color='blue',
            alpha=0.5
        )
        sns.kdeplot(
            data=synthetic_acc,
            label='Synthetic',
            color='red',
            alpha=0.5
        )
        plt.title('加速度分布对比')
        plt.xlabel('加速度 (m/s²)')
        plt.ylabel('密度')
        plt.legend()
        plt.savefig(self.output_dir / 'acceleration_distribution.png')
        plt.close()
        
        return metrics
        
    def _compare_turn_rate_distributions(self) -> Dict[str, float]:
        """
        比较转向率分布
        
        Returns:
            Dict[str, float]: 转向率相关指标
        """
        # 计算转向率
        oord_turn = np.diff(self.oord_data['orientation']) / \
                    np.diff(self.oord_data['timestamp'])
        synthetic_turn = np.diff(self.synthetic_data['orientation']) / \
                        np.diff(self.synthetic_data['timestamp'])
        
        # 处理角度环绕
        oord_turn = np.where(
            oord_turn > 180,
            oord_turn - 360,
            np.where(
                oord_turn < -180,
                oord_turn + 360,
                oord_turn
            )
        )
        synthetic_turn = np.where(
            synthetic_turn > 180,
            synthetic_turn - 360,
            np.where(
                synthetic_turn < -180,
                synthetic_turn + 360,
                synthetic_turn
            )
        )
        
        # 计算KS检验
        ks_stat, p_value = stats.ks_2samp(
            oord_turn,
            synthetic_turn
        )
        
        # 计算基本统计量
        metrics = {
            'turn_rate_ks_stat': ks_stat,
            'turn_rate_ks_p_value': p_value,
            'turn_rate_mean_diff': abs(
                np.mean(oord_turn) - np.mean(synthetic_turn)
            ),
            'turn_rate_std_diff': abs(
                np.std(oord_turn) - np.std(synthetic_turn)
            )
        }
        
        # 绘制分布对比图
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=oord_turn,
            label='OORD',
            color='blue',
            alpha=0.5
        )
        sns.kdeplot(
            data=synthetic_turn,
            label='Synthetic',
            color='red',
            alpha=0.5
        )
        plt.title('转向率分布对比')
        plt.xlabel('转向率 (度/秒)')
        plt.ylabel('密度')
        plt.legend()
        plt.savefig(self.output_dir / 'turn_rate_distribution.png')
        plt.close()
        
        return metrics
        
    def _compare_environment_interaction(self) -> Dict[str, float]:
        """
        比较环境交互特性
        
        Returns:
            Dict[str, float]: 环境交互相关指标
        """
        metrics = {}
        
        # 按环境组计算平均速度
        oord_group_speeds = self.oord_data.groupby('group_label')['speed'].mean()
        synthetic_group_speeds = self.synthetic_data.groupby('group_label')['speed'].mean()
        
        # 计算组间速度差异
        common_groups = set(oord_group_speeds.index) & \
                       set(synthetic_group_speeds.index)
                       
        if not common_groups:
            logger.warning("未找到共同的环境组")
            return metrics
            
        speed_diffs = []
        for group in common_groups:
            diff = abs(
                oord_group_speeds[group] - synthetic_group_speeds[group]
            )
            speed_diffs.append(diff)
            metrics[f'speed_diff_group_{group}'] = diff
            
        metrics['mean_group_speed_diff'] = np.mean(speed_diffs)
        
        # 绘制环境组速度对比图
        plt.figure(figsize=(12, 6))
        x = np.arange(len(common_groups))
        width = 0.35
        
        plt.bar(
            x - width/2,
            [oord_group_speeds[g] for g in common_groups],
            width,
            label='OORD',
            color='blue',
            alpha=0.5
        )
        plt.bar(
            x + width/2,
            [synthetic_group_speeds[g] for g in common_groups],
            width,
            label='Synthetic',
            color='red',
            alpha=0.5
        )
        
        plt.title('环境组平均速度对比')
        plt.xlabel('环境组')
        plt.ylabel('平均速度 (m/s)')
        plt.xticks(x, common_groups, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'environment_interaction.png')
        plt.close()
        
        return metrics
        
    def _generate_report(self, metrics: Dict[str, float]) -> None:
        """
        生成评估报告
        
        Args:
            metrics: 评估指标
        """
        report_file = self.output_dir / 'evaluation_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("轨迹评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 数据概况\n")
            f.write("-" * 30 + "\n")
            f.write(f"OORD轨迹数量: {len(self.oord_data['trajectory_id'].unique())}\n")
            f.write(f"合成轨迹数量: {len(self.synthetic_data['trajectory_id'].unique())}\n")
            f.write(f"OORD数据点数: {len(self.oord_data)}\n")
            f.write(f"合成数据点数: {len(self.synthetic_data)}\n\n")
            
            f.write("2. 速度分布\n")
            f.write("-" * 30 + "\n")
            f.write(f"KS统计量: {metrics['speed_ks_stat']:.4f}\n")
            f.write(f"P值: {metrics['speed_ks_p_value']:.4f}\n")
            f.write(f"平均值差异: {metrics['speed_mean_diff']:.4f} m/s\n")
            f.write(f"标准差差异: {metrics['speed_std_diff']:.4f} m/s\n\n")
            
            f.write("3. 加速度分布\n")
            f.write("-" * 30 + "\n")
            f.write(f"KS统计量: {metrics['acceleration_ks_stat']:.4f}\n")
            f.write(f"P值: {metrics['acceleration_ks_p_value']:.4f}\n")
            f.write(f"平均值差异: {metrics['acceleration_mean_diff']:.4f} m/s²\n")
            f.write(f"标准差差异: {metrics['acceleration_std_diff']:.4f} m/s²\n\n")
            
            f.write("4. 转向率分布\n")
            f.write("-" * 30 + "\n")
            f.write(f"KS统计量: {metrics['turn_rate_ks_stat']:.4f}\n")
            f.write(f"P值: {metrics['turn_rate_ks_p_value']:.4f}\n")
            f.write(f"平均值差异: {metrics['turn_rate_mean_diff']:.4f} 度/秒\n")
            f.write(f"标准差差异: {metrics['turn_rate_std_diff']:.4f} 度/秒\n\n")
            
            if 'mean_group_speed_diff' in metrics:
                f.write("5. 环境交互\n")
                f.write("-" * 30 + "\n")
                f.write(f"环境组平均速度差异: {metrics['mean_group_speed_diff']:.4f} m/s\n")
                for key, value in metrics.items():
                    if key.startswith('speed_diff_group_'):
                        group = key.replace('speed_diff_group_', '')
                        f.write(f"组{group}速度差异: {value:.4f} m/s\n")
                        
            f.write("\n6. 结论\n")
            f.write("-" * 30 + "\n")
            
            # 添加结论
            conclusions = []
            
            # 速度分布结论
            if metrics['speed_ks_p_value'] > 0.05:
                conclusions.append("速度分布相似性良好")
            else:
                conclusions.append("速度分布存在显著差异")
                
            # 加速度分布结论
            if metrics['acceleration_ks_p_value'] > 0.05:
                conclusions.append("加速度分布相似性良好")
            else:
                conclusions.append("加速度分布存在显著差异")
                
            # 转向率分布结论
            if metrics['turn_rate_ks_p_value'] > 0.05:
                conclusions.append("转向率分布相似性良好")
            else:
                conclusions.append("转向率分布存在显著差异")
                
            # 环境交互结论
            if 'mean_group_speed_diff' in metrics:
                if metrics['mean_group_speed_diff'] < 2.0:  # 阈值可调
                    conclusions.append("环境交互特性表现良好")
                else:
                    conclusions.append("环境交互特性需要改进")
                    
            for conclusion in conclusions:
                f.write(f"- {conclusion}\n")
                
        logger.info(f"评估报告已保存至: {report_file}") 