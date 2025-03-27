"""轨迹评估器模块

负责评估生成的轨迹与OORD数据的相似度。

输入:
- 生成的轨迹数据
- OORD参考数据
- 环境数据

输出:
- 评估报告（图表和统计数据）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class Evaluator:
    """轨迹评估器类"""

    def __init__(self, config: dict, output_dir: str):
        """初始化评估器

        Args:
            config: 配置字典
            output_dir: 输出目录路径
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置图表样式
        plt.style.use('seaborn')
        sns.set_palette("husl")

    def load_synthetic_data(self, batch_dir: str) -> pd.DataFrame:
        """加载生成的轨迹数据

        Args:
            batch_dir: 批处理输出目录

        Returns:
            pd.DataFrame: 包含所有生成轨迹的DataFrame
        """
        logger.info(f"加载生成的轨迹数据: {batch_dir}")
        
        # 读取所有轨迹文件
        trajectories = []
        batch_path = Path(batch_dir)
        for traj_file in batch_path.glob("trajectory_*.csv"):
            df = pd.read_csv(traj_file)
            df['trajectory_id'] = traj_file.stem
            trajectories.append(df)

        # 合并所有轨迹
        if not trajectories:
            raise ValueError(f"在 {batch_dir} 中未找到轨迹文件")
            
        return pd.concat(trajectories, ignore_index=True)

    def load_processed_oord_data(self, oord_path: str) -> pd.DataFrame:
        """加载处理好的OORD数据

        Args:
            oord_path: OORD数据文件路径

        Returns:
            pd.DataFrame: OORD数据
        """
        logger.info(f"加载OORD参考数据: {oord_path}")
        return pd.read_csv(oord_path)

    def compare_global_distributions(self, synthetic_df: pd.DataFrame,
                                  oord_df: pd.DataFrame) -> Dict:
        """比较全局统计分布

        Args:
            synthetic_df: 生成的轨迹数据
            oord_df: OORD参考数据

        Returns:
            Dict: 比较结果
        """
        logger.info("比较全局统计分布...")
        
        results = {}
        
        # 比较速度分布
        self._compare_distribution(
            synthetic_df['speed_mps'],
            oord_df['speed_mps'],
            'speed_distribution.png',
            '速度分布对比 (m/s)',
            results
        )
        
        # 比较加速度分布
        synthetic_acc = synthetic_df.groupby('trajectory_id')['speed_mps'].diff() / \
                       synthetic_df.groupby('trajectory_id')['timestamp'].diff()
        oord_acc = oord_df.groupby('trajectory_id')['speed_mps'].diff() / \
                  oord_df.groupby('trajectory_id')['timestamp'].diff()
        
        self._compare_distribution(
            synthetic_acc,
            oord_acc,
            'acceleration_distribution.png',
            '加速度分布对比 (m/s²)',
            results
        )
        
        # 比较转向率分布
        synthetic_turn = synthetic_df.groupby('trajectory_id')['heading_degrees'].diff() / \
                        synthetic_df.groupby('trajectory_id')['timestamp'].diff()
        oord_turn = oord_df.groupby('trajectory_id')['heading_degrees'].diff() / \
                   oord_df.groupby('trajectory_id')['timestamp'].diff()
        
        self._compare_distribution(
            synthetic_turn,
            oord_turn,
            'turn_rate_distribution.png',
            '转向率分布对比 (度/s)',
            results
        )
        
        return results

    def compare_environment_interaction(self, synthetic_df: pd.DataFrame,
                                     oord_df: pd.DataFrame) -> Dict:
        """比较与环境的交互特征

        Args:
            synthetic_df: 生成的轨迹数据
            oord_df: OORD参考数据

        Returns:
            Dict: 比较结果
        """
        logger.info("比较环境交互特征...")
        
        results = {}
        
        # 按坡度等级分组比较
        slope_bins = self.config['SLOPE_BINS']
        synthetic_df['slope_group'] = pd.cut(synthetic_df['slope_magnitude'],
                                           bins=slope_bins,
                                           labels=[f"{a:.1f}-{b:.1f}" for a, b in 
                                                 zip(slope_bins[:-1], slope_bins[1:])])
        oord_df['slope_group'] = pd.cut(oord_df['slope_magnitude'],
                                       bins=slope_bins,
                                       labels=[f"{a:.1f}-{b:.1f}" for a, b in 
                                             zip(slope_bins[:-1], slope_bins[1:])])
        
        # 绘制不同坡度下的速度箱线图
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        sns.boxplot(data=synthetic_df, x='slope_group', y='speed_mps')
        plt.title('生成轨迹: 坡度-速度关系')
        plt.xticks(rotation=45)
        
        plt.subplot(122)
        sns.boxplot(data=oord_df, x='slope_group', y='speed_mps')
        plt.title('OORD数据: 坡度-速度关系')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'slope_speed_relationship.png')
        plt.close()
        
        # 计算每个坡度组的统计量
        for slope_group in synthetic_df['slope_group'].unique():
            syn_speeds = synthetic_df[synthetic_df['slope_group'] == slope_group]['speed_mps']
            oord_speeds = oord_df[oord_df['slope_group'] == slope_group]['speed_mps']
            
            results[f'slope_group_{slope_group}'] = {
                'synthetic_mean': syn_speeds.mean(),
                'oord_mean': oord_speeds.mean(),
                'synthetic_std': syn_speeds.std(),
                'oord_std': oord_speeds.std(),
                'ks_statistic': stats.ks_2samp(syn_speeds, oord_speeds).statistic
            }
        
        return results

    def _compare_distribution(self, synthetic_data: pd.Series,
                            oord_data: pd.Series,
                            output_filename: str,
                            title: str,
                            results: Dict) -> None:
        """比较两个分布并生成可视化

        Args:
            synthetic_data: 生成的数据
            oord_data: OORD参考数据
            output_filename: 输出文件名
            title: 图表标题
            results: 结果字典
        """
        # 计算统计量
        syn_stats = {
            'mean': synthetic_data.mean(),
            'std': synthetic_data.std(),
            'median': synthetic_data.median(),
            'q1': synthetic_data.quantile(0.25),
            'q3': synthetic_data.quantile(0.75)
        }
        
        oord_stats = {
            'mean': oord_data.mean(),
            'std': oord_data.std(),
            'median': oord_data.median(),
            'q1': oord_data.quantile(0.25),
            'q3': oord_data.quantile(0.75)
        }
        
        # 执行K-S检验
        ks_stat, p_value = stats.ks_2samp(synthetic_data.dropna(),
                                         oord_data.dropna())
        
        # 绘制分布对比图
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=synthetic_data, label='生成轨迹')
        sns.kdeplot(data=oord_data, label='OORD数据')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / output_filename)
        plt.close()
        
        # 保存结果
        metric_name = output_filename.split('.')[0]
        results[metric_name] = {
            'synthetic_stats': syn_stats,
            'oord_stats': oord_stats,
            'ks_test': {
                'statistic': ks_stat,
                'p_value': p_value
            }
        }

    def save_report(self, results: Dict) -> None:
        """保存评估报告

        Args:
            results: 评估结果
        """
        # 保存JSON格式的详细结果
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # 生成文本报告
        report_lines = [
            "轨迹生成评估报告",
            "=" * 50,
            "",
            "1. 全局统计比较",
            "-" * 20
        ]
        
        for metric, data in results.items():
            if metric.endswith('_distribution'):
                report_lines.extend([
                    f"\n{metric.replace('_', ' ').title()}:",
                    f"生成轨迹: 均值={data['synthetic_stats']['mean']:.2f}, "
                    f"标准差={data['synthetic_stats']['std']:.2f}",
                    f"OORD数据: 均值={data['oord_stats']['mean']:.2f}, "
                    f"标准差={data['oord_stats']['std']:.2f}",
                    f"K-S检验: 统计量={data['ks_test']['statistic']:.3f}, "
                    f"p值={data['ks_test']['p_value']:.3f}"
                ])
        
        report_lines.extend([
            "",
            "2. 环境交互分析",
            "-" * 20
        ])
        
        for key, data in results.items():
            if key.startswith('slope_group'):
                report_lines.extend([
                    f"\n{key}:",
                    f"生成轨迹: 均值={data['synthetic_mean']:.2f}, "
                    f"标准差={data['synthetic_std']:.2f}",
                    f"OORD数据: 均值={data['oord_mean']:.2f}, "
                    f"标准差={data['oord_std']:.2f}",
                    f"K-S统计量: {data['ks_statistic']:.3f}"
                ])
        
        # 保存文本报告
        with open(self.output_dir / 'evaluation_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"评估报告已保存到: {self.output_dir}")

    def evaluate_batch(self, batch_dir: str, oord_path: str) -> None:
        """评估一批生成的轨迹

        Args:
            batch_dir: 批处理输出目录
            oord_path: OORD数据文件路径
        """
        # 加载数据
        synthetic_df = self.load_synthetic_data(batch_dir)
        oord_df = self.load_processed_oord_data(oord_path)
        
        # 执行评估
        results = {}
        results.update(self.compare_global_distributions(synthetic_df, oord_df))
        results.update(self.compare_environment_interaction(synthetic_df, oord_df))
        
        # 保存报告
        self.save_report(results) 