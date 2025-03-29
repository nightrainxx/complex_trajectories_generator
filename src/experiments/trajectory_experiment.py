"""
轨迹实验模块
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from .window_feature_extractor import WindowFeatureExtractor
from .environment_pattern_analyzer import EnvironmentPatternAnalyzer
from src.data_processing.trajectory_processor import TrajectoryProcessor

logger = logging.getLogger(__name__)

class TrajectoryExperiment:
    """单条轨迹实验"""
    
    def __init__(self, sequence_id: int, exp_dir: str,
                 dem_file: str, landcover_file: str):
        """初始化实验
        
        Args:
            sequence_id: 轨迹序号
            exp_dir: 实验目录
            dem_file: DEM文件路径
            landcover_file: 地表覆盖文件路径
        """
        self.sequence_id = sequence_id
        self.exp_dir = Path(exp_dir)
        
        # 创建特征提取器和分析器
        self.feature_extractor = WindowFeatureExtractor(window_size_seconds=30)
        self.env_analyzer = EnvironmentPatternAnalyzer()
        
        # 创建轨迹处理器
        self.trajectory_processor = TrajectoryProcessor(
            dem_file=dem_file,
            landcover_file=landcover_file
        )
        
        logger.info(f"初始化轨迹 {sequence_id} 的实验，输出目录: {exp_dir}")
        
    def _create_directories(self) -> None:
        """创建实验目录结构"""
        dirs = [
            'training/features',
            'training/env_patterns',
            'training/motion_patterns',
            'training/analysis',
            'validation/plots',
            'validation/stats',
            'models'
        ]
        
        for dir_path in dirs:
            os.makedirs(self.exp_dir / dir_path, exist_ok=True)
            
        logger.info("创建实验目录结构完成")
        
    def _load_trajectory(self, trajectory_file: str) -> pd.DataFrame:
        """加载轨迹数据
        
        Args:
            trajectory_file: 轨迹文件路径
            
        Returns:
            轨迹DataFrame
        """
        df = pd.read_csv(trajectory_file)
        logger.info(f"加载轨迹数据，共 {len(df)} 个点")
        return df
        
    def _process_environment(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """处理环境特征
        
        Args:
            trajectory_df: 轨迹DataFrame
            
        Returns:
            环境特征DataFrame
        """
        env_features = self.trajectory_processor.process_trajectory(trajectory_df)
        logger.info("处理环境特征完成")
        return env_features
        
    def _analyze_patterns(self, window_features: pd.DataFrame) -> Dict:
        """分析环境-速度关系模式
        
        Args:
            window_features: 窗口特征DataFrame
            
        Returns:
            环境模式字典
        """
        patterns = self.env_analyzer.analyze_patterns(window_features)
        
        # 保存环境模式
        output_file = self.exp_dir / 'training/env_patterns/patterns.json'
        self.env_analyzer.save_patterns(patterns, str(output_file))
        
        return patterns
        
    def _plot_training_analysis(self, window_features: pd.DataFrame,
                              patterns: Dict) -> None:
        """绘制训练分析图
        
        Args:
            window_features: 窗口特征DataFrame
            patterns: 环境模式字典
        """
        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 速度-坡度散点图
        ax1 = plt.subplot(221)
        ax1.scatter(window_features['avg_slope'], 
                   window_features['avg_speed'],
                   alpha=0.5)
        ax1.set_xlabel('平均坡度 (度)')
        ax1.set_ylabel('平均速度 (m/s)')
        ax1.set_title('速度-坡度关系')
        ax1.grid(True)
        
        # 2. 速度-横坡散点图
        ax2 = plt.subplot(222)
        ax2.scatter(window_features['avg_cross_slope'],
                   window_features['avg_speed'],
                   alpha=0.5)
        ax2.set_xlabel('平均横坡 (度)')
        ax2.set_ylabel('平均速度 (m/s)')
        ax2.set_title('速度-横坡关系')
        ax2.grid(True)
        
        # 3. 不同地表类型的速度箱线图
        ax3 = plt.subplot(223)
        window_features.boxplot(column='avg_speed',
                              by='dominant_landcover',
                              ax=ax3)
        ax3.set_xlabel('地表类型')
        ax3.set_ylabel('平均速度 (m/s)')
        ax3.set_title('不同地表类型的速度分布')
        
        # 4. 环境组的样本数量
        ax4 = plt.subplot(224)
        sample_counts = pd.Series({k: v['sample_count'] for k, v in patterns.items()})
        sample_counts.plot(kind='bar', ax=ax4)
        ax4.set_xlabel('环境组')
        ax4.set_ylabel('样本数量')
        ax4.set_title('环境组样本分布')
        plt.xticks(rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        output_file = self.exp_dir / 'training/analysis/patterns_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练分析图已保存到: {output_file}")
        
    def _validate_patterns(self, window_features: pd.DataFrame,
                         patterns: Dict) -> Dict:
        """验证环境模式
        
        Args:
            window_features: 窗口特征DataFrame
            patterns: 环境模式字典
            
        Returns:
            验证结果字典
        """
        # 预测每个窗口的速度
        predicted_speeds = []
        for _, row in window_features.iterrows():
            group = self.env_analyzer.create_environment_group(
                slope=row['avg_slope'],
                cross_slope=row['avg_cross_slope'],
                landcover=row['dominant_landcover']
            )
            
            if group in patterns:
                predicted_speeds.append(patterns[group]['mean_speed'])
            else:
                # 如果找不到匹配的环境组，使用相似环境组的平均速度
                similar_groups = [
                    p for g, p in patterns.items()
                    if g.startswith(f"LC{row['dominant_landcover']}")
                ]
                if similar_groups:
                    predicted_speeds.append(
                        np.mean([p['mean_speed'] for p in similar_groups])
                    )
                else:
                    # 如果没有相似的环境组，使用全局平均速度
                    predicted_speeds.append(window_features['avg_speed'].mean())
                    
        # 计算验证指标
        mse = mean_squared_error(
            window_features['avg_speed'],
            predicted_speeds
        )
        rmse = np.sqrt(mse)
        r2 = r2_score(
            window_features['avg_speed'],
            predicted_speeds
        )
        
        # 计算平均误差
        errors = np.array(predicted_speeds) - window_features['avg_speed']
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        results = {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'predicted_speeds': predicted_speeds
        }
        
        # 保存验证结果
        output_file = self.exp_dir / 'validation/stats/validation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"验证结果已保存到: {output_file}")
        return results
        
    def _plot_validation_results(self, window_features: pd.DataFrame,
                               validation_results: Dict) -> None:
        """绘制验证结果图
        
        Args:
            window_features: 窗口特征DataFrame
            validation_results: 验证结果字典
        """
        predicted_speeds = validation_results['predicted_speeds']
        
        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 实际速度vs预测速度散点图
        ax1 = plt.subplot(221)
        ax1.scatter(window_features['avg_speed'],
                   predicted_speeds,
                   alpha=0.5)
        ax1.plot([0, 10], [0, 10], 'r--')  # 对角线
        ax1.set_xlabel('实际速度 (m/s)')
        ax1.set_ylabel('预测速度 (m/s)')
        ax1.set_title('实际vs预测速度')
        ax1.grid(True)
        
        # 2. 速度误差直方图
        ax2 = plt.subplot(222)
        errors = predicted_speeds - window_features['avg_speed']
        ax2.hist(errors, bins=30, alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('速度误差 (m/s)')
        ax2.set_ylabel('频数')
        ax2.set_title('速度误差分布')
        ax2.grid(True)
        
        # 3. 速度时间序列
        ax3 = plt.subplot(212)
        ax3.plot(window_features.index,
                window_features['avg_speed'],
                label='实际速度')
        ax3.plot(window_features.index,
                predicted_speeds,
                label='预测速度')
        ax3.set_xlabel('窗口序号')
        ax3.set_ylabel('速度 (m/s)')
        ax3.set_title('速度时间序列')
        ax3.legend()
        ax3.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        output_file = self.exp_dir / 'validation/plots/validation_results.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"验证结果图已保存到: {output_file}")
        
    def run(self, trajectory_file: str) -> None:
        """运行实验
        
        Args:
            trajectory_file: 轨迹文件路径
        """
        # 1. 创建目录
        self._create_directories()
        
        # 2. 加载数据
        trajectory_df = self._load_trajectory(trajectory_file)
        env_features = self._process_environment(trajectory_df)
        
        # 3. 提取窗口特征
        window_features = self.feature_extractor.extract_window_features(
            trajectory_df, env_features
        )
        
        # 保存窗口特征
        window_features.to_csv(
            self.exp_dir / 'training/features/window_features.csv',
            index=False
        )
        
        # 4. 分析环境模式
        patterns = self._analyze_patterns(window_features)
        
        # 5. 绘制训练分析图
        self._plot_training_analysis(window_features, patterns)
        
        # 6. 验证
        validation_results = self._validate_patterns(window_features, patterns)
        
        # 7. 绘制验证结果图
        self._plot_validation_results(window_features, validation_results)
        
        logger.info(f"轨迹 {self.sequence_id} 的实验完成") 