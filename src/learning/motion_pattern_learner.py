"""
基于窗口化轨迹与有效坡度的运动模式学习模块
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging
from src.utils.config import (
    LEARNING_WINDOW_DURATION_SECONDS,
    LEARNING_WINDOW_SLIDE_SECONDS,
    EFFECTIVE_SLOPE_ALONG_BINS,
    EFFECTIVE_SLOPE_ALONG_LABELS,
    EFFECTIVE_CROSS_SLOPE_BINS,
    EFFECTIVE_CROSS_SLOPE_LABELS,
    MIN_SAMPLES_PER_GROUP,
    LANDCOVER_MAPPING,
    DEFAULT_SPEED_MPS,
    MAX_SPEED_MPS,
    MIN_SPEED_MPS,
    LEARNED_PATTERNS_PATH,
    WINDOWED_FEATURES_PATH,
    ANALYSIS_REPORT_PATH
)
import scipy.stats
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

class MotionPatternLearner:
    """基于窗口化轨迹的运动模式学习器"""
    
    def __init__(self):
        """初始化学习器"""
        self.windowed_features_df = None
        self.learned_patterns = {}
        self.analysis_report = {
            'r2_scores': {},
            'sample_counts': {},
            'warnings': [],
            'residual_analysis': {}
        }
    
    def calculate_effective_slopes(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算有效坡度（沿路径坡度和横坡坡度）
        
        Args:
            df: 包含slope_magnitude、slope_aspect和heading_degrees的DataFrame
            
        Returns:
            添加了slope_along_path和cross_slope列的DataFrame
        """
        # 将角度转换为弧度
        heading_rad = np.deg2rad(df['heading_degrees'])
        aspect_rad = np.deg2rad(df['slope_aspect'])
        
        # 处理平地情况（slope_aspect = -1）
        flat_mask = df['slope_aspect'] == -1
        df['slope_along_path'] = np.where(
            flat_mask,
            0.0,
            df['slope_magnitude'] * np.cos(heading_rad - aspect_rad)
        )
        
        df['cross_slope'] = np.where(
            flat_mask,
            0.0,
            df['slope_magnitude'] * np.abs(np.sin(heading_rad - aspect_rad))
        )
        
        return df
    
    def process_window(self, window_df: pd.DataFrame) -> Optional[Dict]:
        """处理单个时间窗口的数据
        
        Args:
            window_df: 窗口内的数据点DataFrame
            
        Returns:
            窗口特征字典，如果窗口无效则返回None
        """
        if len(window_df) < 2:  # 确保窗口内至少有2个点
            logger.warning(f"窗口内点数不足: {len(window_df)} < 2")
            return None
            
        # 计算聚合运动特征
        avg_speed = window_df['speed_mps'].mean()
        speed_std = window_df['speed_mps'].std()
        max_speed = window_df['speed_mps'].max()
        
        # 计算聚合环境特征
        dominant_landcover = int(window_df['landcover'].mode()[0])
        avg_slope_magnitude = window_df['slope_magnitude'].mean()
        avg_slope_along_path = window_df['slope_along_path'].mean()
        avg_cross_slope = window_df['cross_slope'].mean()
        
        # 获取窗口的时空信息
        start_time = window_df.index[0]
        end_time = window_df.index[-1]
        center_pos = window_df[['longitude', 'latitude']].mean()
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'longitude': center_pos['longitude'],
            'latitude': center_pos['latitude'],
            'avg_speed': avg_speed,
            'speed_std': speed_std,
            'max_speed': max_speed,
            'dominant_landcover': dominant_landcover,
            'avg_slope_magnitude': avg_slope_magnitude,
            'avg_slope_along_path': avg_slope_along_path,
            'avg_cross_slope': avg_cross_slope
        }
    
    def apply_windowing(self, df: pd.DataFrame) -> List[Dict]:
        """对轨迹数据应用滑动窗口
        
        Args:
            df: 轨迹DataFrame，必须按时间戳排序
            
        Returns:
            窗口特征列表
        """
        window_features = []
        
        # 确保数据有时间索引
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame没有时间索引，尝试使用时间戳列创建")
            # 将时间戳转换为秒级时间戳
            df.index = pd.to_datetime(df['timestamp'].astype(float), unit='s')
        
        # 计算窗口大小和滑动步长（秒）
        window_size = pd.Timedelta(seconds=LEARNING_WINDOW_DURATION_SECONDS)
        slide_size = pd.Timedelta(seconds=LEARNING_WINDOW_SLIDE_SECONDS)
        
        # 获取时间范围
        start_time = df.index[0]
        end_time = df.index[-1]
        current_time = start_time
        
        while current_time + window_size <= end_time:
            # 获取窗口内的数据
            window_mask = (df.index >= current_time) & (df.index < current_time + window_size)
            window_df = df[window_mask]
            
            # 处理窗口数据
            window_features_dict = self.process_window(window_df)
            if window_features_dict:
                window_features.append(window_features_dict)
                
            # 滑动窗口
            current_time += slide_size
            
        if not window_features:
            logger.warning("没有生成任何有效的窗口特征，尝试使用整个轨迹作为一个窗口")
            window_features_dict = self.process_window(df)
            if window_features_dict:
                window_features.append(window_features_dict)
        
        logger.info(f"生成了 {len(window_features)} 个窗口特征")
        return window_features
    
    def create_group_label(self, row: pd.Series) -> str:
        """创建环境组标签
        
        Args:
            row: 包含环境特征的Series
            
        Returns:
            环境组标签字符串
        """
        # 离散化坡度特征
        slope_along_bin = pd.cut(
            [row['avg_slope_along_path']], 
            bins=EFFECTIVE_SLOPE_ALONG_BINS,
            labels=EFFECTIVE_SLOPE_ALONG_LABELS
        )[0]
        
        cross_slope_bin = pd.cut(
            [row['avg_cross_slope']], 
            bins=EFFECTIVE_CROSS_SLOPE_BINS,
            labels=EFFECTIVE_CROSS_SLOPE_LABELS
        )[0]
        
        # 获取地物类型标签
        landcover_label = LANDCOVER_MAPPING.get(int(row['dominant_landcover']), 'Unknown')
        
        return f"LC{landcover_label}_SA{slope_along_bin}_CS{cross_slope_bin}"
    
    def learn_from_trajectory(self, trajectory_df: pd.DataFrame) -> None:
        """从单条轨迹学习运动模式
        
        Args:
            trajectory_df: 轨迹DataFrame，包含所有必需列
        """
        logger.info("开始处理轨迹数据...")
        
        # 1. 计算有效坡度
        trajectory_df = self.calculate_effective_slopes(trajectory_df)
        logger.info("完成有效坡度计算")
        
        # 2. 应用窗口化处理
        window_features = self.apply_windowing(trajectory_df)
        logger.info(f"生成了 {len(window_features)} 个窗口特征")
        
        if not window_features:
            raise ValueError("没有生成任何窗口特征，请检查数据")
        
        # 3. 创建窗口化特征DataFrame
        self.windowed_features_df = pd.DataFrame(window_features)
        
        # 4. 添加环境组标签
        self.windowed_features_df['group_label'] = self.windowed_features_df.apply(
            self.create_group_label, axis=1
        )
        
        # 5. 进行统计分析
        self._perform_statistical_analysis()
        
        # 6. 保存中间结果
        if not os.path.exists(os.path.dirname(WINDOWED_FEATURES_PATH)):
            os.makedirs(os.path.dirname(WINDOWED_FEATURES_PATH))
        self.windowed_features_df.to_csv(WINDOWED_FEATURES_PATH)
        logger.info("完成轨迹学习")
    
    def analyze_residuals(self) -> Dict:
        """分析回归模型的残差
        
        Returns:
            包含残差分析结果的字典
        """
        # 准备特征和目标变量
        X = self.windowed_features_df[['avg_slope_magnitude', 'avg_slope_along_path', 'avg_cross_slope']]
        y = self.windowed_features_df['avg_speed']
        
        # 拟合线性回归模型
        reg = LinearRegression()
        reg.fit(X, y)
        
        # 计算预测值和残差
        y_pred = reg.predict(X)
        residuals = y - y_pred
        
        # 计算残差统计量
        residual_stats = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skewness': float(pd.Series(residuals).skew()),
            'kurtosis': float(pd.Series(residuals).kurtosis()),
            'shapiro_test': scipy.stats.shapiro(residuals),
            'residuals': residuals.tolist(),
            'predicted': y_pred.tolist(),
            'actual': y.tolist()
        }
        
        # 保存残差分析结果
        self.analysis_report['residual_analysis'] = residual_stats
        
        return residual_stats
        
    def fit_gmm_to_residuals(self, n_components=2) -> Dict:
        """使用混合高斯模型拟合残差分布
        
        Args:
            n_components: 高斯分量的数量
            
        Returns:
            包含拟合结果的字典
        """
        # 获取残差数据
        residual_analysis = self.analysis_report.get('residual_analysis', self.analyze_residuals())
        residuals = np.array(residual_analysis['residuals']).reshape(-1, 1)
        
        # 拟合GMM
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(residuals)
        
        # 计算BIC和AIC
        bic = gmm.bic(residuals)
        aic = gmm.aic(residuals)
        
        # 保存GMM结果
        gmm_results = {
            'weights': gmm.weights_.tolist(),
            'means': gmm.means_.reshape(-1).tolist(),
            'stds': np.sqrt(gmm.covariances_).reshape(-1).tolist(),
            'bic': float(bic),
            'aic': float(aic)
        }
        
        self.analysis_report['gmm_analysis'] = gmm_results
        return gmm_results
        
    def plot_residual_analysis(self, output_dir: str = 'data/learned') -> None:
        """生成残差分析图
        
        Args:
            output_dir: 输出目录路径
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        from src.utils import plot_style
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取残差数据
        residual_analysis = self.analysis_report.get('residual_analysis', self.analyze_residuals())
        residuals = np.array(residual_analysis['residuals'])
        predicted = np.array(residual_analysis['predicted'])
        actual = np.array(residual_analysis['actual'])
        
        # 拟合GMM
        gmm_results = self.fit_gmm_to_residuals()
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 残差分布和GMM拟合结果
        ax1 = plt.subplot(221)
        sns.histplot(residuals, stat='density', bins=30, alpha=0.6, ax=ax1)
        
        # 绘制GMM各分量和总体分布
        x = np.linspace(min(residuals), max(residuals), 200)
        total_density = np.zeros_like(x)
        
        for i, (weight, mean, std) in enumerate(zip(
            gmm_results['weights'], 
            gmm_results['means'], 
            gmm_results['stds']
        )):
            density = weight * stats.norm.pdf(x, mean, std)
            total_density += density
            ax1.plot(x, density, '--', label=f'高斯分量 {i+1}')
            
        ax1.plot(x, total_density, 'r-', label='混合分布')
        ax1.set_title('残差分布与混合高斯拟合')
        ax1.set_xlabel('残差值 (m/s)')
        ax1.set_ylabel('密度')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Q-Q图
        ax2 = plt.subplot(222)
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('残差Q-Q图')
        ax2.grid(True)
        
        # 3. 残差vs预测值散点图
        ax3 = plt.subplot(223)
        ax3.scatter(predicted, residuals, alpha=0.5)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_title('残差 vs 预测值')
        ax3.set_xlabel('预测速度 (m/s)')
        ax3.set_ylabel('残差 (m/s)')
        ax3.grid(True)
        
        # 4. 实际值vs预测值散点图
        ax4 = plt.subplot(224)
        ax4.scatter(predicted, actual, alpha=0.5)
        ax4.plot([min(predicted), max(predicted)], 
                [min(predicted), max(predicted)], 'r--')
        ax4.set_title('实际值 vs 预测值')
        ax4.set_xlabel('预测速度 (m/s)')
        ax4.set_ylabel('实际速度 (m/s)')
        ax4.grid(True)
        
        # 添加总标题
        plt.suptitle('运动模式回归分析诊断图', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # 保存图形
        plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"残差分析图已保存到 {output_dir}/residual_analysis.png")
        
        # 输出统计信息
        logger.info("\n残差统计分析:")
        logger.info(f"均值: {residual_analysis['mean']:.4f}")
        logger.info(f"标准差: {residual_analysis['std']:.4f}")
        logger.info(f"偏度: {residual_analysis['skewness']:.4f}")
        logger.info(f"峰度: {residual_analysis['kurtosis']:.4f}")
        shapiro_stat, shapiro_p = residual_analysis['shapiro_test']
        logger.info(f"Shapiro-Wilk正态性检验: 统计量={shapiro_stat:.4f}, p值={shapiro_p:.4f}")
        
        # 输出GMM分析结果
        logger.info("\nGMM分析结果:")
        for i, (weight, mean, std) in enumerate(zip(
            gmm_results['weights'], 
            gmm_results['means'], 
            gmm_results['stds']
        )):
            logger.info(f"分量 {i+1}:")
            logger.info(f"  权重: {weight:.4f}")
            logger.info(f"  均值: {mean:.4f}")
            logger.info(f"  标准差: {std:.4f}")
        logger.info(f"BIC: {gmm_results['bic']:.4f}")
        logger.info(f"AIC: {gmm_results['aic']:.4f}")

    def _perform_statistical_analysis(self) -> None:
        """执行统计分析，计算每个环境组的运动特征"""
        # 1. 计算基本的R²值和残差分析
        X = self.windowed_features_df[['avg_slope_magnitude', 'avg_slope_along_path', 'avg_cross_slope']]
        y = self.windowed_features_df['avg_speed']
        
        reg = LinearRegression()
        reg.fit(X, y)
        r2 = r2_score(y, reg.predict(X))
        self.analysis_report['r2_scores']['basic'] = r2
        logger.info(f"基本R²值: {r2:.4f}")
        
        # 执行残差分析
        self.analyze_residuals()
        
        # 2. 按环境组进行分组统计
        grouped_stats = self.windowed_features_df.groupby('group_label').agg({
            'avg_speed': ['count', 'mean', 'std', 'min', 'max'],
            'speed_std': 'mean',
            'avg_slope_along_path': 'mean',
            'avg_cross_slope': 'mean'
        })
        
        # 3. 处理每个环境组
        for group_label, stats in grouped_stats.iterrows():
            sample_count = stats[('avg_speed', 'count')]
            
            if sample_count < MIN_SAMPLES_PER_GROUP:
                self.analysis_report['warnings'].append(
                    f"Group {group_label} has insufficient samples: {sample_count}"
                )
                logger.warning(f"环境组 {group_label} 样本不足: {sample_count}")
                continue
                
            self.learned_patterns[group_label] = {
                'typical_speed': float(stats[('avg_speed', 'mean')]),
                'speed_std': float(stats[('speed_std', 'mean')]),
                'min_speed': max(float(stats[('avg_speed', 'min')]), MIN_SPEED_MPS),
                'max_speed': min(float(stats[('avg_speed', 'max')]), MAX_SPEED_MPS),
                'avg_slope_along': float(stats[('avg_slope_along_path', 'mean')]),
                'avg_cross_slope': float(stats[('avg_cross_slope', 'mean')]),
                'sample_count': int(sample_count)
            }
            
        self.analysis_report['sample_counts'] = grouped_stats['avg_speed']['count'].to_dict()
        logger.info(f"完成 {len(self.learned_patterns)} 个环境组的统计分析")
    
    def save_results(self) -> None:
        """保存学习结果和分析报告"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(LEARNED_PATTERNS_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(ANALYSIS_REPORT_PATH), exist_ok=True)
        
        # 保存学习到的模式
        with open(LEARNED_PATTERNS_PATH, 'w') as f:
            json.dump(self.learned_patterns, f, indent=2)
            
        # 保存分析报告
        with open(ANALYSIS_REPORT_PATH, 'w') as f:
            json.dump(self.analysis_report, f, indent=2)
            
        logger.info(f"结果已保存到 {LEARNED_PATTERNS_PATH}")
        logger.info(f"分析报告已保存到 {ANALYSIS_REPORT_PATH}")
    
    def get_speed_for_conditions(self, landcover: int, slope_along: float, 
                               cross_slope: float) -> Dict[str, float]:
        """根据给定的环境条件获取预测的速度特征
        
        Args:
            landcover: 地物类型编码
            slope_along: 沿路径坡度
            cross_slope: 横坡坡度
            
        Returns:
            包含速度特征的字典
        """
        # 创建临时Series用于生成组标签
        temp_row = pd.Series({
            'dominant_landcover': landcover,
            'avg_slope_along_path': slope_along,
            'avg_cross_slope': cross_slope
        })
        
        group_label = self.create_group_label(temp_row)
        
        # 如果找到匹配的组，返回其速度特征
        if group_label in self.learned_patterns:
            return self.learned_patterns[group_label]
            
        # 否则返回默认值
        return {
            'typical_speed': DEFAULT_SPEED_MPS,
            'speed_std': 2.0,
            'min_speed': MIN_SPEED_MPS,
            'max_speed': MAX_SPEED_MPS,
            'is_default': True
        }

    def get_environment_group(self, landcover: int, slope_along: float, cross_slope: float) -> str:
        """根据地形特征对环境进行分组"""
        # 地形分类
        if landcover == 0:
            lc_str = "LCUnknown"
        elif landcover == 1:
            lc_str = "LCUrban"
        elif landcover == 2:
            lc_str = "LCRural"
        else:
            lc_str = "LCOther"
            
        # 沿路坡度分类
        if abs(slope_along) < 5:
            sa_str = "SAFlat"
        elif slope_along >= 5 and slope_along < 15:
            sa_str = "SAModUp" if slope_along > 0 else "SAModDown"
        else:
            sa_str = "SASteepUp" if slope_along > 0 else "SASteepDown"
            
        # 横坡分类
        if abs(cross_slope) < 10:
            cs_str = "CSLow"
        elif abs(cross_slope) < 20:
            cs_str = "CSMedium"
        else:
            cs_str = "CSHigh"
            
        return f"{lc_str}_{sa_str}_{cs_str}"

    def load_results(self) -> None:
        """从文件加载学习结果"""
        if os.path.exists(LEARNED_PATTERNS_PATH):
            with open(LEARNED_PATTERNS_PATH, 'r') as f:
                self.learned_patterns = json.load(f)
            logger.info(f"从 {LEARNED_PATTERNS_PATH} 加载模型")
        else:
            raise FileNotFoundError(f"找不到学习结果文件: {LEARNED_PATTERNS_PATH}")
            
        if os.path.exists(ANALYSIS_REPORT_PATH):
            with open(ANALYSIS_REPORT_PATH, 'r') as f:
                self.analysis_report = json.load(f)
            logger.info(f"从 {ANALYSIS_REPORT_PATH} 加载分析报告")
        else:
            raise FileNotFoundError(f"找不到分析报告文件: {ANALYSIS_REPORT_PATH}")
            
        if os.path.exists(WINDOWED_FEATURES_PATH):
            self.windowed_features_df = pd.read_csv(WINDOWED_FEATURES_PATH)
            logger.info(f"从 {WINDOWED_FEATURES_PATH} 加载窗口特征") 