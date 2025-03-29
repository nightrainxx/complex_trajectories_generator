"""
训练运动模式的主脚本
"""
import os
import pandas as pd
import logging
from pathlib import Path
from src.learning.motion_pattern_learner import MotionPatternLearner
from src.utils.config import WINDOWED_FEATURES_PATH, LEARNED_PATTERNS_PATH

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_trajectory_data(file_path: str) -> pd.DataFrame:
    """加载轨迹数据
    
    Args:
        file_path: 轨迹CSV文件路径
        
    Returns:
        处理后的轨迹DataFrame
    """
    logger.info(f"Loading trajectory data from {file_path}")
    df = pd.read_csv(file_path)
    
    # 设置时间索引
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    
    # 确保数据按时间排序
    df.sort_index(inplace=True)
    
    # 验证必需的列
    required_columns = [
        'longitude', 'latitude', 'speed_mps', 'heading_degrees',
        'slope_magnitude', 'slope_aspect', 'landcover'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
        
    return df

def main():
    """主函数"""
    # 设置输入文件路径
    base_dir = Path("/home/yzc/data/Sucess_or_Die/complex_trajectories_generator")
    trajectory_file = base_dir / "data/processed/trajectory_1_processed.csv"
    
    try:
        # 1. 加载轨迹数据
        trajectory_df = load_trajectory_data(str(trajectory_file))
        logger.info(f"Loaded trajectory with {len(trajectory_df)} points")
        
        # 2. 创建学习器实例
        learner = MotionPatternLearner()
        
        # 3. 执行学习过程
        logger.info("Starting motion pattern learning...")
        learner.learn_from_trajectory(trajectory_df)
        
        # 4. 生成残差分析图
        logger.info("生成残差分析图...")
        learner.plot_residual_analysis()
        
        # 5. 保存结果
        learner.save_results()
        logger.info(f"Results saved to {LEARNED_PATTERNS_PATH}")
        logger.info(f"Windowed features saved to {WINDOWED_FEATURES_PATH}")
        
        # 6. 输出分析报告摘要
        logger.info("\n分析报告摘要:")
        logger.info(f"R² Score: {learner.analysis_report['r2_scores']['basic']:.4f}")
        logger.info(f"环境组总数: {len(learner.learned_patterns)}")
        logger.info(f"样本不足的组数: {len(learner.analysis_report['warnings'])}")
        
        # 7. 输出残差分析结果
        residual_analysis = learner.analysis_report['residual_analysis']
        logger.info("\n残差分布分析:")
        logger.info(f"均值: {residual_analysis['mean']:.4f}")
        logger.info(f"标准差: {residual_analysis['std']:.4f}")
        logger.info(f"偏度: {residual_analysis['skewness']:.4f}")
        logger.info(f"峰度: {residual_analysis['kurtosis']:.4f}")
        shapiro_stat, shapiro_p = residual_analysis['shapiro_test']
        logger.info(f"Shapiro-Wilk正态性检验: 统计量={shapiro_stat:.4f}, p值={shapiro_p:.4f}")
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 