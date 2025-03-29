"""
轨迹评估示例脚本
演示如何使用评估器评估生成轨迹的质量
"""

import logging
from pathlib import Path

from src.utils.config import config
from src.core.evaluator import Evaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    logger.info("开始轨迹评估示例")
    
    # 创建评估器
    evaluator = Evaluator()
    
    # 加载数据
    logger.info("加载数据...")
    oord_file = config.paths.PROCESSED_OORD_FILE
    synthetic_dir = config.paths.TRAJECTORY_DIR
    
    try:
        evaluator.load_data(oord_file, synthetic_dir)
    except FileNotFoundError as e:
        logger.error(f"找不到数据文件: {e}")
        return
    except ValueError as e:
        logger.error(f"数据加载错误: {e}")
        return
        
    # 执行评估
    logger.info("执行评估...")
    try:
        metrics = evaluator.evaluate()
    except Exception as e:
        logger.error(f"评估过程出错: {e}")
        return
        
    # 输出主要指标
    logger.info("\n主要评估指标:")
    logger.info("-" * 30)
    
    # 速度分布
    logger.info("速度分布:")
    logger.info(f"- KS检验P值: {metrics['speed_ks_p_value']:.4f}")
    logger.info(f"- 平均值差异: {metrics['speed_mean_diff']:.4f} m/s")
    
    # 加速度分布
    logger.info("\n加速度分布:")
    logger.info(f"- KS检验P值: {metrics['acceleration_ks_p_value']:.4f}")
    logger.info(f"- 平均值差异: {metrics['acceleration_mean_diff']:.4f} m/s²")
    
    # 转向率分布
    logger.info("\n转向率分布:")
    logger.info(f"- KS检验P值: {metrics['turn_rate_ks_p_value']:.4f}")
    logger.info(f"- 平均值差异: {metrics['turn_rate_mean_diff']:.4f} 度/秒")
    
    # 环境交互
    if 'mean_group_speed_diff' in metrics:
        logger.info("\n环境交互:")
        logger.info(
            f"- 环境组平均速度差异: {metrics['mean_group_speed_diff']:.4f} m/s"
        )
        
    # 输出文件位置
    logger.info("\n输出文件:")
    logger.info(f"- 评估报告: {evaluator.output_dir}/evaluation_report.txt")
    logger.info(f"- 分布图表: {evaluator.output_dir}/*.png")
    
    logger.info("\n示例完成")

if __name__ == '__main__':
    main() 