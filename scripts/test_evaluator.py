"""
测试轨迹评估器功能
"""

import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np

from src.evaluation import Evaluator
from src.utils.logging_utils import setup_logging

def main():
    # 设置日志
    setup_logging(
        log_file='logs/evaluation.log',
        log_level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    output_dir = Path('data/output/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载OORD数据
    logger.info("加载OORD数据...")
    oord_data = pd.read_csv(
        'data/output/intermediate/processed_oord_data.csv'
    )
    
    # 加载生成的轨迹数据
    logger.info("加载生成的轨迹数据...")
    trajectories = []
    group_labels_list = []
    
    # 从trajectory_generation目录加载所有轨迹
    traj_dir = Path('data/output/trajectory_generation')
    for traj_file in traj_dir.glob('trajectory_*.json'):
        with open(traj_file, 'r') as f:
            traj_data = json.load(f)
            trajectories.append(traj_data)
            
            # 如果有环境组标签文件，也加载它
            group_file = traj_file.parent / f"{traj_file.stem}_groups.json"
            if group_file.exists():
                with open(group_file, 'r') as gf:
                    group_labels = json.load(gf)
                    group_labels_list.append(group_labels)
    
    if not trajectories:
        logger.error("未找到生成的轨迹数据")
        return
    
    # 创建评估器
    evaluator = Evaluator(
        oord_data=oord_data,
        output_dir=str(output_dir)
    )
    
    # 评估单条轨迹
    logger.info("评估第一条轨迹...")
    single_result = evaluator.evaluate_trajectory(
        trajectories[0],
        group_labels_list[0] if group_labels_list else None
    )
    
    # 评估所有轨迹
    logger.info("评估所有轨迹...")
    batch_results = evaluator.evaluate_batch(
        trajectories,
        group_labels_list if group_labels_list else None
    )
    
    # 绘制分布对比图
    logger.info("生成分布对比图...")
    evaluator.plot_distributions(trajectories)
    
    # 生成评估报告
    logger.info("生成评估报告...")
    evaluator.generate_report(
        batch_results,
        output_dir / 'evaluation_report.md'
    )
    
    logger.info("评估完成")
    logger.info(f"评估结果已保存到: {output_dir}")

if __name__ == "__main__":
    main() 