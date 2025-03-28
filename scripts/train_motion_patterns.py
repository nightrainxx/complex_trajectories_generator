"""
运动模式学习脚本
从OORD数据中学习目标在不同环境下的运动特性
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing import TerrainLoader, OORDProcessor, MotionPatternLearner
from src.utils.logging_utils import setup_logging

def main():
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("开始运动模式学习")
    
    # 创建输出目录
    output_dir = Path("data/output/intermediate")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载地形数据
    logger.info("加载地形数据...")
    terrain_loader = TerrainLoader()
    terrain_loader.load_dem("data/input/gis/dem_30m_100km.tif")
    terrain_loader.load_landcover("data/input/gis/landcover_30m_100km.tif")
    
    # 加载OORD轨迹数据
    logger.info("加载OORD轨迹数据...")
    oord_processor = OORDProcessor(terrain_loader)
    trajectories = []
    
    oord_dir = Path("data/input/oord")
    for file in oord_dir.glob("*.csv"):
        try:
            traj = oord_processor.load_trajectory(file)
            if traj is not None and len(traj) > 0:
                trajectories.append(traj)
                logger.info(f"成功加载轨迹: {file.name}")
            else:
                logger.warning(f"轨迹为空: {file.name}")
        except Exception as e:
            logger.error(f"加载轨迹失败 {file.name}: {str(e)}")
    
    if not trajectories:
        logger.error("未找到有效的轨迹数据")
        return
    
    logger.info(f"共加载 {len(trajectories)} 条轨迹")
    
    # 创建运动模式学习器
    logger.info("开始学习运动模式...")
    learner = MotionPatternLearner(terrain_loader)
    
    try:
        # 执行学习
        learner.learn_from_trajectories(trajectories)
        
        # 保存学习结果
        output_file = output_dir / "learned_patterns.pkl"
        learner.save_patterns(str(output_file))
        logger.info(f"学习结果已保存到: {output_file}")
        
        # 输出学习结果摘要
        patterns = learner.get_learned_patterns()
        
        logger.info("\n=== 学习结果摘要 ===")
        
        # 坡度-速度关系
        slope_speed = patterns['slope_speed_model']
        logger.info("\n坡度-速度关系:")
        logger.info(f"平地速度(0-5度): {slope_speed.iloc[0]['mean']:.2f} m/s")
        logger.info(f"速度因子范围: {slope_speed['speed_factor'].min():.2f} - {slope_speed['speed_factor'].max():.2f}")
        
        # 地表类型-速度关系
        landcover_speed = patterns['landcover_speed_stats']
        logger.info("\n地表类型-速度关系:")
        for lc_type, stats in landcover_speed.iterrows():
            logger.info(f"类型 {lc_type}: 平均速度 = {stats['mean']:.2f} m/s, 速度因子 = {stats['speed_factor']:.2f}")
        
        # 转向率统计
        turn_rate = patterns['turn_rate_stats']
        logger.info("\n转向率统计:")
        logger.info(f"平均值: {turn_rate['mean']:.2f} rad/s")
        logger.info(f"标准差: {turn_rate['std']:.2f} rad/s")
        logger.info(f"90百分位: {turn_rate['percentiles']['90']:.2f} rad/s")
        
        # 加速度统计
        accel = patterns['acceleration_stats']
        logger.info("\n加速度统计:")
        logger.info(f"平均值: {accel['mean']:.2f} m/s²")
        logger.info(f"标准差: {accel['std']:.2f} m/s²")
        logger.info(f"90百分位: {accel['percentiles']['90']:.2f} m/s²")
        
        # 环境聚类
        clusters = patterns['environment_clusters']
        logger.info(f"\n环境聚类 (k={clusters['n_clusters']}):")
        for i, stats in enumerate(clusters['cluster_stats']):
            logger.info(f"\n簇 {i+1}:")
            logger.info(f"样本数: {stats['size']}")
            logger.info(f"平均坡度: {stats['slope_mean']:.1f}° (±{stats['slope_std']:.1f}°)")
            logger.info(f"主要地表类型: {stats['landcover_mode']}")
            logger.info(f"平均速度: {stats['speed_mean']:.2f} m/s (±{stats['speed_std']:.2f} m/s)")
        
    except Exception as e:
        logger.error(f"学习过程失败: {str(e)}")
        raise
    
    logger.info("运动模式学习完成")

if __name__ == "__main__":
    main() 