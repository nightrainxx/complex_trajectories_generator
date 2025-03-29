"""
运行轨迹实验的主脚本
"""
import os
import logging
from pathlib import Path
from src.experiments.trajectory_experiment import TrajectoryExperiment

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    # 设置基础路径
    base_dir = Path('/home/yzc/data/Sucess_or_Die/complex_trajectories_generator')
    core_trajectories_dir = base_dir / 'data/core_trajectories'
    experiments_dir = base_dir / 'data/experiments'
    
    # 设置DEM和地表覆盖文件路径
    dem_file = str(base_dir / 'data/raw/dem/dem_30m_100km.tif')
    landcover_file = str(base_dir / 'data/raw/landcover/landcover_30m_100km.tif')
    
    logger.info(f"使用DEM文件: {dem_file}")
    logger.info(f"使用地表覆盖文件: {landcover_file}")
    
    # 获取所有轨迹文件
    trajectory_files = sorted(
        core_trajectories_dir.glob('sequence_*_core.csv')
    )
    
    logger.info(f"找到 {len(trajectory_files)} 个轨迹文件")
    
    # 运行每个轨迹的实验
    for trajectory_file in trajectory_files:
        # 从文件名中提取序号
        sequence_id = int(trajectory_file.name.split('_')[1])
        
        # 创建实验目录
        exp_dir = experiments_dir / f'sequence_{sequence_id}'
        
        logger.info(f"开始处理轨迹 {sequence_id}")
        
        # 创建并运行实验
        experiment = TrajectoryExperiment(
            sequence_id=sequence_id,
            exp_dir=exp_dir,
            dem_file=dem_file,
            landcover_file=landcover_file
        )
        experiment.run(str(trajectory_file))
        
        logger.info(f"轨迹 {sequence_id} 处理完成")
        
    logger.info("所有轨迹实验完成")

if __name__ == '__main__':
    main() 