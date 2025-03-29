"""
轨迹数据预处理脚本
"""
import os
import logging
from pathlib import Path
from src.data_processing.trajectory_processor import TrajectoryProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    # 设置文件路径
    base_dir = Path("/home/yzc/data/Sucess_or_Die/complex_trajectories_generator")
    dem_file = base_dir / "data/terrain/dem.tif"
    landcover_file = base_dir / "data/terrain/landcover.tif"
    input_file = base_dir / "data/oord/trajectory_1_core.csv"
    output_file = base_dir / "data/processed/trajectory_1_processed.csv"
    
    try:
        # 创建预处理器
        processor = TrajectoryProcessor(str(dem_file), str(landcover_file))
        
        # 处理轨迹数据
        processor.process_trajectory_file(str(input_file), str(output_file))
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 