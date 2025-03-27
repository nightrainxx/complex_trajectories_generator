"""主程序入口

用于启动轨迹生成流程，包括参数解析和日志配置。
"""

import argparse
import logging
from pathlib import Path
import json
import sys

from src.generator.batch_generator import BatchGenerator
from config import *

def setup_logging(log_file: str = None):
    """配置日志系统

    Args:
        log_file: 日志文件路径，如果为None则只输出到控制台
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def create_config_json():
    """创建用于批处理的配置JSON文件"""
    config = {
        'output_dir': str(OUTPUT_DIR),
        'dem_path': str(DEM_PATH),
        'landcover_path': str(LANDCOVER_PATH),
        'slope_magnitude_path': str(SLOPE_MAGNITUDE_PATH),
        'slope_aspect_path': str(SLOPE_ASPECT_PATH),
        'NUM_TRAJECTORIES_TO_GENERATE': NUM_TRAJECTORIES_TO_GENERATE,
        'NUM_END_POINTS': NUM_END_POINTS,
        'NUM_TRAJECTORIES_PER_END': NUM_TRAJECTORIES_PER_END,
        'MIN_START_END_DISTANCE_METERS': MIN_START_END_DISTANCE_METERS,
        'URBAN_LANDCOVER_CODES': URBAN_LANDCOVER_CODES,
        'IMPASSABLE_LANDCOVER_CODES': IMPASSABLE_LANDCOVER_CODES,
        'SLOPE_BINS': SLOPE_BINS,
        'LANDCOVER_SPEED_FACTORS': LANDCOVER_SPEED_FACTORS,
        'MOTION_CONSTRAINTS': MOTION_CONSTRAINTS,
        'TERRAIN_CONSTRAINTS': TERRAIN_CONSTRAINTS,
        'BASE_SPEED': BASE_SPEED,
        'MAX_SLOPE': MAX_SLOPE,
        'EVALUATION_METRICS': EVALUATION_METRICS
    }

    config_path = OUTPUT_DIR / 'batch_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return config_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='复杂轨迹生成器')
    parser.add_argument('--log-file', help='日志文件路径')
    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)

    try:
        # 检查必要的输入文件
        if not DEM_PATH.exists():
            logger.error(f"DEM文件不存在: {DEM_PATH}")
            sys.exit(1)
        if not LANDCOVER_PATH.exists():
            logger.error(f"土地覆盖文件不存在: {LANDCOVER_PATH}")
            sys.exit(1)

        # 创建配置文件
        config_path = create_config_json()
        logger.info(f"配置文件已创建: {config_path}")

        # 初始化并运行批处理生成器
        generator = BatchGenerator(str(config_path))
        generator.generate_batch()

        logger.info("轨迹生成完成")

    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 