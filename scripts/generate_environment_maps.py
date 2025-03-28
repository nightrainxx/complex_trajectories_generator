"""
环境地图生成脚本
根据地形数据和学习到的运动模式生成增强的环境地图
"""

import logging
from pathlib import Path

from src.data_processing import TerrainLoader, EnvironmentMapper
from src.utils.logging_utils import setup_logging

def main():
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("开始生成环境地图")
    
    # 加载地形数据
    logger.info("加载地形数据...")
    terrain_loader = TerrainLoader()
    terrain_loader.load_dem("data/input/gis/dem_30m_100km.tif")
    terrain_loader.load_landcover("data/input/gis/landcover_30m_100km.tif")
    
    # 加载学习到的运动模式
    logger.info("加载运动模式...")
    import pickle
    with open("data/output/intermediate/learned_patterns.pkl", 'rb') as f:
        motion_patterns = pickle.load(f)
    
    # 创建环境地图生成器
    mapper = EnvironmentMapper(
        terrain_loader=terrain_loader,
        motion_patterns=motion_patterns,
        output_dir="data/output/intermediate"
    )
    
    try:
        # 生成地图
        mapper.generate_maps()
        logger.info("环境地图生成完成")
        
    except Exception as e:
        logger.error(f"地图生成失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 