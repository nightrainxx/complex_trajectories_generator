"""
使用PointSelector选择轨迹的起终点对
"""

import json
import logging
from pathlib import Path

import yaml

from src.data_processing import TerrainLoader
from src.trajectory_generation import PointSelector
from src.utils.logging_utils import setup_logging

def main():
    # 设置日志
    setup_logging(
        log_file='logs/point_selection.log',
        log_level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # 加载配置
    with open('config/point_selection.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    output_dir = Path('data/output/points')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载地形数据
    terrain_loader = TerrainLoader(
        dem_file='data/input/gis/dem_30m_100km.tif',
        landcover_file='data/input/gis/landcover_30m_100km.tif'
    )
    
    # 创建点选择器
    selector = PointSelector(terrain_loader, config)
    
    try:
        # 选择起终点对
        point_pairs = selector.select_points()
        
        # 保存结果
        output_file = output_dir / 'selected_points.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(point_pairs, f, indent=2)
        logger.info(f"已保存选择的起终点对到: {output_file}")
        
        # 可视化结果
        vis_file = output_dir / 'points_visualization.png'
        selector.visualize_points(str(vis_file))
        
    except Exception as e:
        logger.error(f"选择起终点对时出错: {e}")
        raise

if __name__ == '__main__':
    main() 