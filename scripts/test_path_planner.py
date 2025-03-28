"""
测试路径规划器功能
"""

import logging
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json

from src.data_processing import TerrainLoader, EnvironmentMapper
from src.trajectory_generation import PathPlanner
from src.utils.logging_utils import setup_logging

def main():
    # 设置日志
    setup_logging(
        log_file='logs/path_planning.log',
        log_level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    output_dir = Path('data/output/path_planning')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载地形数据
    logger.info("加载地形数据...")
    terrain_loader = TerrainLoader()
    terrain_loader.load_dem("data/input/gis/dem_30m_100km.tif")
    terrain_loader.load_landcover("data/input/gis/landcover_30m_100km.tif")
    
    # 加载学习到的运动模式
    logger.info("加载运动模式...")
    with open("data/output/intermediate/learned_patterns.pkl", 'rb') as f:
        motion_patterns = pickle.load(f)
    
    # 创建环境地图生成器
    mapper = EnvironmentMapper(
        terrain_loader=terrain_loader,
        motion_patterns=motion_patterns,
        output_dir="data/output/intermediate"
    )
    
    # 生成环境地图
    mapper.generate_maps()
    
    # 创建路径规划器
    planner = PathPlanner(
        terrain_loader=terrain_loader,
        environment_mapper=mapper,
        config={}
    )
    
    # 加载测试用的起终点对
    with open("data/output/points/selected_points.json", 'r') as f:
        point_pairs = json.load(f)
    
    # 选择第一对点进行测试
    start_coord, end_coord = point_pairs[0]
    
    # 转换为像素坐标
    start_point = terrain_loader.transform_coordinates(
        start_coord[0],
        start_coord[1]
    )
    end_point = terrain_loader.transform_coordinates(
        end_coord[0],
        end_coord[1]
    )
    
    # 规划路径
    logger.info("开始路径规划...")
    path = planner.plan_path(start_point, end_point)
    
    if path is None:
        logger.error("未找到有效路径")
        return
    
    # 平滑路径
    smoothed_path = planner.smooth_path(path)
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 绘制成本图
    plt.imshow(
        np.log(planner.cost_map + 1),
        cmap='YlOrRd',
        alpha=0.7
    )
    plt.colorbar(label='Log(Cost + 1)')
    
    # 绘制原始路径
    path_array = np.array(path)
    plt.plot(
        path_array[:, 1],
        path_array[:, 0],
        'b-',
        label='Original Path',
        alpha=0.5
    )
    
    # 绘制平滑路径
    smoothed_array = np.array(smoothed_path)
    plt.plot(
        smoothed_array[:, 1],
        smoothed_array[:, 0],
        'g-',
        label='Smoothed Path',
        linewidth=2
    )
    
    # 标记起终点
    plt.plot(
        start_point[1],
        start_point[0],
        'go',
        label='Start',
        markersize=10
    )
    plt.plot(
        end_point[1],
        end_point[0],
        'ro',
        label='End',
        markersize=10
    )
    
    plt.title('Path Planning Result')
    plt.legend()
    
    # 保存结果
    output_file = output_dir / 'path_planning_test.png'
    plt.savefig(output_file)
    logger.info(f"已保存可视化结果到: {output_file}")
    plt.close()
    
    # 输出路径统计信息
    path_length = len(path)
    smoothed_length = len(smoothed_path)
    total_cost = sum(planner.cost_map[r, c] for r, c in path)
    smoothed_cost = sum(planner.cost_map[r, c] for r, c in smoothed_path)
    
    logger.info("\n=== 路径统计 ===")
    logger.info(f"原始路径长度: {path_length} 点")
    logger.info(f"平滑路径长度: {smoothed_length} 点")
    logger.info(f"原始路径总成本: {total_cost:.2f}")
    logger.info(f"平滑路径总成本: {smoothed_cost:.2f}")

if __name__ == "__main__":
    main() 