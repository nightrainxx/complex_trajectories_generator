"""
测试轨迹生成器功能
"""

import logging
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np

from src.data_processing import TerrainLoader, EnvironmentMapper
from src.trajectory_generation import PathPlanner, TrajectoryGenerator
from src.utils.logging_utils import setup_logging

def main():
    # 设置日志
    setup_logging(
        log_file='logs/trajectory_generation.log',
        log_level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    output_dir = Path('data/output/trajectory_generation')
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
    
    # 创建轨迹生成器
    generator_config = {
        'dt': 0.1,  # 时间步长（秒）
        'MAX_ACCELERATION': 2.0,  # 最大加速度（米/秒²）
        'MAX_DECELERATION': 3.0,  # 最大减速度（米/秒²）
        'MAX_SPEED': 20.0,  # 最大速度（米/秒）
        'MIN_SPEED': 0.1  # 最小速度（米/秒）
    }
    
    generator = TrajectoryGenerator(
        terrain_loader=terrain_loader,
        environment_mapper=mapper,
        config=generator_config
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
    
    # 生成轨迹
    logger.info("开始生成轨迹...")
    trajectory = generator.generate_trajectory(path)
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 绘制地形
    plt.imshow(
        terrain_loader.dem_data,
        cmap='terrain',
        alpha=0.7
    )
    plt.colorbar(label='Elevation (m)')
    
    # 绘制轨迹
    positions = np.array(trajectory['positions'])
    plt.plot(
        positions[:, 1],
        positions[:, 0],
        'g-',
        label='Trajectory',
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
    
    plt.title('Generated Trajectory')
    plt.legend()
    
    # 保存轨迹图
    output_file = output_dir / 'trajectory_test.png'
    plt.savefig(output_file)
    logger.info(f"已保存轨迹图到: {output_file}")
    plt.close()
    
    # 绘制速度曲线
    plt.figure(figsize=(12, 6))
    plt.plot(
        trajectory['timestamps'],
        trajectory['speeds'],
        'b-',
        label='Speed'
    )
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed Profile')
    plt.legend()
    
    # 保存速度图
    output_file = output_dir / 'speed_profile.png'
    plt.savefig(output_file)
    logger.info(f"已保存速度图到: {output_file}")
    plt.close()
    
    # 输出轨迹统计信息
    total_time = trajectory['timestamps'][-1]
    total_distance = sum(
        np.sqrt(
            sum((a - b)**2 for a, b in zip(p1, p2))
        ) * terrain_loader.resolution
        for p1, p2 in zip(
            trajectory['positions'][:-1],
            trajectory['positions'][1:]
        )
    )
    avg_speed = total_distance / total_time
    max_speed = max(trajectory['speeds'])
    min_speed = min(trajectory['speeds'])
    
    logger.info("\n=== 轨迹统计 ===")
    logger.info(f"总时间: {total_time:.2f} 秒")
    logger.info(f"总距离: {total_distance:.2f} 米")
    logger.info(f"平均速度: {avg_speed:.2f} 米/秒")
    logger.info(f"最大速度: {max_speed:.2f} 米/秒")
    logger.info(f"最小速度: {min_speed:.2f} 米/秒")
    
    # 保存轨迹数据
    output_file = output_dir / 'trajectory_test.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamps': trajectory['timestamps'],
            'coordinates': trajectory['coordinates'],
            'speeds': trajectory['speeds'],
            'headings': trajectory['headings']
        }, f, indent=2)
    logger.info(f"已保存轨迹数据到: {output_file}")

if __name__ == "__main__":
    main() 