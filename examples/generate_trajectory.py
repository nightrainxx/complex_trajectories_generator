"""
轨迹生成示例脚本
演示如何使用轨迹生成器生成轨迹
"""

import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.config import config
from src.core.terrain import TerrainLoader
from src.core.trajectory import EnvironmentBasedGenerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_trajectory(
        trajectory: dict,
        terrain_loader: TerrainLoader,
        save_path: Path
    ) -> None:
    """
    绘制轨迹
    
    Args:
        trajectory: 轨迹数据
        terrain_loader: 地形加载器
        save_path: 保存路径
    """
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 绘制轨迹
    x = trajectory['x']
    y = trajectory['y']
    speeds = trajectory['speed']
    
    # 轨迹和速度颜色映射
    points = ax1.scatter(
        x, y,
        c=speeds,
        cmap='viridis',
        s=10
    )
    ax1.plot(x, y, 'k-', alpha=0.3)
    ax1.set_title('轨迹（颜色表示速度）')
    ax1.set_xlabel('X坐标（米）')
    ax1.set_ylabel('Y坐标（米）')
    fig.colorbar(points, ax=ax1, label='速度（米/秒）')
    
    # 绘制速度-时间曲线
    timestamps = np.array(trajectory['timestamp'])
    timestamps = timestamps - timestamps[0]  # 从0开始
    ax2.plot(timestamps, speeds)
    ax2.set_title('速度-时间曲线')
    ax2.set_xlabel('时间（秒）')
    ax2.set_ylabel('速度（米/秒）')
    ax2.grid(True)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """主函数"""
    logger.info("开始轨迹生成示例")
    
    # 创建地形加载器
    terrain_loader = TerrainLoader()
    
    # 加载地形数据
    logger.info("加载地形数据...")
    terrain_loader.load_dem(config.paths.DEM_FILE)
    terrain_loader.load_landcover(config.paths.LANDCOVER_FILE)
    terrain_loader.load_slope(config.paths.SLOPE_FILE)
    terrain_loader.load_aspect(config.paths.ASPECT_FILE)
    
    # 创建轨迹生成器
    generator = EnvironmentBasedGenerator(
        terrain_loader=terrain_loader,
        dt=config.motion.DT,
        max_waypoints=10,
        min_waypoint_dist=1000.0,
        max_waypoint_dist=5000.0
    )
    
    # 设置起点和终点
    start_point = (0.0, 0.0)
    end_point = (50000.0, 50000.0)
    
    # 生成轨迹
    logger.info("生成轨迹...")
    trajectory = generator.generate_trajectory(
        start_point,
        end_point
    )
    
    # 保存轨迹数据
    output_dir = config.paths.TRAJECTORY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trajectory_file = output_dir / "example_trajectory.json"
    with open(trajectory_file, 'w') as f:
        json.dump(trajectory, f, indent=2)
    logger.info(f"轨迹数据已保存至: {trajectory_file}")
    
    # 绘制轨迹
    plot_file = output_dir / "example_trajectory.png"
    plot_trajectory(trajectory, terrain_loader, plot_file)
    logger.info(f"轨迹图形已保存至: {plot_file}")
    
    logger.info("示例完成")

if __name__ == '__main__':
    main() 