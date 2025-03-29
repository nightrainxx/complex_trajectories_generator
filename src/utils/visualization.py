"""
可视化工具模块
提供轨迹可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional

def plot_trajectory_on_map(
        dem_data: np.ndarray,
        real_path: np.ndarray,
        sim_path: np.ndarray,
        output_file: Path,
        title: str = "轨迹对比"
    ) -> None:
    """
    在DEM地图上绘制轨迹对比图
    
    Args:
        dem_data: DEM高程数据
        real_path: 真实轨迹点坐标 [[row, col], ...]
        sim_path: 模拟轨迹点坐标 [[row, col], ...]
        output_file: 输出文件路径
        title: 图表标题
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制DEM地图作为背景
    plt.imshow(
        dem_data,
        cmap='terrain',
        aspect='equal'
    )
    plt.colorbar(label='高程 (米)')
    
    # 绘制真实轨迹
    plt.plot(
        real_path[:, 1],  # col坐标作为x
        real_path[:, 0],  # row坐标作为y
        'b-',
        linewidth=2,
        label='真实轨迹',
        alpha=0.8
    )
    
    # 绘制模拟轨迹
    plt.plot(
        sim_path[:, 1],
        sim_path[:, 0],
        'r--',
        linewidth=2,
        label='模拟轨迹',
        alpha=0.8
    )
    
    # 标记起点和终点
    plt.plot(
        real_path[0, 1],
        real_path[0, 0],
        'go',
        markersize=10,
        label='起点'
    )
    plt.plot(
        real_path[-1, 1],
        real_path[-1, 0],
        'ro',
        markersize=10,
        label='终点'
    )
    
    plt.title(title)
    plt.xlabel('列坐标')
    plt.ylabel('行坐标')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_speed_map(
        speed_map: np.ndarray,
        output_file: Path,
        title: str = "速度地图",
        cmap: str = 'viridis'
    ) -> None:
    """
    绘制速度地图
    
    Args:
        speed_map: 速度地图数据
        output_file: 输出文件路径
        title: 图表标题
        cmap: 颜色映射
    """
    plt.figure(figsize=(10, 8))
    
    im = plt.imshow(speed_map, cmap=cmap)
    plt.colorbar(im, label='速度 (米/秒)')
    
    plt.title(title)
    plt.xlabel('列坐标')
    plt.ylabel('行坐标')
    plt.grid(True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_terrain_maps(
        dem_data: np.ndarray,
        slope_magnitude: np.ndarray,
        slope_aspect: np.ndarray,
        landcover: Optional[np.ndarray] = None,
        output_dir: Path = None
    ) -> None:
    """
    绘制地形相关的地图
    
    Args:
        dem_data: DEM高程数据
        slope_magnitude: 坡度大小数据
        slope_aspect: 坡向数据
        landcover: 土地覆盖类型数据
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = Path('outputs/terrain_maps')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制DEM地图
    plt.figure(figsize=(10, 8))
    im = plt.imshow(dem_data, cmap='terrain')
    plt.colorbar(im, label='高程 (米)')
    plt.title('DEM高程图')
    plt.xlabel('列坐标')
    plt.ylabel('行坐标')
    plt.grid(True)
    plt.savefig(
        output_dir / 'dem_map.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # 绘制坡度图
    plt.figure(figsize=(10, 8))
    im = plt.imshow(slope_magnitude, cmap='YlOrRd')
    plt.colorbar(im, label='坡度 (度)')
    plt.title('坡度图')
    plt.xlabel('列坐标')
    plt.ylabel('行坐标')
    plt.grid(True)
    plt.savefig(
        output_dir / 'slope_magnitude_map.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # 绘制坡向图
    plt.figure(figsize=(10, 8))
    im = plt.imshow(slope_aspect, cmap='hsv')
    plt.colorbar(im, label='坡向 (度)')
    plt.title('坡向图')
    plt.xlabel('列坐标')
    plt.ylabel('行坐标')
    plt.grid(True)
    plt.savefig(
        output_dir / 'slope_aspect_map.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # 如果有土地覆盖数据，绘制土地覆盖图
    if landcover is not None:
        plt.figure(figsize=(10, 8))
        im = plt.imshow(landcover, cmap='tab20')
        plt.colorbar(im, label='土地覆盖类型')
        plt.title('土地覆盖图')
        plt.xlabel('列坐标')
        plt.ylabel('行坐标')
        plt.grid(True)
        plt.savefig(
            output_dir / 'landcover_map.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close() 