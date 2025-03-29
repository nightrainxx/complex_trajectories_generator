"""
轨迹可视化模块
在DEM背景下展示轨迹
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from matplotlib.colors import LightSource
from typing import Optional

from src.utils.config import config

def visualize_trajectory_on_dem(
        trajectory_file: Path,
        dem_file: Path,
        output_file: Path,
        landcover_file: Optional[Path] = None,
        show_waypoints: bool = True,
        show_speed: bool = True
    ):
    """
    在DEM背景下可视化轨迹
    
    Args:
        trajectory_file: 轨迹文件路径
        dem_file: DEM文件路径
        output_file: 输出文件路径
        landcover_file: 土地覆盖文件路径（可选）
        show_waypoints: 是否显示路径点
        show_speed: 是否显示速度信息
    """
    # 加载轨迹数据
    with open(trajectory_file, 'r') as f:
        trajectory = json.load(f)
    
    # 加载DEM数据
    with rasterio.open(dem_file) as src:
        dem = src.read(1)
        extent = [
            src.bounds.left,
            src.bounds.right,
            src.bounds.bottom,
            src.bounds.top
        ]
    
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    
    # 添加主图（DEM阴影+轨迹）
    ax1 = plt.subplot(111)
    
    # 计算DEM阴影
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem, vert_exag=2)
    
    # 绘制DEM阴影
    ax1.imshow(
        hillshade,
        extent=extent,
        cmap='gray',
        alpha=0.5
    )
    
    # 绘制等高线
    levels = np.linspace(dem.min(), dem.max(), 20)
    contours = ax1.contour(
        dem,
        levels=levels,
        extent=extent,
        colors='k',
        alpha=0.3,
        linewidths=0.5
    )
    
    # 绘制轨迹
    points = np.array(list(zip(trajectory['x'], trajectory['y'])))
    speeds = np.array(trajectory['speed'])
    
    # 使用速度作为颜色
    norm = plt.Normalize(speeds.min(), speeds.max())
    line = ax1.scatter(
        points[:, 0],
        points[:, 1],
        c=speeds,
        cmap='viridis',
        norm=norm,
        s=2,
        alpha=0.8
    )
    
    # 添加起点和终点标记
    ax1.plot(
        points[0, 0],
        points[0, 1],
        'go',
        markersize=10,
        label='起点'
    )
    ax1.plot(
        points[-1, 0],
        points[-1, 1],
        'ro',
        markersize=10,
        label='终点'
    )
    
    # 添加图例和颜色条
    plt.colorbar(line, label='速度 (m/s)')
    ax1.legend()
    
    # 设置标题和轴标签
    ax1.set_title('轨迹可视化（DEM背景）')
    ax1.set_xlabel('X坐标 (m)')
    ax1.set_ylabel('Y坐标 (m)')
    
    # 保存图形
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    # 设置输入输出路径
    trajectory_file = config.paths.TRAJECTORY_DIR / "example_trajectory.json"
    dem_file = config.paths.DEM_FILE
    output_file = config.paths.TRAJECTORY_DIR / "trajectory_on_dem.png"
    landcover_file = config.paths.LANDCOVER_FILE
    
    # 可视化轨迹
    visualize_trajectory_on_dem(
        trajectory_file=trajectory_file,
        dem_file=dem_file,
        output_file=output_file,
        landcover_file=landcover_file
    )
    
    print(f"轨迹可视化已保存至: {output_file}")

if __name__ == '__main__':
    main() 