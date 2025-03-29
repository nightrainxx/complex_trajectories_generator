"""
验证脚本
实现验证方式一：给定真实路径骨架进行模拟，对比动态行为
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

# 从统一配置文件导入配置
from config import config

# 导入相关模块
from src.core.terrain.loader import TerrainLoader
from src.core.learning.motion_pattern_learner import MotionPatternLearner
from src.core.motion.simulator import MotionSimulator, EnvironmentMaps
from src.core.validation.validator import TrajectoryValidator

def load_oord_trajectory(trajectory_file: Path) -> pd.DataFrame:
    """
    加载OORD轨迹数据
    
    Args:
        trajectory_file: 轨迹文件路径
        
    Returns:
        pd.DataFrame: 轨迹数据框
    """
    df = pd.read_csv(trajectory_file)
    
    # 确保必要的列存在
    required_columns = [
        'timestamp', 'longitude', 'latitude',
        'speed_mps', 'acceleration_mps2', 'turn_rate_dps'
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"轨迹文件缺少必要的列: {col}")
            
    return df

def extract_path_skeleton(
        trajectory_df: pd.DataFrame,
        terrain_loader: TerrainLoader,
        min_distance: float = 5.0  # 最小点间距（米）
    ) -> List[Tuple[int, int]]:
    """
    提取路径骨架
    
    Args:
        trajectory_df: 轨迹数据框
        terrain_loader: 地形加载器实例
        min_distance: 最小点间距（米）
        
    Returns:
        List[Tuple[int, int]]: 路径点列表（像素坐标）
    """
    path_points = []
    last_point = None
    
    for _, row in trajectory_df.iterrows():
        # 转换到像素坐标
        row_idx, col_idx = terrain_loader.lonlat_to_pixel(
            row['longitude'],
            row['latitude']
        )
        current_point = np.array([row_idx, col_idx])
        
        # 如果是第一个点，直接添加
        if last_point is None:
            path_points.append(tuple(map(int, current_point)))
            last_point = current_point
            continue
            
        # 计算与上一个点的距离
        distance = np.linalg.norm(current_point - last_point)
        
        # 如果距离超过阈值，添加新点
        if distance >= min_distance:
            path_points.append(tuple(map(int, current_point)))
            last_point = current_point
            
    # 确保添加最后一个点
    if len(path_points) > 0 and path_points[-1] != tuple(map(int, current_point)):
        path_points.append(tuple(map(int, current_point)))
        
    return path_points

def save_trajectory_to_csv(trajectory_points, output_path):
    """
    将轨迹点保存为CSV文件
    
    Args:
        trajectory_points: 轨迹点列表
        output_path: 输出文件路径
    """
    # 创建包含轨迹点信息的数据框
    df = pd.DataFrame([
        {
            'timestamp': point[0],
            'longitude': point[1],
            'latitude': point[2],
            'speed': point[3],
            'heading': point[4]
        }
        for point in trajectory_points
    ])
    
    # 保存为CSV文件
    df.to_csv(output_path, index=False)
    
    return df

def plot_trajectories_comparison(original_df, natural_df, forced_df, output_path):
    """
    绘制轨迹对比图
    
    Args:
        original_df: 原始轨迹数据
        natural_df: 自然模式模拟轨迹数据
        forced_df: 强制模式模拟轨迹数据
        output_path: 输出图像路径
    """
    plt.figure(figsize=(15, 10))
    
    # 1. 绘制轨迹路径对比
    plt.subplot(221)
    plt.plot(original_df['longitude'], original_df['latitude'], 'k-', label='原始轨迹')
    plt.plot(natural_df['longitude'], natural_df['latitude'], 'b-', label='自然模式模拟')
    plt.plot(forced_df['longitude'], forced_df['latitude'], 'r-', label='强制模式模拟')
    plt.title('轨迹路径对比')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend()
    plt.grid(True)
    
    # 2. 绘制速度对比
    plt.subplot(222)
    if 'speed_mps' in original_df.columns:
        plt.plot(original_df.index, original_df['speed_mps'], 'k-', label='原始速度')
    elif 'speed' in original_df.columns:
        plt.plot(original_df.index, original_df['speed'], 'k-', label='原始速度')
        
    plt.plot(natural_df.index, natural_df['speed'], 'b-', label='自然模式速度')
    plt.plot(forced_df.index, forced_df['speed'], 'r-', label='强制模式速度')
    plt.title('速度对比')
    plt.xlabel('时间点索引')
    plt.ylabel('速度 (m/s)')
    plt.legend()
    plt.grid(True)
    
    # 3. 绘制速度直方图
    plt.subplot(223)
    plt.hist(natural_df['speed'], bins=20, alpha=0.5, label='自然模式速度', color='blue')
    plt.hist(forced_df['speed'], bins=20, alpha=0.5, label='强制模式速度', color='red')
    plt.title('速度分布对比')
    plt.xlabel('速度 (m/s)')
    plt.ylabel('频率')
    plt.legend()
    plt.grid(True)
    
    # 4. 绘制速度-时间图
    plt.subplot(224)
    plt.plot(natural_df['timestamp'], natural_df['speed'], 'b-', label='自然模式')
    plt.plot(forced_df['timestamp'], forced_df['speed'], 'r-', label='强制模式')
    plt.title('速度-时间图')
    plt.xlabel('时间 (s)')
    plt.ylabel('速度 (m/s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 加载地形数据
    logger.info("加载地形数据...")
    terrain_loader = TerrainLoader()
    terrain_loader.load_dem(
        "/home/yzc/data/Sucess_or_Die/complex_trajectories_generator/data/terrain/dem.tif"
    )
    terrain_loader.load_landcover(
        "/home/yzc/data/Sucess_or_Die/complex_trajectories_generator/data/terrain/landcover.tif"
    )
    
    # 获取环境组学习结果文件路径
    environment_groups_path = config['paths']['ENVIRONMENT_GROUPS_FILE']
    
    # 读取测试轨迹
    test_trajectory_path = Path(
        "/home/yzc/data/Sucess_or_Die/complex_trajectories_generator/data/oord/trajectory_1_core.csv"
    )
    logger.info(f"选择测试轨迹: {test_trajectory_path}")
    
    # 从测试轨迹学习运动模式
    logger.info("从测试轨迹学习运动模式...")
    logger.info(f"学习轨迹: {test_trajectory_path}")
    pattern_learner = MotionPatternLearner(
        terrain_loader=terrain_loader,
        min_samples_per_group=20
    )
    pattern_learner.learn_from_single_trajectory(test_trajectory_path)
    
    # 保存环境组数据
    logger.info(f"保存环境组数据到: {environment_groups_path}")
    pattern_learner.save_environment_groups(environment_groups_path)
    
    # 生成环境地图
    logger.info("生成环境地图...")
    env_maps = pattern_learner.generate_environment_maps()
    
    # 输出环境地图统计信息
    typical_speed_values = env_maps.typical_speed[env_maps.typical_speed > 0]
    max_speed_values = env_maps.max_speed[env_maps.max_speed > 0]
    stddev_values = env_maps.speed_stddev[env_maps.speed_stddev > 0]
    
    if len(typical_speed_values) > 0:
        logger.info(f"典型速度范围: [{np.min(typical_speed_values):.2f}, {np.max(typical_speed_values):.2f}] m/s")
        logger.info(f"典型速度平均值: {np.mean(typical_speed_values):.2f} m/s")
        logger.info(f"典型速度不同值数量: {len(np.unique(np.round(typical_speed_values, 2)))}")
    
    if len(max_speed_values) > 0:
        logger.info(f"最大速度范围: [{np.min(max_speed_values):.2f}, {np.max(max_speed_values):.2f}] m/s")
        
    if len(stddev_values) > 0:
        logger.info(f"速度标准差范围: [{np.min(stddev_values):.2f}, {np.max(stddev_values):.2f}] m/s")
    
    # 加载测试轨迹
    logger.info("加载测试轨迹...")
    test_trajectory = pd.read_csv(test_trajectory_path)
    
    # 提取路径骨架
    logger.info("提取路径骨架...")
    path_skeleton = test_trajectory[['longitude', 'latitude']].values
    
    # 降采样路径
    sample_rate = max(1, len(path_skeleton) // 200)  # 限制为最多200个点
    path_skeleton = path_skeleton[::sample_rate]
    logger.info(f"降采样后路径包含 {len(path_skeleton)} 个点")
    
    # 确保路径骨架有足够的点
    if len(path_skeleton) < 2:
        logger.error("路径骨架点数不足，无法进行模拟")
        return
    
    # 如果是英国的轨迹（经度为负数），将其转换到我们的DEM区域
    sample_lon, sample_lat = path_skeleton[0]
    if sample_lon < 0:
        logger.info("检测到英国区域轨迹，正在转换坐标...")
        # 将英国的轨迹平移到中国区域
        # 假设目标区域在东经116度，北纬40度附近（北京）
        target_lon, target_lat = 116.3, 40.0
        min_lon, min_lat = np.min(path_skeleton, axis=0)
        offset_lon = target_lon - min_lon
        offset_lat = target_lat - min_lat
        
        # 应用平移
        path_skeleton_adjusted = path_skeleton.copy()
        path_skeleton_adjusted[:, 0] += offset_lon
        path_skeleton_adjusted[:, 1] += offset_lat
        
        logger.info(f"已将轨迹从 ({min_lon:.2f}, {min_lat:.2f}) 平移到 ({min_lon + offset_lon:.2f}, {min_lat + offset_lat:.2f})")
        path_skeleton = path_skeleton_adjusted
    
    # 转换路径骨架为UTM坐标
    path_utm = []
    for lon, lat in path_skeleton:
        try:
            east, north = terrain_loader.lonlat_to_utm(lon, lat)
            path_utm.append((east, north))
        except Exception as e:
            logger.error(f"坐标转换错误: {lon}, {lat} -> {e}")
    
    logger.info(f"已将路径转换为UTM坐标，包含 {len(path_utm)} 个点")
    
    # 创建运动模拟器
    from src.generator.motion_simulator import MotionSimulator, EnvironmentParams
    
    simulator = MotionSimulator()
    
    # 定义环境参数获取函数
    def get_env_params(x, y):
        # 将UTM坐标转换为像素坐标
        try:
            i, j = terrain_loader.utm_to_pixel(x, y)
            i, j = int(i), int(j)
            
            # 检查坐标是否在范围内
            height, width = env_maps.typical_speed.shape
            if 0 <= i < height and 0 <= j < width:
                return EnvironmentParams(
                    max_speed=float(env_maps.max_speed[i, j]),
                    typical_speed=float(env_maps.typical_speed[i, j]),
                    speed_stddev=float(env_maps.speed_stddev[i, j]),
                    slope_magnitude=float(env_maps.slope_magnitude[i, j]),
                    slope_aspect=float(env_maps.slope_aspect[i, j]),
                    landcover_code=int(env_maps.landcover[i, j]) if env_maps.landcover is not None else 0
                )
        except Exception as e:
            logger.error(f"获取环境参数错误: {x}, {y} -> {e}")
        
        # 默认返回默认参数
        return EnvironmentParams(
            max_speed=10.0,
            typical_speed=5.0,
            speed_stddev=1.0
        )
    
    # 运行两种模式的模拟
    logger.info("运行模式1: force_path=False（允许自然速度变化）")
    trajectory_natural = simulator.simulate_motion(
        path_utm,
        get_env_params,
        force_path=False
    )
    
    logger.info("运行模式2: force_path=True（严格沿路径）")
    trajectory_forced = simulator.simulate_motion(
        path_utm,
        get_env_params,
        force_path=True
    )
    
    # 保存模拟轨迹
    natural_df = save_trajectory_to_csv(trajectory_natural, "output/simulated_trajectory_natural.csv")
    forced_df = save_trajectory_to_csv(trajectory_forced, "output/simulated_trajectory_forced.csv")
    logger.info("模拟轨迹已保存")
    
    # 对比两种模式的速度差异
    speeds_natural = [point[3] for point in trajectory_natural]
    speeds_forced = [point[3] for point in trajectory_forced]
    
    logger.info("速度统计对比:")
    logger.info(f"自然模式: 平均={np.mean(speeds_natural):.2f}m/s, 最大={np.max(speeds_natural):.2f}m/s, "
              f"最小={np.min(speeds_natural):.2f}m/s, 标准差={np.std(speeds_natural):.2f}m/s, "
              f"不同值数量={len(set([round(s, 1) for s in speeds_natural]))}")
    logger.info(f"强制模式: 平均={np.mean(speeds_forced):.2f}m/s, 最大={np.max(speeds_forced):.2f}m/s, "
              f"最小={np.min(speeds_forced):.2f}m/s, 标准差={np.std(speeds_forced):.2f}m/s, "
              f"不同值数量={len(set([round(s, 1) for s in speeds_forced]))}")
    
    # 绘制轨迹对比图
    plot_trajectories_comparison(
        test_trajectory,
        natural_df,
        forced_df,
        "output/trajectory_comparison.png"
    )
    logger.info("轨迹对比图已保存到 output/trajectory_comparison.png")

if __name__ == "__main__":
    main() 