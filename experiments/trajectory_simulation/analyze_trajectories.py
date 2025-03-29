#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轨迹分析脚本
用于计算轨迹模拟结果的各项指标
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# 项目根目录
PROJECT_ROOT = Path('/home/yzc/data/Sucess_or_Die/complex_trajectories_generator')

# 输入文件路径
ORIGINAL_TRAJECTORY = PROJECT_ROOT / 'data/oord/trajectory_1_core.csv'
NATURAL_TRAJECTORY = PROJECT_ROOT / 'output/simulated_trajectory_natural.csv'
FORCED_TRAJECTORY = PROJECT_ROOT / 'output/simulated_trajectory_forced.csv'

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / 'experiments/trajectory_simulation'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_trajectory(file_path):
    """加载轨迹文件"""
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载轨迹: {file_path}")
        print(f"包含 {len(df)} 个点")
        
        # 打印列名，帮助调试
        print(f"列名: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        print(f"加载失败: {e}")
        return None

def compute_statistics(df, speed_col='speed'):
    """计算轨迹的速度统计量"""
    if df is None or len(df) == 0:
        return None
    
    if speed_col not in df.columns:
        speed_candidates = [col for col in df.columns if 'speed' in col.lower()]
        if speed_candidates:
            speed_col = speed_candidates[0]
            print(f"使用列 '{speed_col}' 作为速度数据")
        else:
            print(f"无法找到速度列")
            return None
    
    stats = {
        'count': len(df),
        'mean_speed': df[speed_col].mean(),
        'median_speed': df[speed_col].median(),
        'min_speed': df[speed_col].min(),
        'max_speed': df[speed_col].max(),
        'std_speed': df[speed_col].std(),
        'unique_speed_values': len(df[speed_col].round(2).unique())
    }
    
    # 如果有时间和位置数据，计算加速度和转向率
    if 'timestamp' in df.columns and 'longitude' in df.columns and 'latitude' in df.columns:
        # 计算加速度
        if len(df) > 1:
            df = df.sort_values('timestamp')
            df['speed_diff'] = df[speed_col].diff()
            df['time_diff'] = df['timestamp'].diff()
            df.loc[df['time_diff'] > 0, 'acceleration'] = df['speed_diff'] / df['time_diff']
            
            stats['mean_acceleration'] = df['acceleration'].mean()
            stats['max_acceleration'] = df['acceleration'].max()
            stats['min_acceleration'] = df['acceleration'].min()
            stats['std_acceleration'] = df['acceleration'].std()
    
    return stats

def calculate_path_fidelity(original_df, simulated_df):
    """计算路径一致性指标"""
    if original_df is None or simulated_df is None:
        return None
    
    # 确保原始轨迹和模拟轨迹有经纬度列
    required_cols = ['longitude', 'latitude']
    if not all(col in original_df.columns for col in required_cols) or \
       not all(col in simulated_df.columns for col in required_cols):
        print("路径一致性计算需要经纬度列")
        return None
    
    # 为简化计算，我们采样原始轨迹的点
    sample_rate = max(1, len(original_df) // 100)  # 最多100个点
    original_samples = original_df.iloc[::sample_rate]
    
    # 计算每个原始点到最近的模拟点的距离
    distances = []
    for _, orig_point in original_samples.iterrows():
        # 计算当前原始点到所有模拟点的距离
        dist_to_sim = np.sqrt(
            (simulated_df['longitude'] - orig_point['longitude'])**2 + 
            (simulated_df['latitude'] - orig_point['latitude'])**2
        )
        # 找到最小距离
        min_dist = dist_to_sim.min()
        distances.append(min_dist)
    
    # 计算统计量
    fidelity = {
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'std_distance': np.std(distances)
    }
    
    return fidelity

def generate_speed_profile(original_df, natural_df, forced_df, output_path):
    """生成速度分布对比图"""
    plt.figure(figsize=(12, 8))
    
    # 速度柱状图
    plt.subplot(2, 1, 1)
    
    # 确定速度列名
    speed_cols = {
        'original': next((col for col in original_df.columns if 'speed' in col.lower()), None),
        'natural': next((col for col in natural_df.columns if 'speed' in col.lower()), None),
        'forced': next((col for col in forced_df.columns if 'speed' in col.lower()), None)
    }
    
    # 绘制直方图
    if speed_cols['original']:
        plt.hist(original_df[speed_cols['original']], alpha=0.5, bins=20, label='原始轨迹')
    if speed_cols['natural']:
        plt.hist(natural_df[speed_cols['natural']], alpha=0.5, bins=20, label='自然模式')
    if speed_cols['forced']:
        plt.hist(forced_df[speed_cols['forced']], alpha=0.5, bins=20, label='强制模式')
    
    plt.title('速度分布对比')
    plt.xlabel('速度 (m/s)')
    plt.ylabel('频率')
    plt.legend()
    plt.grid(True)
    
    # 速度-时间曲线
    plt.subplot(2, 1, 2)
    
    # 确保数据按时间排序
    if 'timestamp' in original_df.columns and speed_cols['original']:
        sorted_orig = original_df.sort_values('timestamp')
        plt.plot(sorted_orig['timestamp'], sorted_orig[speed_cols['original']], 'k-', label='原始轨迹')
        
    if 'timestamp' in natural_df.columns and speed_cols['natural']:
        sorted_natural = natural_df.sort_values('timestamp')
        plt.plot(sorted_natural['timestamp'], sorted_natural[speed_cols['natural']], 'b-', label='自然模式')
        
    if 'timestamp' in forced_df.columns and speed_cols['forced']:
        sorted_forced = forced_df.sort_values('timestamp')
        plt.plot(sorted_forced['timestamp'], sorted_forced[speed_cols['forced']], 'r-', label='强制模式')
    
    plt.title('速度-时间曲线')
    plt.xlabel('时间 (s)')
    plt.ylabel('速度 (m/s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"速度分析图已保存至: {output_path}")

def create_metrics_table(original_stats, natural_stats, forced_stats, output_path):
    """创建指标表格，保存为Markdown格式"""
    with open(output_path, 'w') as f:
        f.write("# 轨迹模拟实验指标表\n\n")
        
        # 速度统计表
        f.write("## 1. 速度统计\n\n")
        f.write("| 指标 | 原始轨迹 | 自然模式 | 强制模式 |\n")
        f.write("|------|---------|---------|----------|\n")
        
        metrics = [
            ('轨迹点数', 'count'),
            ('平均速度 (m/s)', 'mean_speed'),
            ('中位数速度 (m/s)', 'median_speed'),
            ('最小速度 (m/s)', 'min_speed'),
            ('最大速度 (m/s)', 'max_speed'),
            ('速度标准差 (m/s)', 'std_speed'),
            ('不同速度值数量', 'unique_speed_values')
        ]
        
        for name, key in metrics:
            orig_val = f"{original_stats.get(key, 'N/A'):.2f}" if original_stats and key in original_stats and isinstance(original_stats[key], (int, float)) else 'N/A'
            nat_val = f"{natural_stats.get(key, 'N/A'):.2f}" if natural_stats and key in natural_stats and isinstance(natural_stats[key], (int, float)) else 'N/A'
            force_val = f"{forced_stats.get(key, 'N/A'):.2f}" if forced_stats and key in forced_stats and isinstance(forced_stats[key], (int, float)) else 'N/A'
            
            f.write(f"| {name} | {orig_val} | {nat_val} | {force_val} |\n")
        
        # 如果有加速度数据
        if (original_stats and 'mean_acceleration' in original_stats) or \
           (natural_stats and 'mean_acceleration' in natural_stats) or \
           (forced_stats and 'mean_acceleration' in forced_stats):
            
            f.write("\n## 2. 加速度统计\n\n")
            f.write("| 指标 | 原始轨迹 | 自然模式 | 强制模式 |\n")
            f.write("|------|---------|---------|----------|\n")
            
            accel_metrics = [
                ('平均加速度 (m/s²)', 'mean_acceleration'),
                ('最大加速度 (m/s²)', 'max_acceleration'),
                ('最小加速度 (m/s²)', 'min_acceleration'),
                ('加速度标准差 (m/s²)', 'std_acceleration')
            ]
            
            for name, key in accel_metrics:
                orig_val = f"{original_stats.get(key, 'N/A'):.2f}" if original_stats and key in original_stats and isinstance(original_stats[key], (int, float)) else 'N/A'
                nat_val = f"{natural_stats.get(key, 'N/A'):.2f}" if natural_stats and key in natural_stats and isinstance(natural_stats[key], (int, float)) else 'N/A'
                force_val = f"{forced_stats.get(key, 'N/A'):.2f}" if forced_stats and key in forced_stats and isinstance(forced_stats[key], (int, float)) else 'N/A'
                
                f.write(f"| {name} | {orig_val} | {nat_val} | {force_val} |\n")
        
        print(f"指标表已保存至: {output_path}")

def main():
    """主函数"""
    print("开始分析轨迹数据...")
    
    # 加载轨迹数据
    original_df = load_trajectory(ORIGINAL_TRAJECTORY)
    natural_df = load_trajectory(NATURAL_TRAJECTORY)
    forced_df = load_trajectory(FORCED_TRAJECTORY)
    
    # 计算统计量
    print("\n计算统计量...")
    original_stats = compute_statistics(original_df, speed_col='speed_mps' if 'speed_mps' in original_df.columns else 'speed')
    natural_stats = compute_statistics(natural_df)
    forced_stats = compute_statistics(forced_df)
    
    # 计算路径一致性
    print("\n计算路径一致性...")
    natural_fidelity = calculate_path_fidelity(original_df, natural_df)
    forced_fidelity = calculate_path_fidelity(original_df, forced_df)
    
    if natural_fidelity and forced_fidelity:
        print(f"自然模式平均偏差: {natural_fidelity['mean_distance']:.6f}")
        print(f"强制模式平均偏差: {forced_fidelity['mean_distance']:.6f}")
    
    # 生成速度分析图
    print("\n生成速度分析图...")
    speed_profile_path = OUTPUT_DIR / 'speed_profile.png'
    generate_speed_profile(original_df, natural_df, forced_df, speed_profile_path)
    
    # 创建指标表
    print("\n创建指标表...")
    metrics_path = OUTPUT_DIR / 'trajectory_metrics.md'
    create_metrics_table(original_stats, natural_stats, forced_stats, metrics_path)
    
    print("\n分析完成！")
    print(f"所有结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 