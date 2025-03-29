"""
生成测试轨迹数据
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

def generate_test_trajectory():
    """生成测试轨迹数据"""
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    # 生成时间序列（100个点，每点间隔1秒）
    timestamps = np.arange(100)
    
    # 生成起点（北京附近）
    start_lon, start_lat = 116.0, 39.0
    
    # 生成轨迹点
    trajectory = []
    current_lon = start_lon
    current_lat = start_lat
    current_speed = 10.0
    current_heading = 45.0
    
    for t in timestamps:
        # 添加一些随机变化
        speed_change = np.random.normal(0, 1)  # 速度变化
        heading_change = np.random.normal(0, 5)  # 方向变化
        
        # 更新状态
        current_speed = np.clip(current_speed + speed_change, 5, 15)
        current_heading = (current_heading + heading_change) % 360
        
        # 计算位置变化（简化的平面投影）
        heading_rad = np.deg2rad(current_heading)
        lon_change = np.cos(heading_rad) * current_speed * 0.0001  # 简化的经度变化
        lat_change = np.sin(heading_rad) * current_speed * 0.0001  # 简化的纬度变化
        
        current_lon += lon_change
        current_lat += lat_change
        
        # 记录点
        trajectory.append({
            'timestamp': t,
            'longitude': current_lon,
            'latitude': current_lat,
            'speed_mps': current_speed,
            'acceleration_mps2': speed_change,
            'turn_rate_dps': heading_change
        })
    
    # 创建DataFrame
    df = pd.DataFrame(trajectory)
    
    # 保存到文件
    output_dir = Path("/home/yzc/data/Sucess_or_Die/complex_trajectories_generator/data/oord")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "trajectory_1_core.csv"
    df.to_csv(output_file, index=False)
    print(f"Generated test trajectory with {len(df)} points")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    generate_test_trajectory() 