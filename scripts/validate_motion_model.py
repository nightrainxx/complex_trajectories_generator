"""验证运动模型的性能

比较原始轨迹和模型生成的运动特征
"""
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from src.utils import plot_style
from src.learning.motion_pattern_learner import MotionPatternLearner
from src.data_processing.trajectory_processor import TrajectoryProcessor
import json
from math import sin, cos, sqrt, atan2, radians

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trajectory(file_path):
    """加载轨迹数据并进行预处理
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        处理后的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 重命名列以匹配预期格式
    column_mapping = {
        'heading_deg': 'heading_degrees',
        'calculated_speed_ms': 'speed_mps',
        'acceleration_magnitude_ms2': 'acceleration_mps2',
        'angular_velocity_rads': 'turn_rate_dps'
    }
    df = df.rename(columns=column_mapping)
    
    # 将角速度从弧度/秒转换为度/秒
    df['turn_rate_dps'] = df['turn_rate_dps'] * 180 / np.pi
    
    # 计算时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df['time_s'] = df['delta_time_s'].cumsum()
    
    return df

def generate_motion_features(learner, trajectory_df, processor):
    """生成运动特征
    
    Args:
        learner: 运动模式学习器
        trajectory_df: 原始轨迹DataFrame
        processor: 轨迹处理器
        
    Returns:
        生成的特征DataFrame
    """
    try:
        # 处理轨迹获取环境特征
        env_features = processor.process_trajectory(trajectory_df)
        
        # 初始化结果DataFrame
        result_df = pd.DataFrame()
        result_df['time_s'] = trajectory_df['time_s']
        result_df['distance_m'] = trajectory_df['distance_m']
        result_df['latitude'] = trajectory_df['latitude']
        result_df['longitude'] = trajectory_df['longitude']
        result_df['altitude_m'] = trajectory_df['altitude_m']
        result_df['timestamp_ms'] = trajectory_df['timestamp_ms']
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp_ms'], unit='ms')
        
        # 使用学习器预测速度
        speeds = []
        current_speed = trajectory_df['speed_mps'].iloc[0]
        
        for i in range(len(trajectory_df)):
            # 获取当前环境特征
            landcover = int(env_features['landcover'][i])
            slope_along = float(env_features['slope_along_path'][i])
            cross_slope = float(env_features['cross_slope'][i])
            
            # 预测速度
            speed_info = learner.get_speed_for_conditions(landcover, slope_along, cross_slope)
            predicted_speed = speed_info['typical_speed']
            
            # 如果有残差分析结果，添加噪声
            if 'gmm_analysis' in learner.analysis_report:
                gmm = learner.analysis_report['gmm_analysis']
                # 随机选择一个高斯分量
                component = np.random.choice(len(gmm['weights']), p=gmm['weights'])
                # 从选定的分量中采样噪声
                noise = np.random.normal(gmm['means'][component], gmm['stds'][component])
                predicted_speed += noise
            
            # 确保速度不低于最小值
            predicted_speed = max(predicted_speed, 0.1)
            speeds.append(predicted_speed)
            
            # 更新当前速度
            current_speed = predicted_speed
            
        result_df['speed_mps'] = speeds
        
        # 计算累计时间
        result_df['time_s'] = trajectory_df['delta_time_s'].cumsum()
        
        return result_df
        
    except Exception as e:
        logging.error(f"生成运动特征时出错: {str(e)}")
        raise

def calculate_cumulative_distance(df):
    """计算轨迹的累积距离
    
    Args:
        df: 包含longitude和latitude列的DataFrame
        
    Returns:
        累积距离数组（米）
    """
    distances = [0]  # 第一个点的累积距离为0
    for i in range(1, len(df)):
        dist = haversine_distance(
            df['latitude'].iloc[i-1], df['longitude'].iloc[i-1],
            df['latitude'].iloc[i], df['longitude'].iloc[i]
        )
        distances.append(distances[-1] + dist)
    
    return np.array(distances)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # 地球半径（米）
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def plot_comparison(original_df: pd.DataFrame, generated_df: pd.DataFrame, 
                    output_dir: str = "data/validation"):
    """绘制原始轨迹和生成轨迹的对比图"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 计算累积距离 (用于绘图)
        original_distance = calculate_cumulative_distance(original_df)
        generated_distance = calculate_cumulative_distance(generated_df)
        
        # 创建统计信息字典
        statistics = {}
        
        # 计算总距离
        statistics["总距离 (m)"] = original_distance[-1]
        
        # 计算总时长
        original_duration = (original_df['timestamp'].iloc[-1] - original_df['timestamp'].iloc[0]).total_seconds()
        generated_duration = (generated_df['timestamp'].iloc[-1] - generated_df['timestamp'].iloc[0]).total_seconds()
        duration_diff = generated_duration - original_duration
        duration_diff_percent = (duration_diff / original_duration) * 100
        
        statistics["总时长 (原始) (s)"] = original_duration
        statistics["总时长 (生成) (s)"] = generated_duration
        statistics["总时长差异 (s)"] = duration_diff
        statistics["总时长差异百分比 (%)"] = duration_diff_percent
        
        # 计算速度统计
        statistics["平均速度 (原始) (m/s)"] = original_df['speed_mps'].mean()
        statistics["平均速度 (生成) (m/s)"] = generated_df['speed_mps'].mean()
        statistics["速度标准差 (原始) (m/s)"] = original_df['speed_mps'].std()
        statistics["速度标准差 (生成) (m/s)"] = generated_df['speed_mps'].std()
        
        # 计算速度差异
        speed_diff = generated_df['speed_mps'].values - original_df['speed_mps'].values
        statistics["速度差异均值 (m/s)"] = np.mean(speed_diff)
        statistics["速度差异标准差 (m/s)"] = np.std(speed_diff)
        statistics["速度差异最大值 (m/s)"] = np.max(speed_diff)
        statistics["速度差异最小值 (m/s)"] = np.min(speed_diff)
        
        # 转换时间轴为分钟显示
        original_minutes = [(t - original_df['timestamp'].iloc[0]).total_seconds()/60 
                           for t in original_df['timestamp']]
        generated_minutes = [(t - generated_df['timestamp'].iloc[0]).total_seconds()/60 
                            for t in generated_df['timestamp']]
        
        # =============== 创建更详细的可视化 ===============
        # 1. 速度-时间剖面图 (单独大图)
        plt.figure(figsize=(12, 6))
        plt.plot(original_minutes, original_df['speed_mps'], 'b-', linewidth=2, label='原始轨迹')
        plt.plot(generated_minutes, generated_df['speed_mps'], 'r--', linewidth=2, label='生成轨迹')
        plt.xlabel('时间 (min)')
        plt.ylabel('速度 (m/s)')
        plt.title('速度-时间剖面对比')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speed_time_profile.png'), bbox_inches='tight')
        plt.close()
        
        # 2. 四面板组合图
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 2.1 速度-时间剖面
        axs[0, 0].plot(original_minutes, original_df['speed_mps'], 'b-', linewidth=2, label='原始轨迹')
        axs[0, 0].plot(generated_minutes, generated_df['speed_mps'], 'r--', linewidth=2, label='生成轨迹')
        axs[0, 0].set_xlabel('时间 (min)')
        axs[0, 0].set_ylabel('速度 (m/s)')
        axs[0, 0].set_title('速度-时间剖面')
        axs[0, 0].grid(True)
        axs[0, 0].legend(loc='best')
        
        # 2.2 累积距离-时间
        axs[0, 1].plot(original_minutes, original_distance/1000, 'b-', linewidth=2, label='原始轨迹')
        axs[0, 1].plot(generated_minutes, generated_distance/1000, 'r--', linewidth=2, label='生成轨迹')
        axs[0, 1].set_xlabel('时间 (min)')
        axs[0, 1].set_ylabel('累积距离 (km)')
        axs[0, 1].set_title('累积距离-时间对比')
        axs[0, 1].grid(True)
        axs[0, 1].legend(loc='best')
        
        # 2.3 速度差异直方图
        n, bins, patches = axs[1, 0].hist(speed_diff, bins=20, density=True, alpha=0.7, color='g')
        axs[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        # 添加正态分布拟合曲线
        mu, sigma = np.mean(speed_diff), np.std(speed_diff)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        from scipy import stats as scipy_stats
        axs[1, 0].plot(x, scipy_stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                      label=f'正态分布拟合\n(μ={mu:.2f}, σ={sigma:.2f})')
        axs[1, 0].set_xlabel('速度差异 (m/s)')
        axs[1, 0].set_ylabel('概率密度')
        axs[1, 0].set_title('速度差异分布')
        axs[1, 0].grid(True)
        axs[1, 0].legend(loc='best')
        
        # 2.4 原始速度 vs 速度差异散点图
        axs[1, 1].scatter(original_df['speed_mps'], speed_diff, alpha=0.5, s=50)
        axs[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        # 添加趋势线
        z = np.polyfit(original_df['speed_mps'], speed_diff, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(original_df['speed_mps'].min(), original_df['speed_mps'].max(), 100)
        axs[1, 1].plot(x_trend, p(x_trend), 'r-', linewidth=2, 
                      label=f'趋势线 (y={z[0]:.2f}x{z[1]:+.2f})')
        axs[1, 1].set_xlabel('原始速度 (m/s)')
        axs[1, 1].set_ylabel('速度差异 (m/s)')
        axs[1, 1].set_title('原始速度 vs 速度差异')
        axs[1, 1].grid(True)
        axs[1, 1].legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'validation_plots.png'), bbox_inches='tight')
        plt.close()
        
        # 保存统计信息
        with open(os.path.join(output_dir, 'validation_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=4, ensure_ascii=False)
            
        logger.info(f"验证结果已保存到 {output_dir}")
        
        return statistics
        
    except Exception as e:
        logger.error(f"绘制对比图时出错: {str(e)}", exc_info=True)
        raise

def main():
    """主函数"""
    # 设置路径
    base_dir = Path("/home/yzc/data/Sucess_or_Die/complex_trajectories_generator")
    core_trajectories_dir = base_dir / "data/core_trajectories"
    dem_file = base_dir / "data/terrain/dem.tif"
    landcover_file = base_dir / "data/terrain/landcover.tif"
    
    try:
        # 1. 加载真实轨迹数据
        with open(core_trajectories_dir / "core_trajectories_report.json", 'r') as f:
            trajectories_info = json.load(f)
        
        # 创建轨迹处理器
        processor = TrajectoryProcessor(str(dem_file), str(landcover_file))
        logger.info("创建轨迹处理器")
        
        # 加载训练好的模型
        learner = MotionPatternLearner()
        learner.load_results()
        logger.info("加载训练好的模型")
        
        # 对每条轨迹进行验证
        for traj_info in trajectories_info:
            sequence = traj_info['sequence']
            logger.info(f"\n开始验证轨迹 {sequence}")
            
            # 加载原始轨迹
            traj_file = core_trajectories_dir / f"{sequence}_core.csv"
            original_df = load_trajectory(str(traj_file))
            logger.info(f"加载原始轨迹，共 {len(original_df)} 个点")
            
            # 生成运动特征
            generated_df = generate_motion_features(learner, original_df, processor)
            logger.info("生成运动特征完成")
            
            # 创建轨迹专属的输出目录
            output_dir = base_dir / "data/validation" / sequence
            os.makedirs(output_dir, exist_ok=True)
            
            # 绘制对比图并计算统计指标
            stats = plot_comparison(original_df, generated_df, str(output_dir))
            
            # 输出关键统计信息
            logger.info(f"\n{sequence} 验证结果:")
            logger.info(f"总距离: {stats['总距离 (m)']:.2f} 米")
            logger.info(f"原始轨迹平均速度: {stats['平均速度 (原始) (m/s)']:.2f} m/s")
            logger.info(f"生成轨迹平均速度: {stats['平均速度 (生成) (m/s)']:.2f} m/s")
            logger.info(f"速度差异均值: {stats['速度差异均值 (m/s)']:.2f} m/s")
            logger.info(f"总时长差异: {stats['总时长差异 (s)']:.2f} 秒 ({stats['总时长差异百分比 (%)']:.2f}%)")
        
        logger.info("\n所有轨迹验证完成")
        
    except Exception as e:
        logger.error(f"验证过程出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 