"""
项目配置文件
包含所有重要的常量、路径和参数设置
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CORE_TRAJECTORIES_DIR = DATA_DIR / "core_trajectories"

# GIS数据路径
DEM_DIR = RAW_DATA_DIR / "dem"
LANDCOVER_DIR = RAW_DATA_DIR / "landcover"

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# 确保必要的目录存在
for dir_path in [PROCESSED_DATA_DIR, OUTPUT_DIR, FIGURES_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 分析参数
SLOPE_BINS = [-float('inf'), 5, 10, 15, 20, 25, 30, float('inf')]  # 坡度分级边界值
SLOPE_LABELS = [f"S{i}" for i in range(len(SLOPE_BINS)-1)]  # 坡度等级标签

# 模拟参数
SIMULATION_DT = 1.0  # 模拟时间步长（秒）
MAX_ACCELERATION = 2.0  # 最大加速度 (m/s^2)
MAX_DECELERATION = 4.0  # 最大减速度 (m/s^2)
MAX_TURN_RATE = 45.0  # 最大转向率 (度/秒)

# 轨迹生成约束
TARGET_LENGTH_RANGE = (80_000, 120_000)  # 目标轨迹长度范围（米）
DEFAULT_TARGET_SPEED = 30.0  # 默认目标平均速度（km/h）

# 随机种子（用于复现结果）
RANDOM_SEED = 42

# 轨迹数据列
TRAJECTORY_COLUMNS = [
    'timestamp',      # 时间戳
    'longitude',      # 经度
    'latitude',       # 纬度
    'elevation',      # 海拔高度（米）
    'speed',         # 速度（米/秒）
    'heading',       # 航向角（度）
    'turn_rate',     # 转向率（度/秒）
    'acceleration'   # 加速度（米/秒²）
] 