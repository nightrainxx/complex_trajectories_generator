"""配置文件

包含项目所需的所有配置参数和路径设置。

定义环境地图生成器所需的常量和参数。
"""

import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
INPUT_DIR = DATA_DIR / 'input'
OUTPUT_DIR = DATA_DIR / 'output'
INTERMEDIATE_DIR = OUTPUT_DIR / 'intermediate'

# 确保目录存在
for dir_path in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, INTERMEDIATE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 输入数据路径
GIS_DIR = INPUT_DIR / 'gis'
OORD_DIR = INPUT_DIR / 'oord'

# GIS数据文件
DEM_PATH = GIS_DIR / 'dem_30m_100km.tif'
LANDCOVER_PATH = GIS_DIR / 'landcover_30m_100km.tif'

# 生成的地形属性文件
SLOPE_MAGNITUDE_PATH = INTERMEDIATE_DIR / 'slope_magnitude_30m_100km.tif'
SLOPE_ASPECT_PATH = INTERMEDIATE_DIR / 'slope_aspect_30m_100km.tif'
DZDX_PATH = INTERMEDIATE_DIR / 'dzdx_30m_100km.tif'
DZDY_PATH = INTERMEDIATE_DIR / 'dzdy_30m_100km.tif'

# 环境地图文件
MAX_SPEED_MAP_PATH = INTERMEDIATE_DIR / 'max_speed_map.tif'
TYPICAL_SPEED_MAP_PATH = INTERMEDIATE_DIR / 'typical_speed_map.tif'
SPEED_STDDEV_MAP_PATH = INTERMEDIATE_DIR / 'speed_stddev_map.tif'
COST_MAP_PATH = INTERMEDIATE_DIR / 'cost_map.tif'

# 轨迹生成参数
NUM_TRAJECTORIES_TO_GENERATE = 500  # 要生成的轨迹总数
NUM_END_POINTS = 3                  # 要选择的固定终点数量
NUM_TRAJECTORIES_PER_END = NUM_TRAJECTORIES_TO_GENERATE // NUM_END_POINTS
MIN_START_END_DISTANCE_METERS = 80000  # 起终点最小直线距离（米）

# 地物编码
URBAN_LANDCOVER_CODES = [1, 10]     # 代表城市/建成区的地物编码
IMPASSABLE_LANDCOVER_CODES = [11]   # 代表绝对不可通行的地物编码（如水体）

# 坡度分组
SLOPE_BINS = [0, 5, 10, 15, 20, 25, 30, 35]  # 坡度分组边界（度）

# 坡度相关参数
MAX_SLOPE_THRESHOLD = 45.0  # 最大可通行坡度（度）
SLOPE_SPEED_FACTOR = 0.05   # 坡度对速度的影响因子
BASE_MAX_SPEED = 5.0        # 基础最大速度（m/s）

# 速度相关参数
TYPICAL_SPEED_FACTOR = 0.7  # 典型速度与最大速度的比例
BASE_SPEED_STDDEV_FACTOR = 0.2  # 基础速度标准差因子

# 坡向相关参数
PREFERRED_DIRECTION = 90.0  # 偏好移动方向（度，东向为90度）
ASPECT_INFLUENCE = 0.2      # 坡向对速度的影响程度

# 坡度对标准差的影响
SLOPE_STDDEV_THRESHOLD = 30.0  # 坡度标准差阈值（度）
SLOPE_STDDEV_FACTOR = 0.5      # 坡度对标准差的影响因子

# 地形复杂度相关
COMPLEX_TERRAIN_CODES = [41, 42, 43]  # 复杂地形的土地覆盖代码
COMPLEX_TERRAIN_STDDEV_FACTOR = 1.5   # 复杂地形对标准差的影响因子

# 土地覆盖类型对速度的影响因子
LANDCOVER_SPEED_FACTORS = {
    11: 1.0,  # 城市建成区
    21: 0.9,  # 农田
    31: 0.8,  # 草地
    41: 0.7,  # 灌木林
    42: 0.6,  # 疏林
    43: 0.5,  # 密林
    51: 0.4,  # 湿地
    61: 0.3,  # 裸地
    71: 0.2   # 沙地
}

# 土地覆盖类型对成本的额外影响因子
LANDCOVER_COST_FACTORS = {
    11: 1.0,  # 城市建成区
    21: 1.2,  # 农田
    31: 1.3,  # 草地
    41: 1.5,  # 灌木林
    42: 1.7,  # 疏林
    43: 2.0,  # 密林
    51: 2.5,  # 湿地
    61: 1.8,  # 裸地
    71: 2.2   # 沙地
}

# 不可通行的土地覆盖类型代码
IMPASSABLE_LANDCOVER_CODES = [81, 82, 83]  # 水体、冰川、建筑物等

# 运动约束参数
MOTION_CONSTRAINTS = {
    'max_acceleration': 2.0,     # 最大加速度 (m/s^2)
    'max_deceleration': 4.0,     # 最大减速度 (m/s^2)
    'max_turn_rate': 45.0,       # 最大转向率 (度/秒)
    'min_speed': 0.0,           # 最小速度 (m/s)
    'time_step': 1.0,           # 模拟时间步长 (秒)
    'max_iterations': 10000,     # 最大迭代次数
    'position_threshold': 0.1    # 位置判断阈值 (m)
}

# 地形约束参数
TERRAIN_CONSTRAINTS = {
    'max_uphill_slope': 30.0,    # 最大上坡坡度 (度)
    'max_downhill_slope': 35.0,  # 最大下坡坡度 (度)
    'max_cross_slope': 25.0,     # 最大横坡坡度 (度)
    'k_uphill': 0.1,            # 上坡减速系数
    'k_downhill': 0.05,         # 下坡加速系数
    'k_cross': 0.2,             # 横坡减速系数
    'min_speed_steep_slope': 0.5 # 陡坡最小速度 (m/s)
}

# 基础运动参数
BASE_SPEED = 10.0               # 基准速度 (m/s)
MAX_SLOPE = 35.0               # 最大可通行坡度 (度)

# 评估参数
EVALUATION_METRICS = {
    'speed_distribution': True,
    'acceleration_distribution': True,
    'turn_rate_distribution': True,
    'slope_speed_relationship': True
} 