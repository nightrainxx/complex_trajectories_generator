"""统一配置文件

本文件整合了项目中所有配置参数，包括路径设置、地形参数、运动约束、环境映射参数等。
所有模块都应该从这个文件导入配置，而不是使用局部配置文件。

结构:
1. 路径配置 - 所有输入输出文件路径
2. 地形配置 - 坡度、地物编码等地形相关参数
3. 运动配置 - 速度、加速度、转向等运动约束参数
4. 环境映射配置 - 用于生成环境地图的参数
5. 轨迹生成配置 - 轨迹生成相关参数
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

# =============== 1. 路径配置 ===============

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
GIS_DIR.mkdir(exist_ok=True)
OORD_DIR.mkdir(exist_ok=True)

# GIS数据文件
DEM_PATH = GIS_DIR / 'dem_30m_100km.tif'
LANDCOVER_PATH = GIS_DIR / 'landcover_30m_100km.tif'

# 生成的地形属性文件
SLOPE_MAGNITUDE_PATH = INTERMEDIATE_DIR / 'slope_magnitude_30m_100km.tif'
SLOPE_ASPECT_PATH = INTERMEDIATE_DIR / 'slope_aspect_30m_100km.tif'

# 环境地图文件
MAX_SPEED_MAP_PATH = INTERMEDIATE_DIR / 'max_speed_map.tif'
TYPICAL_SPEED_MAP_PATH = INTERMEDIATE_DIR / 'typical_speed_map.tif'
SPEED_STDDEV_MAP_PATH = INTERMEDIATE_DIR / 'speed_stddev_map.tif'
COST_MAP_PATH = INTERMEDIATE_DIR / 'cost_map.tif'

# 学习结果文件
LEARNED_PATTERNS_FILE = INTERMEDIATE_DIR / 'learned_patterns.json'
ENVIRONMENT_GROUPS_FILE = INTERMEDIATE_DIR / 'environment_groups.json'

# 输出目录
TRAJECTORY_DIR = OUTPUT_DIR / 'trajectory_generation'
EVALUATION_DIR = OUTPUT_DIR / 'evaluation'
TRAJECTORY_DIR.mkdir(exist_ok=True)
EVALUATION_DIR.mkdir(exist_ok=True)

# =============== 2. 地形配置 ===============

# 坡度分级（度）
SLOPE_BINS = [0, 5, 10, 15, 20, 25, 30, 35]

# 坡度限制
MAX_SLOPE_THRESHOLD = 35.0  # 最大可通行坡度（度）

# 土地覆盖类型代码
LANDCOVER_CODES = {
    0: '未知',
    1: '平地',
    2: '山地',
    3: '丘陵',
    4: '水域',
    5: '建筑区',
    11: '城市建成区',
    21: '农田',
    31: '草地',
    41: '灌木林',
    42: '疏林',
    43: '密林',
    51: '湿地',
    61: '裸地',
    71: '沙地',
    81: '水体',
    82: '冰川',
    83: '建筑物'
}

# 复杂地形代码
COMPLEX_TERRAIN_CODES = [2, 3, 41, 42, 43]  # 山地、丘陵、灌木林、疏林、密林

# 不可通行的土地覆盖类型
IMPASSABLE_LANDCOVER_CODES = [4, 5, 81, 82, 83]  # 水域、建筑区、水体、冰川、建筑物

# =============== 3. 运动配置 ===============

# 速度限制
DEFAULT_SPEED = 5.0   # 默认速度（米/秒）
MAX_SPEED = 10.0      # 最大速度（米/秒）
MIN_SPEED = 1.0       # 最小速度（米/秒）
SPEED_STDDEV = 1.0    # 速度标准差（米/秒）

# 加速度约束
MAX_ACCELERATION = 1.0  # 最大加速度（米/秒²）
MAX_DECELERATION = 2.0  # 最大减速度（米/秒²）

# 转向约束
MAX_TURN_RATE = 30.0  # 最大转向率（度/秒）

# 地形运动约束
MAX_UPHILL_SLOPE = 30.0    # 最大上坡坡度（度）
MAX_DOWNHILL_SLOPE = 35.0  # 最大下坡坡度（度）
MAX_CROSS_SLOPE = 25.0     # 最大横坡坡度（度）
MIN_SPEED_STEEP_SLOPE = 0.5 # 陡坡最小速度（米/秒）

# 坡度影响因子
SLOPE_SPEED_FACTOR = 0.05   # 坡度对速度的影响因子
UP_SLOPE_FACTOR = 0.1      # 上坡减速系数
DOWN_SLOPE_FACTOR = 0.05   # 下坡加速系数
CROSS_SLOPE_FACTOR = 0.2   # 横坡减速系数

# 模拟参数
TIME_STEP = 1.0           # 模拟时间步长（秒）
MAX_ITERATIONS = 10000     # 最大迭代次数
POSITION_THRESHOLD = 0.1   # 位置判断阈值（米）

# =============== 4. 环境映射配置 ===============

# 速度映射参数
TYPICAL_SPEED_FACTOR = 0.8  # 典型速度与最大速度的比例

# 标准差映射参数
BASE_SPEED_STDDEV_FACTOR = 0.2  # 基础速度标准差因子
SLOPE_STDDEV_FACTOR = 0.5       # 坡度对标准差的影响因子
COMPLEX_TERRAIN_STDDEV_FACTOR = 1.5  # 复杂地形标准差增加因子

# 土地覆盖类型对速度的影响因子
LANDCOVER_SPEED_FACTORS = {
    0: 1.0,   # 未知
    1: 1.0,   # 平地
    2: 0.7,   # 山地
    3: 0.8,   # 丘陵
    4: 0.0,   # 水域
    5: 0.0,   # 建筑区
    11: 1.0,  # 城市建成区
    21: 0.9,  # 农田
    31: 0.8,  # 草地
    41: 0.7,  # 灌木林
    42: 0.6,  # 疏林
    43: 0.5,  # 密林
    51: 0.4,  # 湿地
    61: 0.7,  # 裸地
    71: 0.5,  # 沙地
    81: 0.0,  # 水体
    82: 0.0,  # 冰川
    83: 0.0   # 建筑物
}

# 土地覆盖类型对成本的影响因子
LANDCOVER_COST_FACTORS = {
    0: 1.2,   # 未知
    1: 1.0,   # 平地
    2: 1.5,   # 山地
    3: 1.3,   # 丘陵
    4: 999.0, # 水域
    5: 999.0, # 建筑区
    11: 1.0,  # 城市建成区
    21: 1.2,  # 农田
    31: 1.3,  # 草地
    41: 1.5,  # 灌木林
    42: 1.7,  # 疏林
    43: 2.0,  # 密林
    51: 2.5,  # 湿地
    61: 1.8,  # 裸地
    71: 2.2,  # 沙地
    81: 999.0,# 水体
    82: 999.0,# 冰川
    83: 999.0 # 建筑物
}

# =============== 5. 轨迹生成配置 ===============

# 轨迹生成参数
NUM_TRAJECTORIES = 500      # 要生成的轨迹总数
NUM_END_POINTS = 3         # 要选择的固定终点数量
MIN_START_END_DISTANCE = 80000.0  # 起终点最小直线距离（米）
MAX_START_END_DISTANCE = 100000.0  # 起终点最大直线距离（米）
MIN_SAMPLES_PER_GROUP = 20 # 每个环境组的最小样本数

# 环境组格式
GROUP_LABEL_FORMAT = "LC{lc}_S{slope}"

# 起点选择参数
MIN_START_POINTS_SPACING = 250.0  # 起点之间的最小间距（米）
MAX_SEARCH_RADIUS = 5000.0       # 最大搜索半径（米）
MAX_SEARCH_ATTEMPTS = 1000       # 最大搜索尝试次数

# 评估指标
EVALUATION_METRICS = {
    'speed_distribution': True,
    'acceleration_distribution': True,
    'turn_rate_distribution': True,
    'slope_speed_relationship': True
}

# 将主要配置按类别组织到字典中，方便其他模块导入
config = {
    'paths': {
        'BASE_DIR': BASE_DIR,
        'DATA_DIR': DATA_DIR,
        'INPUT_DIR': INPUT_DIR,
        'OUTPUT_DIR': OUTPUT_DIR,
        'INTERMEDIATE_DIR': INTERMEDIATE_DIR,
        'GIS_DIR': GIS_DIR,
        'OORD_DIR': OORD_DIR,
        'DEM_PATH': DEM_PATH,
        'LANDCOVER_PATH': LANDCOVER_PATH,
        'SLOPE_MAGNITUDE_PATH': SLOPE_MAGNITUDE_PATH,
        'SLOPE_ASPECT_PATH': SLOPE_ASPECT_PATH,
        'MAX_SPEED_MAP_PATH': MAX_SPEED_MAP_PATH,
        'TYPICAL_SPEED_MAP_PATH': TYPICAL_SPEED_MAP_PATH,
        'SPEED_STDDEV_MAP_PATH': SPEED_STDDEV_MAP_PATH,
        'COST_MAP_PATH': COST_MAP_PATH,
        'LEARNED_PATTERNS_FILE': LEARNED_PATTERNS_FILE,
        'ENVIRONMENT_GROUPS_FILE': ENVIRONMENT_GROUPS_FILE,
        'TRAJECTORY_DIR': TRAJECTORY_DIR,
        'EVALUATION_DIR': EVALUATION_DIR
    },
    'terrain': {
        'SLOPE_BINS': SLOPE_BINS,
        'MAX_SLOPE_THRESHOLD': MAX_SLOPE_THRESHOLD,
        'LANDCOVER_CODES': LANDCOVER_CODES,
        'COMPLEX_TERRAIN_CODES': COMPLEX_TERRAIN_CODES,
        'IMPASSABLE_LANDCOVER_CODES': IMPASSABLE_LANDCOVER_CODES
    },
    'motion': {
        'DEFAULT_SPEED': DEFAULT_SPEED,
        'MAX_SPEED': MAX_SPEED,
        'MIN_SPEED': MIN_SPEED,
        'SPEED_STDDEV': SPEED_STDDEV,
        'MAX_ACCELERATION': MAX_ACCELERATION,
        'MAX_DECELERATION': MAX_DECELERATION,
        'MAX_TURN_RATE': MAX_TURN_RATE,
        'MAX_UPHILL_SLOPE': MAX_UPHILL_SLOPE,
        'MAX_DOWNHILL_SLOPE': MAX_DOWNHILL_SLOPE,
        'MAX_CROSS_SLOPE': MAX_CROSS_SLOPE,
        'MIN_SPEED_STEEP_SLOPE': MIN_SPEED_STEEP_SLOPE,
        'SLOPE_SPEED_FACTOR': SLOPE_SPEED_FACTOR,
        'UP_SLOPE_FACTOR': UP_SLOPE_FACTOR,
        'DOWN_SLOPE_FACTOR': DOWN_SLOPE_FACTOR,
        'CROSS_SLOPE_FACTOR': CROSS_SLOPE_FACTOR,
        'TIME_STEP': TIME_STEP,
        'MAX_ITERATIONS': MAX_ITERATIONS,
        'POSITION_THRESHOLD': POSITION_THRESHOLD
    },
    'mapping': {
        'TYPICAL_SPEED_FACTOR': TYPICAL_SPEED_FACTOR,
        'BASE_SPEED_STDDEV_FACTOR': BASE_SPEED_STDDEV_FACTOR,
        'SLOPE_STDDEV_FACTOR': SLOPE_STDDEV_FACTOR,
        'COMPLEX_TERRAIN_STDDEV_FACTOR': COMPLEX_TERRAIN_STDDEV_FACTOR,
        'LANDCOVER_SPEED_FACTORS': LANDCOVER_SPEED_FACTORS,
        'LANDCOVER_COST_FACTORS': LANDCOVER_COST_FACTORS
    },
    'generation': {
        'NUM_TRAJECTORIES': NUM_TRAJECTORIES,
        'NUM_END_POINTS': NUM_END_POINTS,
        'MIN_START_END_DISTANCE': MIN_START_END_DISTANCE,
        'MAX_START_END_DISTANCE': MAX_START_END_DISTANCE,
        'MIN_SAMPLES_PER_GROUP': MIN_SAMPLES_PER_GROUP,
        'GROUP_LABEL_FORMAT': GROUP_LABEL_FORMAT,
        'MIN_START_POINTS_SPACING': MIN_START_POINTS_SPACING,
        'MAX_SEARCH_RADIUS': MAX_SEARCH_RADIUS,
        'MAX_SEARCH_ATTEMPTS': MAX_SEARCH_ATTEMPTS,
        'EVALUATION_METRICS': EVALUATION_METRICS
    }
} 