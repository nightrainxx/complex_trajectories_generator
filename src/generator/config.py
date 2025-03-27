"""环境地图生成器配置文件

定义环境地图生成所需的常量和参数。
"""

# 坡度相关参数
MAX_SLOPE_THRESHOLD = 45.0  # 最大可通行坡度（度）
SLOPE_SPEED_FACTOR = 0.02   # 坡度对速度的影响因子
MAX_SPEED = 5.0            # 基础最大速度（米/秒）

# 速度相关参数
TYPICAL_SPEED_FACTOR = 0.8  # 典型速度与最大速度的比例
UP_SLOPE_FACTOR = 0.03     # 上坡减速因子
DOWN_SLOPE_FACTOR = 0.01   # 下坡增速因子
CROSS_SLOPE_FACTOR = 0.02  # 横坡减速因子

# 标准差相关参数
BASE_SPEED_STDDEV_FACTOR = 0.2  # 基础速度标准差因子
SLOPE_STDDEV_FACTOR = 0.5       # 坡度对标准差的影响因子
COMPLEX_TERRAIN_STDDEV_FACTOR = 1.5  # 复杂地形标准差增加因子

# 复杂地形代码
COMPLEX_TERRAIN_CODES = [
    21,  # 山地
    22,  # 丘陵
    23,  # 高原
]

# 土地覆盖速度因子
LANDCOVER_SPEED_FACTORS = {
    11: 1.0,   # 平原
    12: 0.9,   # 草地
    13: 0.8,   # 灌木
    21: 0.7,   # 山地
    22: 0.8,   # 丘陵
    23: 0.9,   # 高原
    31: 0.0,   # 水体
    41: 0.0,   # 冰川
    51: 0.6,   # 建筑区
    61: 0.7,   # 农田
    71: 0.8,   # 林地
    81: 0.0,   # 沼泽
    82: 0.0,   # 盐碱地
}

# 土地覆盖成本因子
LANDCOVER_COST_FACTORS = {
    11: 1.0,   # 平原
    12: 1.2,   # 草地
    13: 1.3,   # 灌木
    21: 1.5,   # 山地
    22: 1.3,   # 丘陵
    23: 1.2,   # 高原
    51: 1.4,   # 建筑区
    61: 1.2,   # 农田
    71: 1.3,   # 林地
}

# 不可通行土地覆盖代码
IMPASSABLE_LANDCOVER_CODES = [
    31,  # 水体
    41,  # 冰川
    81,  # 沼泽
    82,  # 盐碱地
] 