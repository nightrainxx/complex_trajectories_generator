"""
统一配置管理模块
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

@dataclass
class PathConfig:
    """路径配置"""
    # 输入数据路径
    INPUT_DIR: Path = Path("data/input")
    GIS_DIR: Path = INPUT_DIR / "gis"
    OORD_DIR: Path = INPUT_DIR / "oord"
    
    # GIS数据文件
    DEM_FILE: Path = GIS_DIR / "dem_30m_100km.tif"
    LANDCOVER_FILE: Path = GIS_DIR / "landcover_30m_100km.tif"
    
    # 输出数据路径
    OUTPUT_DIR: Path = Path("data/output")
    INTERMEDIATE_DIR: Path = OUTPUT_DIR / "intermediate"
    TRAJECTORY_DIR: Path = OUTPUT_DIR / "trajectory_generation"
    EVALUATION_DIR: Path = OUTPUT_DIR / "evaluation"
    
    # 中间文件
    SLOPE_FILE: Path = INTERMEDIATE_DIR / "slope_magnitude_30m_100km.tif"
    ASPECT_FILE: Path = INTERMEDIATE_DIR / "slope_aspect_30m_100km.tif"
    LEARNED_PATTERNS_FILE: Path = INTERMEDIATE_DIR / "learned_patterns.pkl"
    PROCESSED_OORD_FILE: Path = INTERMEDIATE_DIR / "processed_oord_data.csv"

@dataclass
class TerrainConfig:
    """地形配置"""
    # 坡度分级（度）
    SLOPE_BINS: List[float] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    
    # 坡度限制（度）
    MAX_SLOPE: float = 30.0
    
    # 土地覆盖类型代码
    LANDCOVER_CODES: Dict[int, str] = field(default_factory=lambda: {
        0: '未知',
        1: '平地',
        2: '山地',
        3: '丘陵',
        4: '水域',
        5: '建筑区'
    })

@dataclass
class MotionConfig:
    """运动配置"""
    # 速度相关
    DEFAULT_SPEED: float = 5.0   # 默认速度（米/秒）
    MAX_SPEED: float = 10.0      # 最大速度（米/秒）
    MIN_SPEED: float = 2.0       # 最小速度（米/秒）
    SPEED_STDDEV: float = 1.0    # 速度标准差（米/秒）
    
    # 加速度相关
    MAX_ACCELERATION: float = 1.0  # 最大加速度（米/秒²）
    MAX_DECELERATION: float = 2.0  # 最大减速度（米/秒²）
    
    # 转向相关
    MAX_TURN_RATE: float = 30.0  # 最大转向率（度/秒）
    MAX_SLOPE_DEGREES: float = 30.0  # 最大可通行坡度（度）
    
    # 地形影响因子
    SLOPE_SPEED_FACTOR: float = 0.1  # 坡度对速度的影响因子
    CROSS_SLOPE_FACTOR: float = 0.2  # 横向坡度对速度的影响因子
    
    # 路径跟随参数
    WAYPOINT_THRESHOLD: float = 5.0  # 到达路径点的距离阈值（米）

@dataclass
class GenerationConfig:
    """轨迹生成配置"""
    # 生成数量
    NUM_TRAJECTORIES: int = 500
    NUM_END_POINTS: int = 3
    
    # 距离约束（米）
    MIN_START_END_DISTANCE: float = 80000.0
    MAX_START_END_DISTANCE: float = 100000.0
    
    # 环境组标签格式
    GROUP_LABEL_FORMAT: str = "LC{lc}_S{slope}"

@dataclass
class Config:
    """全局配置"""
    # 版本信息
    VERSION: str = "1.2"
    DATE: str = "2024-03-27"
    
    # 子配置
    paths: PathConfig = PathConfig()
    terrain: TerrainConfig = TerrainConfig()
    motion: MotionConfig = MotionConfig()
    generation: GenerationConfig = GenerationConfig()
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要的目录
        for path in [
            self.paths.INPUT_DIR,
            self.paths.GIS_DIR,
            self.paths.OORD_DIR,
            self.paths.OUTPUT_DIR,
            self.paths.INTERMEDIATE_DIR,
            self.paths.TRAJECTORY_DIR,
            self.paths.EVALUATION_DIR
        ]:
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """从环境变量加载配置"""
        config = cls()
        
        # 示例：从环境变量覆盖配置
        if 'MAX_SPEED' in os.environ:
            config.motion.MAX_SPEED = float(os.environ['MAX_SPEED'])
        
        return config

# 创建全局配置实例
config = Config()

# 配置文件
config = {
    'terrain': {
        # 坡度分级（度）
        'SLOPE_BINS': [0, 5, 10, 15, 20, 25, 30],
        
        # 坡度限制（度）
        'MAX_SLOPE': 30.0,
        
        # 土地覆盖类型代码
        'LANDCOVER_CODES': {
            0: '未知',
            1: '平地',
            2: '山地',
            3: '丘陵',
            4: '水域',
            5: '建筑区'
        },
        
        # 不可通行的土地覆盖类型
        'IMPASSABLE_LANDCOVER_CODES': [4, 5],
        
        # 土地覆盖类型对速度的影响因子
        'SPEED_FACTORS': {
            0: 1.0,  # 未知
            1: 1.0,  # 平地
            2: 0.7,  # 山地
            3: 0.8,  # 丘陵
            4: 0.0,  # 水域
            5: 0.0   # 建筑区
        }
    },
    
    'motion': {
        # 速度相关
        'DEFAULT_SPEED': 5.0,   # 默认速度（米/秒）
        'MAX_SPEED': 10.0,      # 最大速度（米/秒）
        'MIN_SPEED': 2.0,       # 最小速度（米/秒）
        'SPEED_STDDEV': 1.0,    # 速度标准差（米/秒）
        
        # 加速度相关
        'MAX_ACCELERATION': 1.0,  # 最大加速度（米/秒²）
        'MAX_DECELERATION': 2.0,  # 最大减速度（米/秒²）
        
        # 转向相关
        'MAX_TURN_RATE': 30.0,  # 最大转向率（度/秒）
        'MAX_SLOPE_DEGREES': 30.0,  # 最大可通行坡度（度）
        
        # 地形影响因子
        'SLOPE_SPEED_FACTOR': 0.1,  # 坡度对速度的影响因子
        'CROSS_SLOPE_FACTOR': 0.2,  # 横向坡度对速度的影响因子
        
        # 路径跟随参数
        'WAYPOINT_THRESHOLD': 5.0,  # 到达路径点的距离阈值（米）
    }
}

"""
运动模式学习配置文件
"""
from typing import List, Dict, Tuple

# 窗口化参数
LEARNING_WINDOW_DURATION_SECONDS = 3.0  # 窗口时长
LEARNING_WINDOW_SLIDE_SECONDS = 1.0     # 窗口滑动步长

# 有效坡度离散化参数
EFFECTIVE_SLOPE_ALONG_BINS: List[float] = [-90, -15, -5, 5, 15, 90]
EFFECTIVE_SLOPE_ALONG_LABELS: List[str] = ['SteepDown', 'ModDown', 'Flat', 'ModUp', 'SteepUp']

EFFECTIVE_CROSS_SLOPE_BINS: List[float] = [0, 5, 15, 90]
EFFECTIVE_CROSS_SLOPE_LABELS: List[str] = ['Low', 'Medium', 'High']

# 统计分析参数
MIN_SAMPLES_PER_GROUP = 2  # 每个环境组的最小样本数

# 地物类型映射
LANDCOVER_MAPPING: Dict[int, str] = {
    0: 'Unknown',
    1: 'Urban',
    2: 'Agriculture',
    3: 'Forest',
    4: 'Grassland',
    5: 'Water',
    6: 'Barren'
}

# 速度相关参数
DEFAULT_SPEED_MPS = 10.0  # 默认速度，用于处理数据缺失情况
MAX_SPEED_MPS = 20.0     # 最大允许速度
MIN_SPEED_MPS = 2.0      # 最小允许速度

# 文件路径
LEARNED_PATTERNS_PATH = 'data/learned/motion_patterns.json'
WINDOWED_FEATURES_PATH = 'data/learned/windowed_features.csv'
ANALYSIS_REPORT_PATH = 'data/learned/analysis_report.json' 