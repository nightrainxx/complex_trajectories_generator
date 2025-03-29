"""
生成坡度数据
"""

import logging
from pathlib import Path

from src.utils.config import config
from src.core.terrain import TerrainAnalyzer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    logger.info("开始生成坡度数据...")
    
    # 创建地形分析器
    analyzer = TerrainAnalyzer()
    
    # 加载DEM数据
    analyzer.load_dem(config.paths.DEM_FILE)
    
    # 计算坡度和坡向
    analyzer.calculate_slope_magnitude()
    analyzer.calculate_slope_aspect()
    
    # 保存结果
    analyzer.save_results()
    
    logger.info("坡度数据生成完成")

if __name__ == '__main__':
    main() 