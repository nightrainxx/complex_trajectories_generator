"""
生成测试用的地形数据
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

from src.utils.config import config

def create_test_dem():
    """创建测试用的DEM数据"""
    # 创建100x100的网格
    rows, cols = 100, 100
    cell_size = 30  # 30米分辨率
    
    # 生成基础地形（使用高斯分布创建丘陵地形）
    x = np.linspace(-5, 5, cols)
    y = np.linspace(-5, 5, rows)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-((X-2)**2 + (Y-2)**2))
    Z2 = np.exp(-((X+2)**2 + (Y+2)**2))
    dem = (Z1 + Z2) * 1000  # 转换为米
    
    # 创建仿射变换
    transform = from_origin(0, 3000, cell_size, cell_size)
    
    # 保存DEM数据
    with rasterio.open(
        config.paths.DEM_FILE,
        'w',
        driver='GTiff',
        height=rows,
        width=cols,
        count=1,
        dtype=dem.dtype,
        crs='+proj=utm +zone=50 +datum=WGS84',
        transform=transform
    ) as dst:
        dst.write(dem, 1)
        
def create_test_landcover():
    """创建测试用的土地覆盖数据"""
    # 创建100x100的网格
    rows, cols = 100, 100
    cell_size = 30  # 30米分辨率
    
    # 生成基础土地覆盖（随机分配地物类型）
    landcover = np.random.choice(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        size=(rows, cols),
        p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
    )
    
    # 创建一些连续的区域（模拟真实地物分布）
    from scipy.ndimage import gaussian_filter
    noise = gaussian_filter(np.random.randn(rows, cols), sigma=3)
    thresholds = np.percentile(noise, [10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
    
    landcover = np.zeros((rows, cols), dtype=np.int32)
    landcover[noise <= thresholds[0]] = 1  # 城市
    landcover[np.logical_and(noise > thresholds[0], noise <= thresholds[1])] = 2  # 道路
    landcover[np.logical_and(noise > thresholds[1], noise <= thresholds[2])] = 3  # 道路
    landcover[np.logical_and(noise > thresholds[2], noise <= thresholds[3])] = 4  # 田地
    landcover[np.logical_and(noise > thresholds[3], noise <= thresholds[4])] = 5  # 田地
    landcover[np.logical_and(noise > thresholds[4], noise <= thresholds[5])] = 6  # 森林
    landcover[np.logical_and(noise > thresholds[5], noise <= thresholds[6])] = 7  # 森林
    landcover[np.logical_and(noise > thresholds[6], noise <= thresholds[7])] = 8  # 山地
    landcover[np.logical_and(noise > thresholds[7], noise <= thresholds[8])] = 9  # 山地
    landcover[np.logical_and(noise > thresholds[8], noise <= thresholds[9])] = 10  # 城市
    landcover[noise > thresholds[9]] = 11  # 水体
    
    # 创建仿射变换
    transform = from_origin(0, 3000, cell_size, cell_size)
    
    # 保存土地覆盖数据
    with rasterio.open(
        config.paths.LANDCOVER_FILE,
        'w',
        driver='GTiff',
        height=rows,
        width=cols,
        count=1,
        dtype=landcover.dtype,
        crs='+proj=utm +zone=50 +datum=WGS84',
        transform=transform
    ) as dst:
        dst.write(landcover, 1)

def main():
    """主函数"""
    print("开始生成测试数据...")
    
    # 创建必要的目录
    config.paths.GIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 生成DEM数据
    print("生成DEM数据...")
    create_test_dem()
    print(f"DEM数据已保存至: {config.paths.DEM_FILE}")
    
    # 生成土地覆盖数据
    print("生成土地覆盖数据...")
    create_test_landcover()
    print(f"土地覆盖数据已保存至: {config.paths.LANDCOVER_FILE}")
    
    print("测试数据生成完成")

if __name__ == '__main__':
    main() 