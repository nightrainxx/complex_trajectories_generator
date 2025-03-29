import numpy as np
import rasterio
from rasterio.transform import from_origin

# 设置地图范围和分辨率
lon_min = 116.0
lat_min = 39.0
resolution_deg = 0.0001  # 约10米
size = 100

# 创建DEM数据
dem = np.zeros((size, size), dtype=np.float32)
for i in range(size):
    for j in range(size):
        dem[i,j] = 100 + i*2 + j  # 简单的坡度

# 创建土地覆盖数据
landcover = np.ones((size, size), dtype=np.int32)
landcover[40:60, 40:60] = 2  # 中间区域为山地

# 设置地理变换矩阵
transform = from_origin(lon_min, lat_min + size * resolution_deg, resolution_deg, resolution_deg)

# 保存DEM
with rasterio.open('data/terrain/dem.tif', 'w',
                  driver='GTiff',
                  height=size,
                  width=size,
                  count=1,
                  dtype=np.float32,
                  crs='EPSG:4326',
                  transform=transform) as dst:
    dst.write(dem, 1)

# 保存土地覆盖
with rasterio.open('data/terrain/landcover.tif', 'w',
                  driver='GTiff',
                  height=size,
                  width=size,
                  count=1,
                  dtype=np.int32,
                  crs='EPSG:4326',
                  transform=transform) as dst:
    dst.write(landcover, 1) 