"""
TerrainLoader模块的单元测试
"""

import pytest
import numpy as np
import rasterio
from pathlib import Path

from src.data_processing import TerrainLoader

def test_terrain_loader_init():
    """测试TerrainLoader的初始化"""
    loader = TerrainLoader()
    assert loader.dem_data is None
    assert loader.landcover_data is None
    assert loader.transform is None
    assert loader.crs is None
    assert loader.resolution is None
    assert loader.bounds is None

def test_load_dem(test_dem_data, tmp_path):
    """测试DEM数据加载功能"""
    # 准备测试数据
    dem_path = tmp_path / "test_dem.tif"
    with rasterio.open(
        dem_path,
        'w',
        driver='GTiff',
        height=test_dem_data.shape[0],
        width=test_dem_data.shape[1],
        count=1,
        dtype=test_dem_data.dtype,
        crs='+proj=latlong',
        transform=rasterio.transform.from_bounds(
            116.0, 39.0, 116.1, 39.1,
            test_dem_data.shape[1], test_dem_data.shape[0]
        )
    ) as dst:
        dst.write(test_dem_data, 1)
    
    # 测试加载功能
    loader = TerrainLoader()
    dem_array = loader.load_dem(dem_path)
    
    assert dem_array is not None
    assert dem_array.shape == test_dem_data.shape
    assert np.allclose(dem_array, test_dem_data)
    assert loader.transform is not None
    assert loader.crs is not None
    assert loader.resolution is not None
    assert loader.bounds is not None

def test_load_landcover(test_dem_data, test_landcover_data, tmp_path):
    """测试土地覆盖数据加载功能"""
    # 准备DEM测试数据
    dem_path = tmp_path / "test_dem.tif"
    landcover_path = tmp_path / "test_landcover.tif"
    transform = rasterio.transform.from_bounds(
        116.0, 39.0, 116.1, 39.1,
        test_dem_data.shape[1], test_dem_data.shape[0]
    )
    
    # 保存DEM数据
    with rasterio.open(
        dem_path,
        'w',
        driver='GTiff',
        height=test_dem_data.shape[0],
        width=test_dem_data.shape[1],
        count=1,
        dtype=test_dem_data.dtype,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(test_dem_data, 1)
    
    # 保存土地覆盖数据
    with rasterio.open(
        landcover_path,
        'w',
        driver='GTiff',
        height=test_landcover_data.shape[0],
        width=test_landcover_data.shape[1],
        count=1,
        dtype=test_landcover_data.dtype,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(test_landcover_data, 1)
    
    # 测试加载功能
    loader = TerrainLoader()
    loader.load_dem(dem_path)
    landcover_array = loader.load_landcover(landcover_path)
    
    assert landcover_array is not None
    assert landcover_array.shape == test_landcover_data.shape
    assert np.array_equal(landcover_array, test_landcover_data)

def test_coordinate_conversion(tmp_path, test_dem_data):
    """测试坐标转换功能"""
    # 准备测试数据
    dem_path = tmp_path / "test_dem.tif"
    transform = rasterio.transform.from_bounds(
        116.0, 39.0, 116.1, 39.1,
        test_dem_data.shape[1], test_dem_data.shape[0]
    )
    
    with rasterio.open(
        dem_path,
        'w',
        driver='GTiff',
        height=test_dem_data.shape[0],
        width=test_dem_data.shape[1],
        count=1,
        dtype=test_dem_data.dtype,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(test_dem_data, 1)
    
    # 测试坐标转换
    loader = TerrainLoader()
    loader.load_dem(dem_path)
    
    # 测试经纬度到像素坐标的转换
    row, col = loader.get_pixel_coords(116.05, 39.05)
    assert isinstance(row, int)
    assert isinstance(col, int)
    assert 0 <= row < test_dem_data.shape[0]
    assert 0 <= col < test_dem_data.shape[1]
    
    # 测试像素坐标到经纬度的转换
    lon, lat = loader.get_geo_coords(row, col)
    assert isinstance(lon, float)
    assert isinstance(lat, float)
    assert 116.0 <= lon <= 116.1
    assert 39.0 <= lat <= 39.1

def test_get_elevation_and_landcover(tmp_path, test_dem_data, test_landcover_data):
    """测试获取高程和土地覆盖类型功能"""
    # 准备测试数据
    dem_path = tmp_path / "test_dem.tif"
    landcover_path = tmp_path / "test_landcover.tif"
    transform = rasterio.transform.from_bounds(
        116.0, 39.0, 116.1, 39.1,
        test_dem_data.shape[1], test_dem_data.shape[0]
    )
    
    # 保存DEM数据
    with rasterio.open(
        dem_path,
        'w',
        driver='GTiff',
        height=test_dem_data.shape[0],
        width=test_dem_data.shape[1],
        count=1,
        dtype=test_dem_data.dtype,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(test_dem_data, 1)
    
    # 保存土地覆盖数据
    with rasterio.open(
        landcover_path,
        'w',
        driver='GTiff',
        height=test_landcover_data.shape[0],
        width=test_landcover_data.shape[1],
        count=1,
        dtype=test_landcover_data.dtype,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(test_landcover_data, 1)
    
    # 测试获取高程和土地覆盖类型
    loader = TerrainLoader()
    loader.load_dem(dem_path)
    loader.load_landcover(landcover_path)
    
    # 测试中心点的值
    center_lon = 116.05
    center_lat = 39.05
    
    elevation = loader.get_elevation(center_lon, center_lat)
    assert isinstance(elevation, float)
    assert 100 <= elevation <= 140
    
    landcover = loader.get_landcover(center_lon, center_lat)
    assert isinstance(landcover, int)
    assert 1 <= landcover <= 5

def test_error_handling():
    """测试错误处理"""
    loader = TerrainLoader()
    
    # 测试在未加载数据时的错误处理
    with pytest.raises(ValueError):
        loader.get_pixel_coords(116.0, 39.0)
    
    with pytest.raises(ValueError):
        loader.get_geo_coords(0, 0)
    
    with pytest.raises(ValueError):
        loader.get_elevation(116.0, 39.0)
    
    with pytest.raises(ValueError):
        loader.get_landcover(116.0, 39.0) 