"""起点选择器单元测试"""

import pytest
import numpy as np
import rasterio
from pathlib import Path

from src.generator.point_selector import PointSelector
from config import *

@pytest.fixture
def test_data():
    """创建测试用数据"""
    # 创建3000x3000的测试数据，以适应实际终点坐标
    landcover = np.full((3000, 3000), 31, dtype=np.int32)  # 默认为草地
    
    # 添加一些建筑用地和农田
    landcover[400:500, 100:200] = 11  # 建筑用地（靠近第一个终点）
    landcover[1000:1100, 2600:2700] = 21  # 农田（靠近第二个终点）
    landcover[1700:1800, 1400:1500] = 11  # 建筑用地（靠近第三个终点）
    landcover[2500:2600, 2100:2200] = 21  # 农田（靠近第四个终点）
    
    # 添加一些不可通行区域
    landcover[500:600, 500:600] = 81  # 水体
    landcover[1500:1600, 1500:1600] = 82  # 冰川
    
    # 创建坡度数据
    slope = np.zeros((3000, 3000), dtype=np.float32)
    slope[700:800, 700:800] = 50.0  # 添加一些陡峭区域
    slope[2000:2100, 2000:2100] = 45.0  # 添加更多陡峭区域
    
    return {
        'landcover': landcover,
        'slope': slope
    }

@pytest.fixture
def test_files(tmp_path, test_data):
    """创建测试用文件"""
    # 创建测试用GeoTIFF文件
    transform = rasterio.transform.from_origin(-6.0, 58.0, 0.001, 0.001)  # 调整原点和分辨率
    meta = {
        'driver': 'GTiff',
        'height': 3000,
        'width': 3000,
        'count': 1,
        'dtype': np.float32,
        'crs': '+proj=latlong',
        'transform': transform
    }
    
    # 保存土地覆盖文件
    landcover_path = tmp_path / "landcover.tif"
    meta['dtype'] = np.int32
    with rasterio.open(landcover_path, 'w', **meta) as dst:
        dst.write(test_data['landcover'], 1)
    
    # 保存坡度文件
    slope_path = tmp_path / "slope.tif"
    meta['dtype'] = np.float32
    with rasterio.open(slope_path, 'w', **meta) as dst:
        dst.write(test_data['slope'], 1)
    
    return {
        'landcover_path': landcover_path,
        'slope_path': slope_path
    }

@pytest.fixture
def fixed_end_points():
    """固定的终点坐标"""
    return [
        {'pixel': (481, 113), 'coord': (-5.237116, 57.263388)},
        {'pixel': (1095, 2682), 'coord': (-3.970768, 57.097960)},
        {'pixel': (1812, 1427), 'coord': (-4.589400, 56.904782)},
        {'pixel': (2577, 2149), 'coord': (-4.233502, 56.698670)}
    ]

class TestPointSelector:
    """起点选择器测试类"""
    
    def test_init(self, test_files):
        """测试初始化"""
        selector = PointSelector(
            test_files['landcover_path'],
            test_files['slope_path']
        )
        assert selector.landcover_data is not None
        assert selector.slope_data is not None
        assert selector.transform is not None
    
    def test_is_point_accessible(self, test_files):
        """测试点的可通行性判断"""
        selector = PointSelector(
            test_files['landcover_path'],
            test_files['slope_path']
        )
        
        # 测试正常区域
        assert selector.is_point_accessible(450, 150)  # 建筑用地
        assert selector.is_point_accessible(1050, 2650)  # 农田
        assert selector.is_point_accessible(100, 100)  # 草地
        
        # 测试不可通行区域
        assert not selector.is_point_accessible(550, 550)  # 水体
        assert not selector.is_point_accessible(750, 750)  # 陡峭区域
        assert not selector.is_point_accessible(1550, 1550)  # 冰川
    
    def test_calculate_distance(self, test_files):
        """测试距离计算"""
        selector = PointSelector(
            test_files['landcover_path'],
            test_files['slope_path']
        )
        
        # 计算两点间的距离（像素坐标）
        distance = selector.calculate_distance((0, 0), (3, 4))
        assert np.isclose(distance, 5.0)
        
        # 计算实际地理距离（考虑像素大小和纬度）
        point1 = (1000, 1000)  # 选择一个中心点
        point2 = (1003, 1004)  # 在中心点附近选择第二个点
        
        # 计算期望的地理距离
        lon1, lat1 = selector.pixel_to_geo(point1)
        lon2, lat2 = selector.pixel_to_geo(point2)
        dx = (lon2 - lon1) * selector.meters_per_degree * np.cos(np.radians((lat1 + lat2) / 2))
        dy = (lat2 - lat1) * selector.meters_per_degree
        expected_distance = np.sqrt(dx * dx + dy * dy)
        
        # 计算实际距离并比较
        geo_distance = selector.calculate_geo_distance(point1, point2)
        assert np.isclose(geo_distance, expected_distance)
    
    def test_select_start_points(self, test_files, fixed_end_points):
        """测试起点选择"""
        selector = PointSelector(
            test_files['landcover_path'],
            test_files['slope_path']
        )
        
        # 为每个终点选择起点
        for end_point in fixed_end_points:
            start_points = selector.select_start_points(
                end_point['pixel'],
                num_points=5,
                min_distance=MIN_START_END_DISTANCE_METERS
            )
            
            assert len(start_points) > 0
            
            # 验证每个起点
            for start_point in start_points:
                # 验证可通行性
                assert selector.is_point_accessible(start_point[0], start_point[1])
                
                # 验证最小距离约束
                distance = selector.calculate_geo_distance(start_point, end_point['pixel'])
                assert distance >= MIN_START_END_DISTANCE_METERS
    
    def test_invalid_files(self):
        """测试无效文件处理"""
        with pytest.raises(FileNotFoundError):
            PointSelector(
                "nonexistent_landcover.tif",
                "nonexistent_slope.tif"
            )
    
    def test_out_of_bounds(self, test_files):
        """测试边界检查"""
        selector = PointSelector(
            test_files['landcover_path'],
            test_files['slope_path']
        )
        
        # 测试超出范围的坐标
        assert not selector.is_point_accessible(-1, 50)
        assert not selector.is_point_accessible(50, -1)
        assert not selector.is_point_accessible(3000, 50)
        assert not selector.is_point_accessible(50, 3000) 