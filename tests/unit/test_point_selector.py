"""起点选择器单元测试"""

import pytest
import numpy as np
import rasterio
import tempfile
import os
from pathlib import Path
from src.generator.point_selector import PointSelector
from src.generator.config import (
    MAX_SLOPE_THRESHOLD, IMPASSABLE_LANDCOVER_CODES,
    MIN_START_END_DISTANCE_METERS
)

@pytest.fixture
def test_data():
    """生成测试用的环境数据"""
    size = (100, 100)  # 使用较大的尺寸以便测试距离约束
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 生成测试数据
        landcover = np.full(size, 11, dtype=np.int32)  # 默认为平原（可通行）
        landcover[0:10, 0:10] = IMPASSABLE_LANDCOVER_CODES[0]  # 添加不可通行区域
        landcover[90:100, 90:100] = IMPASSABLE_LANDCOVER_CODES[0]  # 添加不可通行区域
        
        slope = np.zeros(size, dtype=np.float32)  # 默认为平地
        slope[10:20, 10:20] = MAX_SLOPE_THRESHOLD + 1  # 添加过陡区域
        slope[80:90, 80:90] = MAX_SLOPE_THRESHOLD + 1  # 添加过陡区域
        
        # 创建测试文件
        meta = {
            'driver': 'GTiff',
            'height': size[0],
            'width': size[1],
            'count': 1,
            'crs': '+proj=latlong',
            'transform': rasterio.transform.Affine(0.001, 0, 0, 0, 0.001, 0)  # 约111米/像素
        }
        
        # 保存landcover数据
        landcover_path = os.path.join(temp_dir, 'landcover.tif')
        meta['dtype'] = np.int32
        with rasterio.open(landcover_path, 'w', **meta) as dst:
            dst.write(landcover, 1)
        
        # 保存slope数据
        slope_path = os.path.join(temp_dir, 'slope.tif')
        meta['dtype'] = np.float32
        with rasterio.open(slope_path, 'w', **meta) as dst:
            dst.write(slope, 1)
        
        yield {
            'landcover_path': landcover_path,
            'slope_path': slope_path,
            'landcover_data': landcover,
            'slope_data': slope,
            'transform': meta['transform']
        }

class TestPointSelector:
    """起点选择器测试类"""
    
    def test_init(self, test_data):
        """测试初始化"""
        selector = PointSelector(
            test_data['landcover_path'],
            test_data['slope_path']
        )
        assert selector.height == 100
        assert selector.width == 100
        assert np.array_equal(selector.landcover_data, test_data['landcover_data'])
        assert np.array_equal(selector.slope_data, test_data['slope_data'])
        assert selector.transform == test_data['transform']
        assert selector.pixel_size_meters == pytest.approx(111.0, rel=0.1)  # 约111米/像素
    
    def test_is_point_accessible(self, test_data):
        """测试点可通行性检查"""
        selector = PointSelector(
            test_data['landcover_path'],
            test_data['slope_path']
        )
        
        # 测试正常点
        assert selector.is_point_accessible(50, 50) == True
        
        # 测试不可通行地物
        assert selector.is_point_accessible(5, 5) == False
        
        # 测试过陡区域
        assert selector.is_point_accessible(15, 15) == False
        
        # 测试边界外的点
        assert selector.is_point_accessible(-1, 0) == False
        assert selector.is_point_accessible(100, 0) == False
    
    def test_calculate_distance(self, test_data):
        """测试距离计算"""
        selector = PointSelector(
            test_data['landcover_path'],
            test_data['slope_path']
        )
        
        # 测试水平距离
        assert selector.calculate_distance((0, 0), (0, 3)) == 3.0
        
        # 测试垂直距离
        assert selector.calculate_distance((0, 0), (4, 0)) == 4.0
        
        # 测试对角线距离
        assert selector.calculate_distance((0, 0), (3, 4)) == 5.0
    
    def test_calculate_geo_distance(self, test_data):
        """测试地理距离计算"""
        selector = PointSelector(
            test_data['landcover_path'],
            test_data['slope_path']
        )
        
        # 测试相邻像素
        dist = selector.calculate_geo_distance((0, 0), (0, 1))
        assert dist == pytest.approx(111.0, rel=0.1)  # 约111米
        
        # 测试对角线
        dist = selector.calculate_geo_distance((0, 0), (1, 1))
        assert dist == pytest.approx(157.0, rel=0.1)  # 约157米（111*√2）
    
    def test_select_start_points(self, test_data):
        """测试起点选择"""
        selector = PointSelector(
            test_data['landcover_path'],
            test_data['slope_path']
        )
        
        # 选择一个终点（中心区域）
        end_point = (50, 50)
        
        # 测试单点选择
        start_points = selector.select_start_points(
            end_point,
            num_points=1,
            min_distance=1000  # 1公里
        )
        assert len(start_points) == 1
        start_point = start_points[0]
        
        # 验证起点可通行性
        assert selector.is_point_accessible(*start_point)
        
        # 验证距离约束
        dist = selector.calculate_geo_distance(start_point, end_point)
        assert dist >= 1000
        
        # 测试多点选择
        start_points = selector.select_start_points(
            end_point,
            num_points=5,
            min_distance=1000
        )
        assert len(start_points) == 5
        
        # 验证所有点的可通行性和距离约束
        for point in start_points:
            assert selector.is_point_accessible(*point)
            dist = selector.calculate_geo_distance(point, end_point)
            assert dist >= 1000
        
        # 验证点之间的最小间距
        for i in range(len(start_points)):
            for j in range(i + 1, len(start_points)):
                dist = selector.calculate_geo_distance(start_points[i], start_points[j])
                assert dist >= 250  # 最小间距为最小距离的1/4
    
    def test_select_start_points_for_all_ends(self, test_data):
        """测试为多个终点选择起点"""
        selector = PointSelector(
            test_data['landcover_path'],
            test_data['slope_path']
        )
        
        # 创建测试终点
        end_points = [
            {'pixel': (50, 50), 'coord': (0.05, 0.05)},
            {'pixel': (30, 70), 'coord': (0.07, 0.03)}
        ]
        
        # 为每个终点选择3个起点
        pairs = selector.select_start_points_for_all_ends(end_points, points_per_end=3)
        
        # 验证结果
        assert len(pairs) == 6  # 2个终点 * 3个起点
        
        # 验证每个起终点对
        for start, end in pairs:
            # 验证起点可通行性
            assert selector.is_point_accessible(*start)
            
            # 验证距离约束
            dist = selector.calculate_geo_distance(start, end)
            assert dist >= MIN_START_END_DISTANCE_METERS
    
    def test_coordinate_conversion(self, test_data):
        """测试坐标转换"""
        selector = PointSelector(
            test_data['landcover_path'],
            test_data['slope_path']
        )
        
        # 测试像素到地理坐标的转换
        pixel = (50, 50)
        lon, lat = selector.pixel_to_geo(pixel)
        assert lon == pytest.approx(0.05, rel=1e-6)
        assert lat == pytest.approx(0.05, rel=1e-6)
        
        # 测试地理到像素坐标的转换
        coord = (0.05, 0.05)
        row, col = selector.geo_to_pixel(coord)
        assert row == 50
        assert col == 50
        
        # 测试转换的可逆性
        pixel2 = selector.geo_to_pixel(selector.pixel_to_geo(pixel))
        assert pixel2[0] == pixel[0]
        assert pixel2[1] == pixel[1] 