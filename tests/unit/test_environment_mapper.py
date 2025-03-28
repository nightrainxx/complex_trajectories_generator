"""环境地图生成器单元测试"""

import pytest
import numpy as np
import rasterio
import tempfile
import os
from pathlib import Path
from src.generator.environment_mapper import EnvironmentMapper
from src.generator.config import (
    MAX_SPEED, MAX_SLOPE_THRESHOLD, IMPASSABLE_LANDCOVER_CODES,
    COMPLEX_TERRAIN_CODES
)

@pytest.fixture
def test_data():
    """生成测试用的环境数据"""
    size = (10, 10)
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 生成测试数据
        landcover = np.random.randint(1, 5, size, dtype=np.int32)  # 1-4为可通行地物
        landcover[0, 0] = IMPASSABLE_LANDCOVER_CODES[0]  # 添加一个不可通行点
        
        slope_magnitude = np.random.uniform(0, 30, size).astype(np.float32)  # 0-30度
        slope_magnitude[0, 1] = MAX_SLOPE_THRESHOLD + 1  # 添加一个过陡点
        
        slope_aspect = np.random.uniform(0, 360, size).astype(np.float32)  # 0-360度
        slope_aspect[1, 1] = -1  # 添加一个平地点
        
        # 创建测试文件
        meta = {
            'driver': 'GTiff',
            'height': size[0],
            'width': size[1],
            'count': 1,
            'crs': None,
            'transform': None
        }
        
        # 保存landcover数据
        landcover_path = os.path.join(temp_dir, 'landcover.tif')
        meta['dtype'] = np.int32
        with rasterio.open(landcover_path, 'w', **meta) as dst:
            dst.write(landcover, 1)
        
        # 保存slope_magnitude数据
        slope_magnitude_path = os.path.join(temp_dir, 'slope_magnitude.tif')
        meta['dtype'] = np.float32
        with rasterio.open(slope_magnitude_path, 'w', **meta) as dst:
            dst.write(slope_magnitude, 1)
        
        # 保存slope_aspect数据
        slope_aspect_path = os.path.join(temp_dir, 'slope_aspect.tif')
        with rasterio.open(slope_aspect_path, 'w', **meta) as dst:
            dst.write(slope_aspect, 1)
        
        yield {
            'landcover_path': landcover_path,
            'slope_magnitude_path': slope_magnitude_path,
            'slope_aspect_path': slope_aspect_path,
            'landcover_data': landcover,
            'slope_magnitude_data': slope_magnitude,
            'slope_aspect_data': slope_aspect
        }

class TestEnvironmentMapper:
    """环境地图生成器测试类"""
    
    def test_init(self, test_data):
        """测试初始化"""
        mapper = EnvironmentMapper(
            test_data['landcover_path'],
            test_data['slope_magnitude_path'],
            test_data['slope_aspect_path']
        )
        assert mapper.height == 10
        assert mapper.width == 10
        assert np.array_equal(mapper.landcover_data, test_data['landcover_data'])
        assert np.array_equal(mapper.slope_magnitude_data, test_data['slope_magnitude_data'])
        assert np.array_equal(mapper.slope_aspect_data, test_data['slope_aspect_data'])
    
    def test_calculate_max_speed_map(self, test_data):
        """测试最大速度地图计算"""
        mapper = EnvironmentMapper(
            test_data['landcover_path'],
            test_data['slope_magnitude_path'],
            test_data['slope_aspect_path']
        )
        max_speed_map = mapper.calculate_max_speed_map()
        
        # 验证不可通行区域
        assert max_speed_map[0, 0] == 0  # 不可通行地物
        assert max_speed_map[0, 1] == 0  # 过陡区域
        
        # 验证可通行区域
        passable_mask = ~np.isin(test_data['landcover_data'], IMPASSABLE_LANDCOVER_CODES)
        passable_mask &= test_data['slope_magnitude_data'] <= MAX_SLOPE_THRESHOLD
        assert np.all(max_speed_map[passable_mask] > 0)
        assert np.all(max_speed_map[passable_mask] <= MAX_SPEED)
    
    def test_calculate_typical_speed_map(self, test_data):
        """测试典型速度地图计算"""
        mapper = EnvironmentMapper(
            test_data['landcover_path'],
            test_data['slope_magnitude_path'],
            test_data['slope_aspect_path']
        )
        typical_speed_map = mapper.calculate_typical_speed_map()
        
        # 验证平地点
        flat_point = typical_speed_map[1, 1]
        max_speed_at_flat = mapper.calculate_max_speed_map()[1, 1]
        assert flat_point <= max_speed_at_flat
        
        # 验证所有点的速度范围
        assert np.all(typical_speed_map >= 0)
        assert np.all(typical_speed_map <= MAX_SPEED)
    
    def test_calculate_speed_stddev_map(self, test_data):
        """测试速度标准差地图计算"""
        mapper = EnvironmentMapper(
            test_data['landcover_path'],
            test_data['slope_magnitude_path'],
            test_data['slope_aspect_path']
        )
        speed_stddev_map = mapper.calculate_speed_stddev_map()
        
        # 验证不可通行区域
        assert speed_stddev_map[0, 0] == 0  # 不可通行地物
        assert speed_stddev_map[0, 1] == 0  # 过陡区域
        
        # 验证复杂地形
        complex_mask = np.isin(test_data['landcover_data'], COMPLEX_TERRAIN_CODES)
        if np.any(complex_mask):
            typical_speed = mapper.calculate_typical_speed_map()
            assert np.all(speed_stddev_map[complex_mask] > typical_speed[complex_mask] * 0.1)
    
    def test_calculate_cost_map(self, test_data):
        """测试成本地图计算"""
        mapper = EnvironmentMapper(
            test_data['landcover_path'],
            test_data['slope_magnitude_path'],
            test_data['slope_aspect_path']
        )
        cost_map = mapper.calculate_cost_map()
        
        # 验证不可通行区域
        assert np.isinf(cost_map[0, 0])  # 不可通行地物
        assert np.isinf(cost_map[0, 1])  # 过陡区域
        
        # 验证可通行区域
        passable_mask = ~np.isin(test_data['landcover_data'], IMPASSABLE_LANDCOVER_CODES)
        passable_mask &= test_data['slope_magnitude_data'] <= MAX_SLOPE_THRESHOLD
        assert np.all(cost_map[passable_mask] > 0)
        assert np.all(np.isfinite(cost_map[passable_mask]))
    
    def test_save_environment_maps(self, test_data):
        """测试环境地图保存"""
        mapper = EnvironmentMapper(
            test_data['landcover_path'],
            test_data['slope_magnitude_path'],
            test_data['slope_aspect_path']
        )
        
        # 计算所有地图
        max_speed_map = mapper.calculate_max_speed_map()
        typical_speed_map = mapper.calculate_typical_speed_map()
        speed_stddev_map = mapper.calculate_speed_stddev_map()
        cost_map = mapper.calculate_cost_map()
        
        # 创建临时目录并保存
        with tempfile.TemporaryDirectory() as temp_dir:
            mapper.save_environment_maps(
                temp_dir,
                max_speed_map,
                typical_speed_map,
                speed_stddev_map,
                cost_map
            )
            
            # 验证文件是否存在
            assert os.path.exists(os.path.join(temp_dir, 'max_speed_map.tif'))
            assert os.path.exists(os.path.join(temp_dir, 'typical_speed_map.tif'))
            assert os.path.exists(os.path.join(temp_dir, 'speed_stddev_map.tif'))
            assert os.path.exists(os.path.join(temp_dir, 'cost_map.tif'))
            
            # 验证文件内容
            with rasterio.open(os.path.join(temp_dir, 'max_speed_map.tif')) as src:
                saved_max_speed = src.read(1)
                assert np.array_equal(saved_max_speed, max_speed_map)
    
    def test_get_environment_params(self, test_data):
        """测试获取环境参数"""
        mapper = EnvironmentMapper(
            test_data['landcover_path'],
            test_data['slope_magnitude_path'],
            test_data['slope_aspect_path']
        )
        
        # 测试正常点
        params = mapper.get_environment_params(5, 5)
        assert 'max_speed' in params
        assert 'typical_speed' in params
        assert 'speed_stddev' in params
        assert 'cost' in params
        assert 'landcover' in params
        assert 'slope_magnitude' in params
        assert 'slope_aspect' in params
        assert params['max_speed'] >= 0
        assert params['typical_speed'] >= 0
        assert params['speed_stddev'] >= 0
        assert params['cost'] >= 0
        
        # 测试不可通行点
        params = mapper.get_environment_params(0, 0)
        assert params['max_speed'] == 0
        assert params['typical_speed'] == 0
        assert params['speed_stddev'] == 0
        assert np.isinf(params['cost'])
        
        # 测试无效坐标
        with pytest.raises(ValueError):
            mapper.get_environment_params(-1, 0)
        with pytest.raises(ValueError):
            mapper.get_environment_params(10, 0) 