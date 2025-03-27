"""环境地图生成器单元测试"""

import pytest
import numpy as np
import rasterio
from pathlib import Path

from src.generator.environment_mapper import EnvironmentMapper
from src.generator.config import (
    MAX_SPEED, MAX_SLOPE_THRESHOLD, SLOPE_SPEED_FACTOR,
    TYPICAL_SPEED_FACTOR, UP_SLOPE_FACTOR, DOWN_SLOPE_FACTOR, CROSS_SLOPE_FACTOR,
    BASE_SPEED_STDDEV_FACTOR, SLOPE_STDDEV_FACTOR, COMPLEX_TERRAIN_STDDEV_FACTOR,
    COMPLEX_TERRAIN_CODES, LANDCOVER_SPEED_FACTORS, LANDCOVER_COST_FACTORS,
    IMPASSABLE_LANDCOVER_CODES
)

@pytest.fixture
def test_data():
    """创建测试用数据"""
    # 创建3000x3000的测试数据
    landcover = np.full((3000, 3000), 31, dtype=np.int32)  # 默认为草地
    
    # 添加一些建筑用地和农田
    landcover[400:500, 100:200] = 11  # 建筑用地
    landcover[1000:1100, 2600:2700] = 21  # 农田
    
    # 添加一些不可通行区域
    landcover[500:600, 500:600] = 81  # 水体
    landcover[1500:1600, 1500:1600] = 82  # 冰川
    
    # 创建坡度大小数据
    slope_magnitude = np.zeros((3000, 3000), dtype=np.float32)
    slope_magnitude[700:800, 700:800] = 50.0  # 陡峭区域
    slope_magnitude[2000:2100, 2000:2100] = 45.0  # 较陡区域
    
    # 创建坡向数据（北为0度，顺时针）
    slope_aspect = np.full((3000, 3000), -1, dtype=np.float32)  # -1表示平地
    slope_aspect[700:800, 700:800] = 45.0  # 东北向
    slope_aspect[2000:2100, 2000:2100] = 180.0  # 南向
    
    return {
        'landcover': landcover,
        'slope_magnitude': slope_magnitude,
        'slope_aspect': slope_aspect
    }

@pytest.fixture
def test_files(tmp_path, test_data):
    """创建测试用文件"""
    # 创建测试用GeoTIFF文件
    transform = rasterio.transform.from_origin(-6.0, 58.0, 0.001, 0.001)
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
    
    # 保存坡度大小文件
    slope_magnitude_path = tmp_path / "slope_magnitude.tif"
    meta['dtype'] = np.float32
    with rasterio.open(slope_magnitude_path, 'w', **meta) as dst:
        dst.write(test_data['slope_magnitude'], 1)
    
    # 保存坡向文件
    slope_aspect_path = tmp_path / "slope_aspect.tif"
    with rasterio.open(slope_aspect_path, 'w', **meta) as dst:
        dst.write(test_data['slope_aspect'], 1)
    
    return {
        'landcover_path': landcover_path,
        'slope_magnitude_path': slope_magnitude_path,
        'slope_aspect_path': slope_aspect_path
    }

class TestEnvironmentMapper:
    """环境地图生成器测试类"""
    
    def test_init(self, test_files):
        """测试初始化"""
        mapper = EnvironmentMapper(
            test_files['landcover_path'],
            test_files['slope_magnitude_path'],
            test_files['slope_aspect_path']
        )
        assert mapper.landcover_data is not None
        assert mapper.slope_magnitude_data is not None
        assert mapper.slope_aspect_data is not None
        assert mapper.transform is not None
    
    def test_calculate_max_speed_map(self, test_files):
        """测试最大速度地图计算"""
        mapper = EnvironmentMapper(
            test_files['landcover_path'],
            test_files['slope_magnitude_path'],
            test_files['slope_aspect_path']
        )
        
        max_speed_map = mapper.calculate_max_speed_map()
        
        # 验证地图形状
        assert max_speed_map.shape == mapper.landcover_data.shape
        
        # 验证不可通行区域速度为0
        water_mask = mapper.landcover_data == 81
        glacier_mask = mapper.landcover_data == 82
        assert np.all(max_speed_map[water_mask] == 0)
        assert np.all(max_speed_map[glacier_mask] == 0)
        
        # 验证陡峭区域速度受限
        steep_mask = mapper.slope_magnitude_data > MAX_SLOPE_THRESHOLD
        assert np.all(max_speed_map[steep_mask] == 0)
        
        # 验证正常区域速度合理
        normal_mask = ~(water_mask | glacier_mask | steep_mask)
        assert np.all(max_speed_map[normal_mask] > 0)
        assert np.all(max_speed_map[normal_mask] <= MAX_SPEED)
    
    def test_calculate_typical_speed_map(self, test_files):
        """测试典型速度地图计算"""
        mapper = EnvironmentMapper(
            test_files['landcover_path'],
            test_files['slope_magnitude_path'],
            test_files['slope_aspect_path']
        )
        
        typical_speed_map = mapper.calculate_typical_speed_map()
        
        # 验证地图形状
        assert typical_speed_map.shape == mapper.landcover_data.shape
        
        # 验证不可通行区域速度为0
        water_mask = mapper.landcover_data == 81
        glacier_mask = mapper.landcover_data == 82
        assert np.all(typical_speed_map[water_mask] == 0)
        assert np.all(typical_speed_map[glacier_mask] == 0)
        
        # 验证陡峭区域速度受限
        steep_mask = mapper.slope_magnitude_data > MAX_SLOPE_THRESHOLD
        assert np.all(typical_speed_map[steep_mask] == 0)
        
        # 验证正常区域速度合理
        normal_mask = ~(water_mask | glacier_mask | steep_mask)
        assert np.all(typical_speed_map[normal_mask] > 0)
        assert np.all(typical_speed_map[normal_mask] <= MAX_SPEED)
        assert np.all(typical_speed_map <= mapper.calculate_max_speed_map())
    
    def test_calculate_speed_stddev_map(self, test_files):
        """测试速度标准差地图计算"""
        mapper = EnvironmentMapper(
            test_files['landcover_path'],
            test_files['slope_magnitude_path'],
            test_files['slope_aspect_path']
        )
        
        stddev_map = mapper.calculate_speed_stddev_map()
        
        # 验证地图形状
        assert stddev_map.shape == mapper.landcover_data.shape
        
        # 验证不可通行区域标准差为0
        water_mask = mapper.landcover_data == 81
        glacier_mask = mapper.landcover_data == 82
        assert np.all(stddev_map[water_mask] == 0)
        assert np.all(stddev_map[glacier_mask] == 0)
        
        # 验证陡峭区域标准差为0
        steep_mask = mapper.slope_magnitude_data > MAX_SLOPE_THRESHOLD
        assert np.all(stddev_map[steep_mask] == 0)
        
        # 验证正常区域标准差合理
        normal_mask = ~(water_mask | glacier_mask | steep_mask)
        assert np.all(stddev_map[normal_mask] >= 0)
        assert np.all(stddev_map[normal_mask] <= MAX_SPEED / 2)  # 标准差不应超过最大速度的一半
    
    def test_calculate_cost_map(self, test_files):
        """测试成本地图计算"""
        mapper = EnvironmentMapper(
            test_files['landcover_path'],
            test_files['slope_magnitude_path'],
            test_files['slope_aspect_path']
        )
        
        cost_map = mapper.calculate_cost_map()
        
        # 验证地图形状
        assert cost_map.shape == mapper.landcover_data.shape
        
        # 验证不可通行区域成本为无穷大
        water_mask = mapper.landcover_data == 81
        glacier_mask = mapper.landcover_data == 82
        assert np.all(np.isinf(cost_map[water_mask]))
        assert np.all(np.isinf(cost_map[glacier_mask]))
        
        # 验证陡峭区域成本为无穷大
        steep_mask = mapper.slope_magnitude_data > MAX_SLOPE_THRESHOLD
        assert np.all(np.isinf(cost_map[steep_mask]))
        
        # 验证正常区域成本合理
        normal_mask = ~(water_mask | glacier_mask | steep_mask)
        assert np.all(cost_map[normal_mask] > 0)  # 成本应为正值
        assert np.all(np.isfinite(cost_map[normal_mask]))  # 成本应为有限值
    
    def test_save_environment_maps(self, test_files, tmp_path):
        """测试环境地图保存"""
        mapper = EnvironmentMapper(
            test_files['landcover_path'],
            test_files['slope_magnitude_path'],
            test_files['slope_aspect_path']
        )
        
        # 计算各种地图
        max_speed_map = mapper.calculate_max_speed_map()
        typical_speed_map = mapper.calculate_typical_speed_map()
        speed_stddev_map = mapper.calculate_speed_stddev_map()
        cost_map = mapper.calculate_cost_map()
        
        # 保存地图
        output_dir = tmp_path / "environment_maps"
        output_dir.mkdir(exist_ok=True)
        
        mapper.save_environment_maps(
            str(output_dir),
            max_speed_map,
            typical_speed_map,
            speed_stddev_map,
            cost_map
        )
        
        # 验证文件是否存在
        assert (output_dir / "max_speed_map.tif").exists()
        assert (output_dir / "typical_speed_map.tif").exists()
        assert (output_dir / "speed_stddev_map.tif").exists()
        assert (output_dir / "cost_map.tif").exists()
        
        # 验证文件内容
        with rasterio.open(output_dir / "max_speed_map.tif") as src:
            saved_max_speed = src.read(1)
            assert np.array_equal(saved_max_speed, max_speed_map)
            assert src.transform == mapper.transform
            assert src.crs == mapper.meta['crs']
    
    def test_invalid_files(self):
        """测试无效文件处理"""
        with pytest.raises(FileNotFoundError):
            EnvironmentMapper(
                "nonexistent_landcover.tif",
                "nonexistent_slope_magnitude.tif",
                "nonexistent_slope_aspect.tif"
            )
    
    def test_inconsistent_shapes(self, test_files, tmp_path):
        """测试数据形状不一致处理"""
        # 创建形状不一致的数据
        small_data = np.zeros((100, 100), dtype=np.float32)
        small_file = tmp_path / "small_slope.tif"
        
        meta = {
            'driver': 'GTiff',
            'height': 100,
            'width': 100,
            'count': 1,
            'dtype': np.float32,
            'crs': '+proj=latlong',
            'transform': rasterio.transform.from_origin(0, 0, 1, 1)
        }
        
        with rasterio.open(small_file, 'w', **meta) as dst:
            dst.write(small_data, 1)
        
        with pytest.raises(ValueError):
            EnvironmentMapper(
                test_files['landcover_path'],
                str(small_file),
                test_files['slope_aspect_path']
            ) 