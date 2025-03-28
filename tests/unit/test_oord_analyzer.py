"""OORD分析器单元测试"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from src.generator.oord_analyzer import OORDAnalyzer

@pytest.fixture
def test_slope_bins():
    """坡度分组边界值"""
    return [0, 5, 10, 15, 20, 90]

@pytest.fixture
def test_trajectory_data():
    """生成测试用的轨迹数据"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='s'),
        'row': np.random.randint(0, 10, 100),
        'col': np.random.randint(0, 10, 100),
        'speed_mps': np.random.uniform(0, 10, 100),
        'heading_degrees': np.random.uniform(0, 360, 100),
        'turn_rate_dps': np.random.uniform(-30, 30, 100),
        'acceleration_mps2': np.random.uniform(-2, 2, 100)
    })

@pytest.fixture
def test_environment_data():
    """生成测试用的环境数据"""
    size = (10, 10)
    dem = np.random.uniform(0, 1000, size)
    slope = np.random.uniform(0, 45, size)
    aspect = np.random.uniform(0, 360, size)
    landcover = np.random.randint(1, 5, size)
    return dem, slope, aspect, landcover

class TestOORDAnalyzer:
    """OORD分析器测试类"""
    
    def test_init(self, test_slope_bins):
        """测试初始化"""
        analyzer = OORDAnalyzer(test_slope_bins)
        assert analyzer.slope_bins == test_slope_bins
        assert analyzer.min_samples_per_group == 100
        assert len(analyzer.environment_groups) == 0
    
    def test_add_environment_info(self, test_slope_bins, test_trajectory_data,
                                test_environment_data):
        """测试添加环境信息"""
        analyzer = OORDAnalyzer(test_slope_bins)
        dem, slope, aspect, landcover = test_environment_data
        
        # 添加环境信息
        df = analyzer.add_environment_info(test_trajectory_data, dem, slope,
                                         aspect, landcover)
        
        # 验证新增列
        assert 'elevation' in df.columns
        assert 'slope_magnitude' in df.columns
        assert 'slope_aspect' in df.columns
        assert 'landcover' in df.columns
        assert 'delta_angle' in df.columns
        assert 'slope_along_path' in df.columns
        assert 'cross_slope' in df.columns
        assert 'slope_bin' in df.columns
        assert 'group_label' in df.columns
        
        # 验证数值范围
        assert df['elevation'].min() >= 0
        assert df['slope_magnitude'].min() >= 0
        assert df['slope_magnitude'].max() <= 45
        assert df['slope_aspect'].min() >= 0
        assert df['slope_aspect'].max() <= 360
        assert df['landcover'].min() >= 1
        assert df['landcover'].max() <= 4
    
    def test_analyze_groups(self, test_slope_bins, test_trajectory_data,
                          test_environment_data):
        """测试环境组分析"""
        analyzer = OORDAnalyzer(test_slope_bins, min_samples_per_group=10)
        dem, slope, aspect, landcover = test_environment_data
        
        # 添加环境信息并分析
        df = analyzer.add_environment_info(test_trajectory_data, dem, slope,
                                         aspect, landcover)
        analyzer.analyze_groups(df)
        
        # 验证环境组
        assert len(analyzer.environment_groups) > 0
        for group in analyzer.environment_groups.values():
            assert group.count >= 10
            assert 0 <= group.max_speed <= 10
            assert 0 <= group.typical_speed <= 10
            assert group.speed_stddev >= 0
            assert -30 <= group.typical_turn_rate <= 30
            assert -2 <= group.typical_acceleration <= 2
    
    def test_analyze_slope_direction_effect(self, test_slope_bins,
                                          test_trajectory_data,
                                          test_environment_data):
        """测试坡向影响分析"""
        analyzer = OORDAnalyzer(test_slope_bins, min_samples_per_group=10)
        dem, slope, aspect, landcover = test_environment_data
        
        # 添加环境信息并分析
        df = analyzer.add_environment_info(test_trajectory_data, dem, slope,
                                         aspect, landcover)
        effect_params = analyzer.analyze_slope_direction_effect(df)
        
        # 验证参数
        for lc_params in effect_params.values():
            assert lc_params['k_uphill'] > 0
            assert lc_params['k_cross'] > 0
            assert lc_params['max_cross_slope_degrees'] > 0
    
    def test_save_and_load_results(self, test_slope_bins, test_trajectory_data,
                                 test_environment_data):
        """测试保存和加载分析结果"""
        analyzer = OORDAnalyzer(test_slope_bins, min_samples_per_group=10)
        dem, slope, aspect, landcover = test_environment_data
        
        # 添加环境信息并分析
        df = analyzer.add_environment_info(test_trajectory_data, dem, slope,
                                         aspect, landcover)
        analyzer.analyze_groups(df)
        
        # 保存结果
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer.save_analysis_results(temp_dir)
            
            # 创建新的分析器并加载结果
            new_analyzer = OORDAnalyzer(test_slope_bins)
            new_analyzer.load_analysis_results(temp_dir)
            
            # 验证加载的结果
            assert len(new_analyzer.environment_groups) == len(analyzer.environment_groups)
            for label, group in analyzer.environment_groups.items():
                loaded_group = new_analyzer.environment_groups[label]
                assert loaded_group.landcover_code == group.landcover_code
                assert loaded_group.slope_bin == group.slope_bin
                assert loaded_group.count == group.count
                assert loaded_group.max_speed == group.max_speed
                assert loaded_group.typical_speed == group.typical_speed
                assert loaded_group.speed_stddev == group.speed_stddev
    
    def test_invalid_data(self, test_slope_bins):
        """测试无效数据处理"""
        analyzer = OORDAnalyzer(test_slope_bins)
        
        # 测试加载不存在的文件
        with pytest.raises(FileNotFoundError):
            analyzer.load_analysis_results("nonexistent_dir")
        
        # 测试数据形状不匹配
        df = pd.DataFrame({'row': [0], 'col': [0]})
        dem_data = np.zeros((10, 10))
        slope_data = np.zeros((20, 20))  # 形状不匹配
        with pytest.raises(ValueError) as excinfo:
            analyzer.add_environment_info(df, dem_data, slope_data,
                                       np.zeros((10, 10)), np.zeros((10, 10)))
        assert "所有环境数据的形状必须一致" in str(excinfo.value)
        
        # 测试像素坐标超出范围
        df = pd.DataFrame({'row': [15], 'col': [0]})  # 行坐标超出范围
        with pytest.raises(ValueError) as excinfo:
            analyzer.add_environment_info(df, dem_data, dem_data,
                                       dem_data, dem_data)
        assert "像素坐标超出范围" in str(excinfo.value)
        
        # 测试缺少必要的列
        df = pd.DataFrame({'invalid_col': [0]})
        with pytest.raises(ValueError) as excinfo:
            analyzer.add_environment_info(df, dem_data, dem_data,
                                       dem_data, dem_data)
        assert "数据缺少必要的列" in str(excinfo.value) 