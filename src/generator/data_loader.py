"""数据加载器模块

此模块负责加载和预处理OORD轨迹数据，功能包括：
1. 读取轨迹文件(CSV/GPX)
2. 统一坐标系
3. 计算运动参数(速度、朝向、转向率、加速度)
4. 将地理坐标转换为像素坐标

输入:
    - OORD轨迹文件 (CSV/GPX格式)
    - GIS环境数据的地理参考信息

输出:
    - 预处理后的轨迹DataFrame，包含：
        - timestamp: 时间戳
        - row, col: 像素坐标
        - lon, lat: 地理坐标
        - speed_mps: 速度(米/秒)
        - heading_degrees: 朝向(度)
        - turn_rate_dps: 转向率(度/秒)
        - acceleration_mps2: 加速度(米/秒²)
        - trajectory_id: 轨迹ID
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import logging
from typing import List, Tuple, Union
import rasterio
from rasterio.transform import Affine
import gpxpy
from datetime import datetime
import pytz

class DataLoader:
    """数据加载器类"""
    
    def __init__(self, gis_transform: Affine, gis_shape: Tuple[int, int]):
        """初始化数据加载器
        
        Args:
            gis_transform: GIS数据的地理变换矩阵
            gis_shape: GIS数据的形状(height, width)
        """
        self.transform = gis_transform
        self.height, self.width = gis_shape
        self.logger = logging.getLogger(__name__)
    
    def load_trajectory_file(self, file_path: str) -> pd.DataFrame:
        """加载单个轨迹文件
        
        支持CSV和GPX格式。
        CSV格式要求包含timestamp、longitude、latitude列。
        
        Args:
            file_path: 轨迹文件路径
            
        Returns:
            pd.DataFrame: 预处理后的轨迹数据
        """
        # 获取文件扩展名
        ext = Path(file_path).suffix.lower()
        
        # 根据文件类型选择加载方法
        if ext == '.csv':
            df = self._load_csv(file_path)
        elif ext == '.gpx':
            df = self._load_gpx(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        # 预处理数据
        df = self._preprocess_trajectory(df)
        
        return df
    
    def load_all_trajectories(self, data_dir: str) -> pd.DataFrame:
        """加载目录下的所有轨迹文件
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            pd.DataFrame: 合并后的轨迹数据
        """
        # 获取所有轨迹文件
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        gpx_files = glob.glob(os.path.join(data_dir, "*.gpx"))
        all_files = csv_files + gpx_files
        
        if not all_files:
            raise FileNotFoundError(f"在目录 {data_dir} 中未找到轨迹文件")
        
        # 加载所有文件
        dfs = []
        for file_path in all_files:
            try:
                df = self.load_trajectory_file(file_path)
                # 添加轨迹ID
                df['trajectory_id'] = Path(file_path).stem
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"加载文件 {file_path} 失败: {e}")
                continue
        
        # 合并所有数据
        if not dfs:
            raise ValueError("没有成功加载任何轨迹文件")
        
        merged_df = pd.concat(dfs, ignore_index=True)
        return merged_df
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """加载CSV格式的轨迹文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            pd.DataFrame: 原始轨迹数据
        """
        try:
            df = pd.read_csv(file_path)
            required_cols = ['timestamp', 'longitude', 'latitude']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV文件缺少必需的列: {required_cols}")
            
            # 确保时间戳格式正确
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        except Exception as e:
            raise ValueError(f"加载CSV文件失败: {e}")
    
    def _load_gpx(self, file_path: str) -> pd.DataFrame:
        """加载GPX格式的轨迹文件
        
        Args:
            file_path: GPX文件路径
            
        Returns:
            pd.DataFrame: 原始轨迹数据
        """
        try:
            with open(file_path, 'r') as gpx_file:
                gpx = gpxpy.parse(gpx_file)
            
            data = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        data.append({
                            'timestamp': point.time.replace(tzinfo=None),
                            'longitude': point.longitude,
                            'latitude': point.latitude
                        })
            
            return pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"加载GPX文件失败: {e}")
    
    def _preprocess_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理轨迹数据
        
        1. 计算像素坐标
        2. 计算速度
        3. 计算朝向
        4. 计算转向率
        5. 计算加速度
        
        Args:
            df: 原始轨迹数据
            
        Returns:
            pd.DataFrame: 预处理后的轨迹数据
        """
        # 确保时间戳已排序
        df = df.sort_values('timestamp')
        
        # 计算像素坐标
        rows, cols = self._convert_to_pixel_coords(df['latitude'].values, df['longitude'].values)
        df['row'] = rows
        df['col'] = cols
        
        # 计算时间差(秒)
        df['dt'] = df['timestamp'].diff().dt.total_seconds()
        
        # 计算位移和速度
        df['dx'] = df['longitude'].diff() * 111320 * np.cos(np.radians(df['latitude']))  # 米
        df['dy'] = df['latitude'].diff() * 111320  # 米
        df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)  # 米
        df['speed_mps'] = df['distance'] / df['dt']
        
        # 计算朝向(度)
        df['heading_degrees'] = np.degrees(np.arctan2(df['dx'], df['dy']))
        df['heading_degrees'] = (90 - df['heading_degrees']) % 360  # 转换为北为0
        
        # 计算转向率(度/秒)
        df['heading_change'] = df['heading_degrees'].diff()
        # 处理角度环绕
        df.loc[df['heading_change'] > 180, 'heading_change'] -= 360
        df.loc[df['heading_change'] < -180, 'heading_change'] += 360
        df['turn_rate_dps'] = df['heading_change'] / df['dt']
        
        # 计算加速度(米/秒²)
        df['acceleration_mps2'] = df['speed_mps'].diff() / df['dt']
        
        # 清理临时列和无效值
        df = df.drop(['dx', 'dy', 'distance', 'dt', 'heading_change'], axis=1)
        df = df.fillna(0)  # 第一个点的差分值设为0
        
        # 移除异常值
        df = self._remove_outliers(df)
        
        return df
    
    def _convert_to_pixel_coords(self, lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """将地理坐标转换为像素坐标
        
        Args:
            lats: 纬度数组
            lons: 经度数组
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (行坐标数组, 列坐标数组)
        """
        # 使用仿射变换矩阵进行转换
        cols, rows = ~self.transform * (lons, lats)
        
        # 确保坐标在有效范围内
        rows = np.clip(rows, 0, self.height - 1)
        cols = np.clip(cols, 0, self.width - 1)
        
        return rows.astype(int), cols.astype(int)
    
    def _remove_outliers(self, df: pd.DataFrame, speed_threshold: float = 50.0,
                        acc_threshold: float = 10.0, turn_rate_threshold: float = 90.0) -> pd.DataFrame:
        """移除异常值
        
        Args:
            df: 轨迹数据
            speed_threshold: 速度阈值(米/秒)
            acc_threshold: 加速度阈值(米/秒²)
            turn_rate_threshold: 转向率阈值(度/秒)
            
        Returns:
            pd.DataFrame: 清理后的数据
        """
        # 创建掩码
        mask = (
            (df['speed_mps'] <= speed_threshold) &
            (abs(df['acceleration_mps2']) <= acc_threshold) &
            (abs(df['turn_rate_dps']) <= turn_rate_threshold)
        )
        
        # 记录被移除的点数
        removed_count = (~mask).sum()
        if removed_count > 0:
            self.logger.warning(f"移除了 {removed_count} 个异常值")
        
        return df[mask].copy() 