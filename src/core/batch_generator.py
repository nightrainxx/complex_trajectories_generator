"""
批量轨迹生成器模块
用于批量生成满足约束的轨迹
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.config import config
from src.core.terrain import TerrainLoader
from src.core.point_selector import PointSelector
from src.core.trajectory import EnvironmentBasedGenerator

logger = logging.getLogger(__name__)

class BatchGenerator:
    """批量轨迹生成器"""
    
    def __init__(
            self,
            terrain_loader: TerrainLoader,
            output_dir: Path = config.paths.TRAJECTORY_DIR,
            num_workers: int = 4
        ):
        """
        初始化批量生成器
        
        Args:
            terrain_loader: 地形数据加载器
            output_dir: 输出目录
            num_workers: 并行工作进程数
        """
        self.terrain_loader = terrain_loader
        self.output_dir = output_dir
        self.num_workers = num_workers
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建轨迹生成器
        self.trajectory_generator = EnvironmentBasedGenerator(
            terrain_loader=terrain_loader,
            dt=config.motion.DT,
            max_waypoints=config.generation.MAX_WAYPOINTS,
            min_waypoint_dist=config.generation.MIN_WAYPOINT_DIST,
            max_waypoint_dist=config.generation.MAX_WAYPOINT_DIST
        )
        
        # 创建起终点选择器
        self.point_selector = PointSelector(
            terrain_loader=terrain_loader,
            min_distance=config.generation.MIN_START_END_DISTANCE_METERS,
            num_end_points=config.generation.NUM_END_POINTS,
            num_trajectories=config.generation.NUM_TRAJECTORIES_TO_GENERATE
        )
        
    def generate_batch(self) -> List[Path]:
        """
        批量生成轨迹
        
        Returns:
            List[Path]: 生成的轨迹文件路径列表
        """
        logger.info("开始批量生成轨迹...")
        
        # 选择起终点对
        generation_pairs = self.point_selector.select_points()
        logger.info(f"已选择{len(generation_pairs)}对起终点")
        
        # 并行生成轨迹
        trajectory_files = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有生成任务
            future_to_pair = {
                executor.submit(
                    self._generate_single_trajectory,
                    start_point,
                    end_point,
                    idx
                ): (start_point, end_point, idx)
                for idx, (start_point, end_point) in enumerate(generation_pairs)
            }
            
            # 收集结果
            for future in as_completed(future_to_pair):
                start_point, end_point, idx = future_to_pair[future]
                try:
                    trajectory_file = future.result()
                    trajectory_files.append(trajectory_file)
                    logger.info(f"轨迹{idx+1}生成完成: {trajectory_file}")
                except Exception as e:
                    logger.error(
                        f"生成轨迹{idx+1}失败: "
                        f"start={start_point}, end={end_point}, "
                        f"error={str(e)}"
                    )
                    
        logger.info(f"批量生成完成，共生成{len(trajectory_files)}条轨迹")
        return trajectory_files
        
    def _generate_single_trajectory(
            self,
            start_point: Tuple[float, float],
            end_point: Tuple[float, float],
            idx: int
        ) -> Path:
        """
        生成单条轨迹
        
        Args:
            start_point: 起点坐标
            end_point: 终点坐标
            idx: 轨迹索引
            
        Returns:
            Path: 轨迹文件路径
        """
        # 生成轨迹
        trajectory = self.trajectory_generator.generate_trajectory(
            start_point,
            end_point
        )
        
        # 添加元数据
        trajectory['metadata'] = {
            'start_point': start_point,
            'end_point': end_point,
            'index': idx,
            'generation_time': None  # 将在保存时添加时间戳
        }
        
        # 保存轨迹
        trajectory_file = self.output_dir / f"trajectory_{idx+1}.json"
        self._save_trajectory(trajectory, trajectory_file)
        
        return trajectory_file
        
    def _save_trajectory(
            self,
            trajectory: Dict[str, Any],
            file_path: Path
        ) -> None:
        """
        保存轨迹数据
        
        Args:
            trajectory: 轨迹数据
            file_path: 保存路径
        """
        # 添加生成时间
        from datetime import datetime
        trajectory['metadata']['generation_time'] = (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # 保存为JSON文件
        with open(file_path, 'w') as f:
            json.dump(trajectory, f, indent=2) 