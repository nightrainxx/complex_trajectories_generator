"""批量轨迹生成器主控脚本

负责协调各个模块，完成从地形分析到轨迹生成的全流程。

输入:
- 配置文件
- GIS数据
- OORD数据

输出:
- 批量生成的轨迹
- 评估报告
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import json

from .terrain_analyzer import TerrainAnalyzer
from .environment_mapper import EnvironmentMapper
from .point_selector import PointSelector
from .path_planner import PathPlanner
from .motion_simulator import MotionSimulator, EnvironmentParams
from .evaluator import Evaluator

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchGenerator:
    """批量轨迹生成器类"""

    def __init__(self, config_path: str):
        """初始化批量生成器

        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path) as f:
            self.config = json.load(f)

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_dir = Path(self.config['output_dir']) / f"synthetic_batch_{timestamp}"
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置副本
        with open(self.batch_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

        # 初始化组件
        self.terrain_analyzer = TerrainAnalyzer(
            self.config['dem_path'],
            self.config['output_dir']
        )
        self.environment_mapper = EnvironmentMapper(
            self.config,
            self.config['output_dir']
        )
        self.point_selector = PointSelector(self.config)
        self.path_planner = PathPlanner(self.config)
        self.motion_simulator = MotionSimulator()
        self.evaluator = Evaluator(
            self.config,
            self.batch_dir / 'evaluation'
        )

    def generate_batch(self) -> None:
        """执行批量轨迹生成流程"""
        logger.info("开始批量生成轨迹...")

        # 1. 地形分析
        logger.info("步骤1: 地形分析")
        self.terrain_analyzer.calculate_terrain_attributes()

        # 2. 构建环境地图
        logger.info("步骤2: 构建环境地图")
        learned_params = self._load_learned_params()
        self.environment_mapper.build_environment_maps(learned_params)

        # 3. 选择起终点对
        logger.info("步骤3: 选择起终点对")
        generation_pairs = self.point_selector.select_start_end_pairs()
        if not generation_pairs:
            logger.error("未能找到合适的起终点对")
            return

        # 4. 为每对起终点生成轨迹
        logger.info("步骤4: 生成轨迹")
        successful_count = 0
        for i, (start_point, end_point) in enumerate(generation_pairs):
            try:
                # 规划路径
                path = self.path_planner.plan_path(start_point, end_point)
                if not path:
                    logger.warning(f"无法为第 {i+1} 对起终点规划路径")
                    continue

                # 模拟运动
                trajectory = self.motion_simulator.simulate_motion(
                    path,
                    self._get_environment_params
                )

                # 保存轨迹
                self._save_trajectory(trajectory, i + 1)
                successful_count += 1

            except Exception as e:
                logger.error(f"生成第 {i+1} 条轨迹时出错: {str(e)}")
                continue

        logger.info(f"成功生成 {successful_count} 条轨迹")

        # 5. 评估生成的轨迹
        if successful_count > 0:
            logger.info("步骤5: 评估轨迹")
            self.evaluator.evaluate_batch(
                str(self.batch_dir),
                self.config['oord_processed_path']
            )

        logger.info("批量生成完成")

    def _load_learned_params(self) -> Dict:
        """加载学习到的参数

        Returns:
            Dict: 学习到的参数
        """
        params_path = Path(self.config['output_dir']) / 'intermediate/learned_params.json'
        if params_path.exists():
            with open(params_path) as f:
                return json.load(f)
        return {}  # 如果文件不存在，返回空字典

    def _get_environment_params(self, lon: float, lat: float) -> EnvironmentParams:
        """获取指定位置的环境参数

        Args:
            lon: 经度
            lat: 纬度

        Returns:
            EnvironmentParams: 环境参数
        """
        # 从环境地图获取基础参数
        max_speed, typical_speed, speed_stddev = \
            self.environment_mapper.get_environment_params(lon, lat)

        # 获取地形信息
        slope, aspect, _, _ = self.terrain_analyzer.get_terrain_info(lon, lat)

        # 创建环境参数对象
        return EnvironmentParams(
            max_speed=max_speed,
            typical_speed=typical_speed,
            speed_stddev=speed_stddev,
            slope_magnitude=slope,
            slope_aspect=aspect
        )

    def _save_trajectory(self, trajectory: List[Tuple], index: int) -> None:
        """保存轨迹到文件

        Args:
            trajectory: 轨迹点列表
            index: 轨迹索引
        """
        import pandas as pd

        # 创建DataFrame
        df = pd.DataFrame(
            trajectory,
            columns=['timestamp', 'lon', 'lat', 'speed_mps', 'heading_degrees']
        )

        # 保存为CSV
        output_path = self.batch_dir / f'trajectory_{index:04d}.csv'
        df.to_csv(output_path, index=False)
        logger.debug(f"轨迹已保存到: {output_path}")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='批量生成轨迹')
    parser.add_argument('config', help='配置文件路径')
    args = parser.parse_args()

    try:
        generator = BatchGenerator(args.config)
        generator.generate_batch()
    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 