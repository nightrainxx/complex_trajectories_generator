# 复杂轨迹生成器

基于地形和环境特征的复杂轨迹生成系统。

## 项目结构

```
complex_trajectories_generator/
├── data/
│   ├── input/
│   │   ├── gis/              # GIS数据
│   │   │   ├── dem_30m_100km.tif
│   │   │   └── landcover_30m_100km.tif
│   │   └── oord/             # OORD数据
│   └── output/
│       ├── intermediate/      # 中间结果
│       ├── trajectory_generation/  # 生成的轨迹
│       └── evaluation/        # 评估结果
├── src/
│   ├── core/
│   │   ├── terrain/          # 地形模块
│   │   │   ├── analyzer.py   # 地形分析
│   │   │   └── loader.py     # 地形数据加载
│   │   └── trajectory/       # 轨迹模块
│   │       ├── generator.py  # 轨迹生成器基类
│   │       └── environment_based.py  # 基于环境的生成器
│   └── utils/
│       ├── config.py         # 配置管理
│       └── logging_utils.py  # 日志工具
├── tests/                    # 单元测试
├── examples/                 # 使用示例
└── requirements.txt          # 项目依赖
```

## 功能特性

1. 地形分析
   - 加载和处理DEM数据
   - 计算坡度和坡向
   - 分析土地覆盖类型
   - 评估地形可通行性

2. 轨迹生成
   - 基于环境特征的路径规划
   - 考虑地形影响的速度规划
   - 平滑的轨迹插值
   - 合理的朝向计算

3. 配置管理
   - 统一的配置接口
   - 可定制的参数设置
   - 环境变量支持

4. 可视化
   - 轨迹可视化
   - 速度分析
   - 地形叠加显示

## 安装

1. 创建conda环境：
```bash
conda create -n wargame python=3.8
conda activate wargame
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用示例

1. 准备数据
   - 将DEM数据放在 `data/input/gis/dem_30m_100km.tif`
   - 将土地覆盖数据放在 `data/input/gis/landcover_30m_100km.tif`

2. 运行示例脚本：
```bash
python examples/generate_trajectory.py
```

3. 查看结果
   - 轨迹数据：`data/output/trajectory_generation/example_trajectory.json`
   - 轨迹图形：`data/output/trajectory_generation/example_trajectory.png`

## 开发指南

1. 代码规范
   - 使用类型注解
   - 编写详细的文档字符串
   - 遵循PEP 8规范

2. 测试
   - 运行单元测试：`python -m unittest discover tests`
   - 测试覆盖率：`coverage run -m unittest discover tests`
   - 生成覆盖率报告：`coverage report`

3. 日志
   - 使用 `logging` 模块记录日志
   - 配置日志级别和格式

## 配置说明

1. 路径配置
   - 输入数据路径
   - 输出数据路径
   - 中间文件路径

2. 地形配置
   - 坡度分级
   - 地物编码
   - 不可通行条件

3. 运动配置
   - 时间步长
   - 速度约束
   - 加速度约束
   - 转向约束

4. 生成配置
   - 轨迹数量
   - 距离约束
   - 环境组标签

## 注意事项

1. 数据要求
   - DEM数据分辨率：30米
   - 土地覆盖数据分辨率：30米
   - 坐标系统：EPSG:4326

2. 性能考虑
   - 大规模数据处理时注意内存使用
   - 轨迹生成过程可能需要较长时间
   - 可以通过调整参数优化性能

3. 限制条件
   - 最大坡度限制
   - 不可通行地物类型
   - 轨迹平滑度要求

## 维护者

- 作者：[Your Name]
- 邮箱：[your.email@example.com]

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 