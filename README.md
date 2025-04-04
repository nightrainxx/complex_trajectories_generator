# 复杂轨迹生成器

基于地形和环境约束的复杂轨迹生成系统。

## 功能特点

- 考虑地形（坡度、坡向）影响的轨迹生成
- 支持多种地物类型的速度约束
- 智能的起终点选择
- 基于A*算法的路径规划
- 真实的运动模拟
- 与OORD数据的对比评估

## 系统要求

- Python 3.8+
- GDAL 3.0+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- richdem
- rasterio

## 安装步骤

1. 创建并激活conda环境：

```bash
conda create -n wargame python=3.8
conda activate wargame
```

2. 安装依赖：

```bash
conda install -c conda-forge gdal
conda install -c conda-forge rasterio
conda install numpy pandas matplotlib seaborn
conda install -c conda-forge richdem
```

3. 克隆代码仓库：

```bash
git clone [repository_url]
cd complex_trajectories_generator
```

## 数据准备

在运行系统之前，请准备以下数据：

1. DEM数据（GeoTIFF格式）
   - 分辨率：30m
   - 覆盖范围：100km x 100km
   - 保存路径：`data/input/gis/dem_30m_100km.tif`

2. 土地覆盖数据（GeoTIFF格式）
   - 分辨率：与DEM相同
   - 覆盖范围：与DEM相同
   - 保存路径：`data/input/gis/landcover_30m_100km.tif`
   - 编码说明：
     - 1: 城市
     - 2: 农田
     - 3: 林地
     - 4: 草地
     - 5: 灌木
     - 10: 建成区
     - 11: 水体

## 配置说明

系统的主要配置参数在 `config.py` 文件中：

- 路径配置：输入/输出文件路径
- 轨迹生成参数：数量、距离约束等
- 地物编码和速度因子
- 运动约束参数
- 地形约束参数
- 评估指标设置

## 使用方法

1. 激活conda环境：

```bash
conda activate wargame
```

2. 运行轨迹生成：

```bash
python main.py --log-file logs/generation.log
```

## 输出说明

系统会在 `data/output` 目录下生成以下内容：

1. 中间结果（`intermediate/`）：
   - 坡度图
   - 坡向图
   - 速度图
   - 成本图

2. 生成的轨迹（`synthetic_batch_[timestamp]/`）：
   - 轨迹CSV文件
   - 配置文件副本
   - 评估报告和图表

## 评估报告

评估报告包含以下内容：

1. 全局统计比较：
   - 速度分布
   - 加速度分布
   - 转向率分布

2. 环境交互分析：
   - 不同坡度下的速度分布
   - 不同地物类型的速度分布

## 注意事项

1. 确保输入数据的坐标系统、分辨率和覆盖范围一致
2. 检查磁盘空间是否足够（建议预留10GB）
3. 对于大规模生成，建议调整配置文件中的参数

## 开发团队

- 作者：[作者名]
- 联系方式：[邮箱]

## 许可证

[许可证类型] 