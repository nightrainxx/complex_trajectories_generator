项目：基于 OORD 数据学习与精细环境约束的合成轨迹生成器 (复杂版)

版本: 1.2
日期: 2025-03-27
目标读者: 开发工程师 (需要具备数据分析、GIS处理与模拟基础)

1. 项目概述与目标

目标: 开发一个高级轨迹生成工具，该工具首先从真实轨迹数据集 (OORD) 中学习目标的运动特性及其与环境（DEM、坡度大小、坡度方向、土地覆盖）的交互规律，然后利用这些知识，在更大的地理范围内批量生成具有逼真动态行为的合成轨迹。

核心要求:

数据驱动学习: 从 OORD 数据中量化分析并提取目标在不同环境下的速度、转向等行为模式。

精细环境感知: 生成的轨迹不仅受坡度大小和土地覆盖影响，还必须考虑坡度方向（坡向）与车辆行驶方向的相互作用（如上下坡、横坡行驶），这会动态影响速度和可行性。

批量生成: 能够自动生成指定数量（N条）的轨迹。

特定起终点: 用户指定终点区域特征（如靠近城市），程序自动选择满足距离约束（如>80km）的起点。

可控属性: 轨迹总长度（主要通过起终点选择实现）和平均速度（通过模拟参数和学习结果影响）应大致可控且符合真实性。

动态真实性: 生成的轨迹应体现速度变化、加减速、以及与精细环境匹配的随机性，力求在多维度统计特性上与 OORD 数据相似。

可评估性: 提供量化和可视化方法，用于评估生成的合成轨迹与 OORD 数据的相似度。

最终交付物:

一个或多个 Python 脚本/模块，能执行学习、地图生成、起终点选择、批量轨迹模拟、评估的全流程。

批量的合成轨迹数据文件（如 CSV）。

评估报告（图表和统计数据）。

学习结果和中间地图文件（可选）。

2. 输入数据

GIS 环境数据 (需放置在 data/input/gis/):

dem_30m_100km.tif: 数字高程模型 (WGS 84, ~30m res)。

landcover_30m_100km.tif: 土地覆盖数据 (分类编码, 与 DEM 对齐)。

(新增/计算生成) slope_magnitude_30m_100km.tif: 坡度大小 (单位：度, 从 DEM 计算)。

(新增/计算生成) slope_aspect_30m_100km.tif: 坡度方向/坡向 (单位：度, 北为0, 顺时针, 从 DEM 计算)。

(备选/计算生成) dzdx_30m_100km.tif 和 dzdy_30m_100km.tif: X 和 Y 方向的地形梯度（如果选择使用梯度向量而非坡度+坡向）。

OORD 轨迹数据 (需放置在 data/input/oord/):

多个原始轨迹文件 (如 CSV/GPX，包含时间戳、经纬度坐标)。

配置文件 (config.py):

数据文件路径。

(新增) NUM_TRAJECTORIES_TO_GENERATE: 要生成的轨迹总数 (e.g., 500)。

(新增) NUM_END_POINTS: 要选择的固定终点数量 (e.g., 3)。

(新增) MIN_START_END_DISTANCE_METERS: 起终点最小直线距离 (e.g., 80000)。

(新增) URBAN_LANDCOVER_CODES: 代表城市/建成区的地物编码列表 (e.g., [1, 10])。

(新增) IMPASSABLE_LANDCOVER_CODES: 代表绝对不可通行的地物编码列表 (e.g., [11], 水体)。

坡度离散化区间 (SLOPE_BINS)。

模拟参数 (dt, MAX_ACCELERATION, MAX_DECELERATION)。

(新增) 车辆稳定性参数 (可选，如最大允许横坡坡度 MAX_CROSS_SLOPE_DEGREES)。

输出目录路径。

3. 输出数据

核心输出 (放置在 data/output/synthetic_batch_XXX/):

trajectory_1.csv, trajectory_2.csv, ..., trajectory_N.csv: 每个文件包含一条完整的合成轨迹 (timestamp, row, col, lon, lat, speed_mps, heading_degrees)。

评估输出 (放置在 data/output/evaluation_report_XXX/):

.png 图表文件：全局/分组运动参数分布对比图。

.log 或 .txt 文件：统计量比较、K-S 检验结果等。

中间/学习结果 (可选，放置在 data/output/intermediate/):

学习到的规则/模型 (learned_params.pkl 或 .json)。

增强的环境地图 (max_speed_map.tif, typical_speed_map.tif, speed_stddev_map.tif, cost_map.tif)。 注意：这里的速度图主要基于坡度大小学习。

计算出的坡度/坡向/梯度图 (如果不在输入中提供)。

4. 技术栈与依赖库 (保持 V1.0 推荐)

Python 3.x, rasterio, numpy, pandas, geopandas, scipy, scikit-learn (可选), pathfinding/skimage.graph, matplotlib, seaborn, logging。

(新增/可选) richdem 或其他库用于更高效的地形属性计算。

5. 详细实现步骤

阶段 0: 初始化与配置

设置环境: 创建虚拟环境并安装所有依赖。

配置管理 (config.py): 定义所有路径、参数、阈值。

日志设置: 配置 logging 模块记录详细信息。

阶段 1: 数据准备与地形分析 (扩展)

加载 GIS 数据: 使用 rasterio 加载 DEM, Landcover。记录元数据。处理 NoData。

计算地形属性 (关键补充):

计算坡度大小 (Slope Magnitude): 从 DEM 计算每个像素的坡度（度）。保存为 slope_magnitude_30m_100km.tif。

计算坡向 (Slope Aspect): 从 DEM 计算每个像素的坡向（度，北为0，顺时针）。处理平坦区域（通常坡向设为 -1 或特定值）。保存为 slope_aspect_30m_100km.tif。

(备选) 计算梯度: 使用 numpy.gradient 或类似方法计算 dz/dx 和 dz/dy。保存为 dzdx...tif, dzdy...tif。

工具: 可使用 gdaldem 命令行工具、rasterio 配合 numpy 或 richdem 库。确保输出与 DEM 对齐。

加载与预处理 OORD 数据:

同 V1.0 指南：读取、统一坐标系、计算瞬时速度、朝向、转向率、加速度。

将地理坐标转换为像素坐标 (row, col)。

存储为包含 timestamp, row, col, lon, lat, speed_mps, heading, turn_rate_dps, acceleration_mps2, trajectory_id 的 DataFrame。

阶段 2: 学习运动特性与环境交互 (可选增强)

关联环境信息:

遍历处理后的 OORD DataFrame。

查询并添加对应的 elevation, slope_magnitude, slope_aspect (或 dzdx, dzdy), landcover。

定义环境分组:

主要依据：landcover 和 离散化的 slope_magnitude 等级。

创建环境组标签 group_label = f"LC{lc_code}_S{slope_mag_label}"。

分组统计分析 (基于坡度大小):

按 group_label 分组。

计算各组的速度 (mean, median, std, max/quantile), 转向率, 加速度统计量。

建立环境-运动规则/模型 (查找表或拟合函数):

目标：确定各组的 MaxSpeed_learned, TypicalSpeed_learned, SpeedStdDev_learned 等 (主要反映坡度大小和地物影响)。

处理数据不足的组。

保存学习结果。

(高级/可选) 学习坡向影响:

在每个 OORD 点，计算其行驶方向 (heading) 与地形坡向 (aspect) 的关系，得到 slope_along_path_oord 和 cross_slope_oord。

尝试分析 OORD 速度与这两个方向性坡度指标的关系（可能需要更复杂的模型，如分段回归或基于规则的分析）。这可以用于直接指导模拟阶段的速度调整函数 f 和 g，而非仅基于设定的规则。

阶段 3: 构建增强的环境地图

初始化地图数组: 创建 max_speed_map, typical_speed_map, speed_stddev_map, cost_map (以及加载或确认 slope_magnitude_map, slope_aspect_map, dzdx_map, dzdy_map 已准备好)。

像素级计算 (速度图):

遍历栅格，获取 landcover_value, slope_magnitude_value。

确定 group_label。

从阶段 2 学习结果查询或计算得到 max_s, typ_s, std_s (主要基于坡度大小)。

填充 max_speed_map, typical_speed_map, speed_stddev_map。

计算成本图 (cost_map for A - 简化方案):*

基于 typical_speed_map (坡度大小影响) 计算成本：cost = pixel_size / typical_speed (如果 typical_speed > 0)。不可通行区域成本设为 np.inf。

注意: 这个成本图主要反映基于坡度大小和地物的平均通行难度，用于简化 A* 规划。

(可选) 保存所有生成的地图。

阶段 4: 批量起终点选择 (新增)

(移至此处或作为独立模块 point_selector.py)

实现 select_start_end_pairs 函数 (参考补丁 V1.1 描述):

加载 landcover_array。

根据 URBAN_LANDCOVER_CODES 随机选择 NUM_END_POINTS 个不同的、可通行的终点 selected_end_points。

对于每个 end_point，循环随机选择候选起点 start_point_cand：

检查起点可通行性 (is_accessible)。

计算与 end_point 的直线距离 (使用像素大小)。

如果距离 > MIN_START_END_DISTANCE_METERS，接受该起点，将其与 end_point 配对，加入 generation_pairs 列表。

持续此过程，直到为所有终点找到足够数量的起点，达到 NUM_TRAJECTORIES_TO_GENERATE 的目标。

记录选择过程和结果。

返回 generation_pairs 列表 [(start1, end1), (start2, end2), ...]。

阶段 5: 批量合成轨迹生成 (核心模拟，包含坡向逻辑)

(移至此处或作为主控脚本 batch_generator.py)

主循环: 遍历 generation_pairs 列表中的每一对 (start_point, end_point)。

5.1 路径规划 (A) - 简化方案:*

输入: start_point, end_point, 以及阶段 3 生成的 基于坡度大小的 cost_map。

运行 A* 找到最低成本路径 path = [(r0, c0), ..., (rn, cn)]。

如果找不到路径，记录错误并跳过当前对。

5.2 Agent-Based 运动模拟 (时间步进 - 关键修改):

初始化: Agent 状态 (pos, speed=0, heading, time=0), 轨迹列表, 路径索引。设置 dt。

模拟循环 (直到接近终点):

获取当前环境参数:

根据 agent_pos 查询 max_speed_map, typical_speed_map, speed_stddev_map 得到 base_max_s, base_typ_s, base_std_s (基于坡度大小)。

查询坡度方向信息: 查询 slope_magnitude_map, slope_aspect_map (或 dzdx_map, dzdy_map) 得到 current_slope_mag, current_aspect (或 current_dzdx, current_dzdy)。

计算方向性坡度指标:

获取 Agent 当前朝向 current_heading (度)。

计算 delta_angle = current_heading - current_aspect (处理角度环绕)。

slope_along_path = current_slope_mag * cos(radians(delta_angle))

cross_slope = current_slope_mag * abs(sin(radians(delta_angle)))

(备选) 使用梯度: heading_vector = (sin(radians(h)), cos(radians(h))), slope_vector = (current_dzdx, current_dzdy). slope_along_path = dot(heading_vector, slope_vector) / ||slope_vector|| (如果需要单位坡度)。计算横坡需要更复杂的向量运算。

动态调整速度约束 (核心修改):

max_speed_adjusted = base_max_s # 初始值

target_speed_base = base_typ_s + np.random.normal(0, base_std_s) # 基础目标+随机性

应用坡度方向约束 (示例规则):

reduction_factor_uphill = max(0.1, 1 - k_uphill * max(0, slope_along_path)) # 上坡减速

reduction_factor_downhill = 1 + k_downhill * max(0, -slope_along_path) # 下坡可能轻微加速，但需上限

reduction_factor_cross = max(0.05, 1 - k_cross * cross_slope**2) # 横坡急剧减速 (平方项更敏感)

max_speed_adjusted *= reduction_factor_uphill * reduction_factor_cross

target_speed_adjusted = target_speed_base * reduction_factor_uphill * reduction_factor_cross

(重要) max_speed_adjusted = np.clip(max_speed_adjusted, 0, MAX_BRAKING_LIMITED_SPEED_ON_DOWNHILL) # 下坡制动限制

(重要) if cross_slope > MAX_CROSS_SLOPE_DEGREES: max_speed_adjusted = min(max_speed_adjusted, VERY_LOW_SPEED) # 超过横坡阈值极大限速

target_speed = np.clip(target_speed_adjusted, 0, max_speed_adjusted) # 最终目标速度

应用加速度限制: 计算 accel_needed, 限制 actual_accel, 更新 next_speed。

最终速度约束: next_speed = np.clip(next_speed, 0, max_speed_adjusted)。

确定目标朝向: 指向下一个路径点 path[path_index]。

应用转向限制 (可选): 基于学习或设定的转向率限制。更新 next_heading。

更新位置: delta_dist = next_speed * dt, 计算 next_pos。

更新 Agent 状态, 更新时间, 记录轨迹点。

路径点切换 & 终止条件。

5.3 (可选) 验证与迭代: 检查生成轨迹的长度和平均速度。注意：引入坡向约束后，实际平均速度可能低于仅基于坡度大小的预期。

5.4 保存轨迹: 将生成的 trajectory 列表保存为唯一的 CSV 文件到批处理输出目录。

记录日志: 详细记录每条轨迹的生成过程和结果。

阶段 6: 评估 (新增)

(作为独立模块 evaluator.py 或脚本)

加载数据: 实现 load_synthetic_data 加载指定批次目录的所有轨迹；实现 load_processed_oord_data 加载处理好的 OORD 数据 (确保包含所需列，包括 group_label 如果要做环境交互比较)。

执行比较:

全局统计比较: 调用 compare_global_distributions 比较 speed_mps, acceleration_mps2, turn_rate_dps 等的分布 (KDE图, 统计量, K-S检验)。

环境交互比较: 调用 compare_environment_interaction 按 group_label (基于地物和坡度大小) 比较 speed_mps 分布。

(可选/较难) 局部路径比较: 实现 compare_local_segments，选择 OORD 片段，强制模拟器跟随路径，比较速度剖面。

可视化检查: 随机抽取几条合成轨迹与 OORD 轨迹叠加在 DEM/Slope/Landcover 地图上进行目视检查，特别关注在不同坡向下的行为是否合理。

保存报告: 将图表和统计结果保存到评估输出目录。

6. 代码结构与最佳实践

模块化: data_loader.py, terrain_analyzer.py (新增), oord_analyzer.py, environment_mapper.py, point_selector.py (新增), path_planner.py, simulator.py (核心修改), evaluator.py (新增), batch_generator.py (或修改 main.py), config.py, utils.py。

配置分离: 所有参数放入 config.py。

测试: 对地形分析、点选择、模拟器核心逻辑（特别是速度调整部分）、评估函数编写单元/集成测试。

文档 & 日志: 详细 README，清晰 Docstrings，注释复杂逻辑（如坡向约束规则），使用 logging 记录过程。

版本控制 (Git)。

7. 潜在挑战与注意事项 (更新)

OORD 数据质量与代表性。

坡向学习的复杂度: 从 OORD 中可靠地学习坡向影响可能需要大量高质量数据和复杂模型。初期可先依赖基于物理直觉的规则。

方向性 A* 的复杂度: 如果简化 A* 方案导致路径选择不佳，实现方向感知的 A* 会显著增加计算成本和实现难度。

模拟参数调优: dt, 加速度限制, 以及新增的坡向影响因子 (k_uphill, k_cross 等) 和阈值 (MAX_CROSS_SLOPE_DEGREES) 需要仔细调整和验证。

像素级模拟精度。

评估指标的解释: K-S 检验在高样本量下易显著；应更关注分布形状和关键统计量的接近程度。可视化非常重要。

这份 V1.2 指南整合了我们讨论的所有要点，为开发一个更真实、功能更完善的合成轨迹生成器提供了详细的蓝图。实施时建议逐步进行，先确保地形分析和模拟器中坡向逻辑的基础实现，再完善批量处理和评估部分。

注意激活wargame虚拟环境再运行。