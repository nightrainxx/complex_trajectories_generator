# 轨迹验证工作日志

## 2025-03-28 验证方式一：给定真实路径骨架进行模拟

### 实现思路

1. 从真实轨迹中提取路径骨架
2. 使用路径骨架作为输入,让模拟器生成轨迹
3. 对比模拟轨迹和真实轨迹的动态行为

### 代码实现

1. 修改了`MotionPatternLearner`类:
   - 添加了`learn_from_single_trajectory`方法,支持从单条轨迹学习
   - 简化了初始化参数,移除了对config的依赖
   - 添加了`EnvironmentMaps`数据类来组织环境地图
   - 实现了`generate_environment_maps`方法生成环境地图

2. 修改了`MotionSimulator`类:
   - 添加了经纬度坐标支持
   - 改进了速度控制逻辑
   - 添加了加速度和转向率的计算
   - 实现了基于路径的运动控制

3. 创建了验证脚本`validate_motion_simulation.py`:
   - 实现了轨迹加载和路径骨架提取
   - 实现了从单条轨迹学习运动模式
   - 实现了轨迹模拟和验证过程

### 问题诊断

#### 1. 速度控制完全失控

1. 速度值异常:
   - 模拟速度远超真实速度(RMSE: 163.15 m/s ≈ 600 km/h)
   - 速度变化趋势与真实情况相反(相关系数: -0.99)
   - 总距离差异超过10km

2. 可能的原因:
   - 环境地图中的速度值可能不合理
   - 坡度/坡向影响因子计算或应用可能有误
   - 速度更新逻辑可能跳过了加减速过程
   - 缺乏合理的速度上限约束

#### 2. 转向行为异常

1. 转向控制问题:
   - 转向率RMSE高达187.47°/s
   - 转弯过于剧烈或时机不当
   - 可能缺乏速度与转向的协调

2. 可能的原因:
   - 转向率限制可能未生效
   - 目标朝向计算可能不准确
   - 未考虑速度对转向能力的影响
   - 路径跟随时缺乏预见性减速

#### 3. 动态特征计算问题

1. 数值计算异常:
   - 加速度和转向率出现NaN值
   - 可能存在除零问题
   - 时间步长(dt)可能过小或无效

### 改进方案

#### 1. 速度控制修复(优先级最高)

1. 环境速度图检查:
   ```python
   # 在MotionPatternLearner中添加验证
   def validate_speed_maps(self):
       """验证生成的速度图是否合理"""
       for group in self.env_groups.values():
           if group.typical_speed > 15.0:  # ~54 km/h
               logger.warning(f"组{group.group_label}的典型速度过高: {group.typical_speed:.2f} m/s")
           if group.max_speed > 20.0:      # ~72 km/h
               logger.warning(f"组{group.group_label}的最大速度过高: {group.max_speed:.2f} m/s")
   ```

2. 严格的速度限制:
   ```python
   class MotionSimulator:
       def _calculate_target_speed(self, env_params, slope_effects):
           # 基础目标速度
           target_speed = min(
               env_params['typical_speed'],
               env_params['max_speed']
           )
           
           # 应用坡度影响
           slope_factor = max(0.5, 1.0 - self.config['SLOPE_SPEED_FACTOR'] * 
                            abs(slope_effects['along_path']))
           cross_slope_factor = max(0.5, 1.0 - self.config['CROSS_SLOPE_FACTOR'] * 
                                  abs(slope_effects['cross']))
           
           # 应用地形影响
           target_speed *= slope_factor * cross_slope_factor
           
           # 强制应用绝对速度限制
           return np.clip(target_speed, self.config['MIN_SPEED'], self.config['MAX_SPEED'])
   ```

3. 改进加速度控制:
   ```python
   def _update_speed(self, current_speed, target_speed, dt):
       """更新速度,确保平滑加减速"""
       speed_diff = target_speed - current_speed
       
       # 根据当前速度调整加减速限制
       if speed_diff > 0:
           max_change = min(
               self.config['MAX_ACCELERATION'] * dt,
               0.2 * current_speed  # 限制单次加速不超过当前速度的20%
           )
       else:
           max_change = max(
               -self.config['MAX_DECELERATION'] * dt,
               -0.3 * current_speed  # 限制单次减速不超过当前速度的30%
           )
       
       speed_change = np.clip(speed_diff, -max_change, max_change)
       return current_speed + speed_change
   ```

#### 2. 转向控制改进

1. 速度感知的转向限制:
   ```python
   def _calculate_max_turn_rate(self, speed):
       """根据速度计算最大转向率"""
       # 速度越高,允许的转向率越小
       base_rate = self.config['MAX_TURN_RATE']  # 基础转向率(度/秒)
       speed_factor = 1.0 / (1.0 + speed * 0.2)  # 速度影响因子
       return base_rate * speed_factor
   ```

2. 预见性转弯控制:
   ```python
   def _look_ahead_path_curvature(self, current_pos, path_points, look_ahead=3):
       """计算前方路径的曲率"""
       if len(path_points) < look_ahead + 1:
           return 0.0
           
       points = path_points[:look_ahead+1]
       angles = []
       for i in range(len(points)-2):
           v1 = points[i+1] - points[i]
           v2 = points[i+2] - points[i+1]
           angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
           angles.append(abs(angle))
           
       return max(angles) if angles else 0.0
   ```

#### 3. 调试和验证改进

1. 详细的状态日志:
   ```python
   def _log_simulation_state(self, agent, target, env_params):
       """记录模拟状态"""
       logger.debug(
           f"T={agent.timestamp:.1f} "
           f"Pos=({agent.position[0]:.1f},{agent.position[1]:.1f}) "
           f"Speed={agent.speed:.2f} "
           f"Heading={np.degrees(agent.heading):.1f}° "
           f"Target=({target[0]:.1f},{target[1]:.1f}) "
           f"TypicalSpeed={env_params['typical_speed']:.2f} "
           f"MaxSpeed={env_params['max_speed']:.2f}"
       )
   ```

2. 可视化工具增强:
   ```python
   def plot_trajectory_comparison(real_traj, sim_traj, output_path):
       """绘制轨迹对比图"""
       fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
       
       # 速度对比
       ax1.plot(real_traj['timestamp'], real_traj['speed_mps'], 'b-', label='真实')
       ax1.plot(sim_traj['timestamp'], sim_traj['speed_mps'], 'r--', label='模拟')
       ax1.set_ylabel('速度 (m/s)')
       ax1.legend()
       
       # 加速度对比
       ax2.plot(real_traj['timestamp'], real_traj['acceleration_mps2'], 'b-')
       ax2.plot(sim_traj['timestamp'], sim_traj['acceleration_mps2'], 'r--')
       ax2.set_ylabel('加速度 (m/s²)')
       
       # 转向率对比
       ax3.plot(real_traj['timestamp'], real_traj['turn_rate_dps'], 'b-')
       ax3.plot(sim_traj['timestamp'], sim_traj['turn_rate_dps'], 'r--')
       ax3.set_ylabel('转向率 (°/s)')
       
       plt.savefig(output_path)
   ```

### 下一步工作计划

1. 速度控制修复(第一阶段):
   - 实现并测试速度地图验证功能
   - 添加严格的速度限制逻辑
   - 改进加速度控制机制
   - 验证基本的速度控制效果

2. 转向控制改进(第二阶段):
   - 实现速度感知的转向限制
   - 添加路径预测和曲率计算
   - 完善转弯减速机制
   - 测试各种转向场景

3. 验证系统完善(持续进行):
   - 实现详细的状态日志记录
   - 开发可视化对比工具
   - 建立完整的验证测试集
   - 收集和分析验证数据

### 参考数值

1. 速度相关:
   - 典型速度范围: 2-10 m/s
   - 最大速度: 10-15 m/s
   - 最小速度: 2 m/s
   - 最大加速度: 2.0 m/s²
   - 最大减速度: 3.0 m/s²

2. 转向相关:
   - 最大转向率: 20-30 °/s
   - 转弯减速起始距离: 20m
   - 最小转弯速度: 2 m/s

3. 地形影响:
   - 坡度影响因子: 0.05
   - 横向坡度因子: 0.1
   - 最大允许坡度: 30° 