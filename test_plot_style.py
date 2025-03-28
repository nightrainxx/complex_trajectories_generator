import matplotlib.pyplot as plt
import numpy as np
import plot_style  # 导入样式设置

# 创建测试数据
x = np.linspace(0, 10, 100)
y1 = 1e3 * np.sin(x)
y2 = 1e-3 * np.cos(x)

# 创建图形
fig, ax = plt.subplots()

# 绘制曲线
ax.plot(x, y1, 'b-', label='正弦曲线')
ax.plot(x, y2, 'r--', label='余弦曲线')

# 设置标题和标签
ax.set_title('三角函数测试图')
ax.set_xlabel('时间 (s)')
ax.set_ylabel(r'幅值 ($\times 10^{-3}$)')  # 使用原始字符串

# 添加图例
ax.legend()

# 保存图片
plt.savefig('test_style.png', bbox_inches='tight')
print("图片已保存为 test_style.png") 