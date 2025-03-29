"""
matplotlib作图样式配置文件
将此文件放在项目根目录下，在绘图时导入即可使用统一的样式
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 添加字体文件路径
font_paths = [
    '/usr/share/fonts/truetype/custom/simsun.ttc',
    '/usr/share/fonts/truetype/custom/times.ttf',
    '/usr/share/fonts/windows/simsun.ttc',  # Windows字体路径
    '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf',  # Linux字体路径
    '/home/yzc/.fonts/simsun.ttc',  # 用户目录字体路径
    '/home/yzc/.fonts/times.ttf'
]

# 检查并添加字体
for font_path in font_paths:
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)

# 设置全局字体样式
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 设置英文和中文字体
plt.rcParams['font.size'] = 16  # 统一字体大小
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置图形基本样式
plt.rcParams['figure.figsize'] = [8, 6]  # 默认图像大小
plt.rcParams['figure.dpi'] = 100  # 显示分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存分辨率

# 设置网格线样式
plt.rcParams['axes.grid'] = True  # 默认显示网格线
plt.rcParams['grid.alpha'] = 0.3  # 网格线透明度
plt.rcParams['grid.linestyle'] = '--'  # 网格线样式为虚线
plt.rcParams['grid.color'] = 'gray'  # 网格线颜色

# 设置其他样式
plt.rcParams['axes.axisbelow'] = True  # 网格线显示在图形下方
plt.rcParams['axes.labelpad'] = 10  # 轴标签和轴线之间的距离
plt.rcParams['legend.fontsize'] = 16  # 图例字体大小
plt.rcParams['xtick.direction'] = 'in'  # 刻度线朝内
plt.rcParams['ytick.direction'] = 'in'  # 刻度线朝内 