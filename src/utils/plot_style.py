"""
统一的绘图样式设置
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

def set_style():
    """设置统一的绘图样式"""
    # 添加自定义字体
    font_paths = [
        '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf',
        '/home/yzc/.fonts/simsun.ttc'
    ]
    
    for font_path in font_paths:
        fm.fontManager.addfont(font_path)
        
    # 字体设置
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.unicode_minus'] = False
    
    # 图形基本设置
    plt.rcParams['figure.figsize'] = [8, 6]  # 默认图像大小
    plt.rcParams['figure.dpi'] = 100  # 显示分辨率
    plt.rcParams['savefig.dpi'] = 300  # 保存分辨率
    
    # 网格线设置
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.color'] = 'gray'
    
    # 坐标轴刻度设置
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    
    # 图例设置
    plt.rcParams['legend.fontsize'] = 16
    
    # 标题设置
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16

# 在导入时自动设置样式
set_style() 