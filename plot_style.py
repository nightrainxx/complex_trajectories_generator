"""
matplotlib作图样式配置文件
将此文件放在项目根目录下，在绘图时导入即可使用统一的样式
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

def setup_style():
    """设置matplotlib的默认样式"""
    # 设置字体路径
    FONT_PATHS = {
        'times': '/usr/share/fonts/truetype/custom/times.ttf',
        'simsun': '/usr/share/fonts/truetype/custom/simsun.ttc'
    }
    
    # 添加字体文件
    for font_path in FONT_PATHS.values():
        if Path(font_path).exists():
            mpl.font_manager.fontManager.addfont(font_path)
    
    # 设置默认参数
    plt.rcParams.update({
        'font.family': ['sans-serif'],
        'font.sans-serif': ['SimSun'],  # 中文默认使用宋体
        'font.serif': ['Times New Roman'],  # 英文默认使用Times New Roman
        'axes.unicode_minus': False,  # 解决负号显示问题
        'font.size': 16,  # 默认字体大小
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.figsize': [8, 6],  # 默认图像大小
        'figure.dpi': 100,  # 默认分辨率
        'savefig.dpi': 300,  # 保存图片的分辨率
        'axes.grid': True,  # 默认显示网格
        'grid.alpha': 0.3,  # 网格透明度
        'grid.linestyle': '--',  # 网格线型
        # 设置数学公式字体
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',
        'mathtext.sf': 'Times New Roman',
        'mathtext.tt': 'Times New Roman',
        'mathtext.cal': 'Times New Roman',
        'mathtext.default': 'regular'
    })

# 在导入模块时自动设置样式
setup_style() 