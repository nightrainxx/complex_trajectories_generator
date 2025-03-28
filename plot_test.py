import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.font_manager import FontProperties

def setup_plot_style():
    """设置绘图全局样式"""
    try:
        # 设置字体
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Serif CJK SC']  # 中文使用Noto字体
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 16
        print("成功设置字体样式")
    except Exception as e:
        print(f"设置样式时出错: {str(e)}")
        sys.exit(1)

def test_plot():
    """测试绘图函数"""
    try:
        # 设置绘图样式
        setup_plot_style()
        
        # 创建测试数据
        x = np.linspace(0, 10, 100)
        y1 = 1e3 * np.sin(x)
        y2 = 1e-3 * np.cos(x)
        
        print("成功创建测试数据")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制曲线
        ax.plot(x, y1, 'b-', label='正弦曲线')
        ax.plot(x, y2, 'r--', label='余弦曲线')
        
        print("成功绘制曲线")
        
        # 设置标题和标签
        ax.set_title('三角函数测试图', fontsize=16)
        ax.set_xlabel('时间 (s)', fontsize=16)
        ax.set_ylabel('幅值 ($\\times 10^{-3}$)', fontsize=16)
        
        # 设置图例
        ax.legend(fontsize=16)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        print("正在保存图片...")
        # 保存图片
        plt.savefig('test_figure.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("图片已保存为 test_figure.png")
        
    except Exception as e:
        print(f"绘图过程中出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_plot() 