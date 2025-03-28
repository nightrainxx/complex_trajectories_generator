"""安装配置文件"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="complex_trajectories_generator",
    version="0.1",
    author="作者名",
    author_email="邮箱",
    description="基于地形和环境约束的复杂轨迹生成系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/complex_trajectories_generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "rasterio",
        "geopandas",
        "scikit-learn",
        "pathfinding",
        "scikit-image",
        "richdem"
    ],
    entry_points={
        "console_scripts": [
            "generate_trajectories=src.main:main",
        ],
    },
) 