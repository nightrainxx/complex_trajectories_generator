"""
日志工具模块
用于配置和管理日志
"""

import logging
from pathlib import Path
import matplotlib
import PIL
import rasterio
import sklearn

def setup_logging(log_file: str = None, log_level: int = logging.INFO) -> None:
    """
    设置日志配置
    
    Args:
        log_file: 日志文件路径，如果为None则只输出到控制台
        log_level: 日志级别，默认为INFO
    """
    # 创建根日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('rasterio').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING) 