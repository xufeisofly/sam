import logging
import sys

logger = logging.getLogger('default')
loglevel = logging.INFO

def init_logging():
    setup_logger(loglevel)
    
    
def set_loglevel(level):
    global loglevel
    loglevel = level


def setup_logger(level):
    global logger

    # 清理现有的处理器，避免重复添加
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    # 控制台日志
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s:%(lineno)s:%(funcName)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 文件日志
    file_handler = logging.FileHandler('access.log')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
