import logging
import sys

logger = logging.getLogger('default')

def setup_logger(level):
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s:%(lineno)s:%(funcName)s - %(message)s")
    handler.setFormatter(formatter)

    file_handler = logging.FileHandler('access.log')  # 输出到文件 app.log
    file_handler.setLevel(level)  # 如果需要设置文件日志级别，可以这里指定
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s:%(lineno)s:%(funcName)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(handler)
    logger.addHandler(file_handler)