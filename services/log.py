"""统一日志配置：输出时间、级别、模块与 key=value 字段，替代静默吞异常。"""

import logging
import os

_CONFIGURED = False


def setup_logging():
    """幂等初始化；日志级别可用 LOG_LEVEL 覆盖（默认 INFO）。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger("app")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name):
    setup_logging()
    return logging.getLogger(f"app.{name}")
