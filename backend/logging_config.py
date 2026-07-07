"""
结构化日志配置 (Logging Config)

JSON 格式结构化日志，按日轮转，记录每次请求的关键信息。

日志字段:
    - timestamp: ISO 8601 时间戳
    - request_id: 请求唯一标识
    - level: 日志级别
    - message: 日志消息
    - extra: 额外上下文 (user_message, response, rag_used, validation_score, etc.)

用法:
    from backend.logging_config import setup_logging
    logger = setup_logging()

    然后在 app.py 中使用:
    logger.info("chat_request", extra={...})
"""

import os
import sys
import json
import logging
import uuid
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

LOG_DIR = Path(__file__).parent.parent / "logs"


class JsonFormatter(logging.Formatter):
    """JSON 格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": getattr(record, "request_id", "N/A"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 添加额外字段
        if hasattr(record, "extra_fields"):
            log_entry["extra"] = record.extra_fields

        # 添加异常信息
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, ensure_ascii=False)


class DailyRotatingFileHandler(logging.Handler):
    """按日期轮转的文件 Handler"""

    def __init__(self, log_dir: Path, prefix: str = "firefly"):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.prefix = prefix
        self.current_date = None
        self.file = None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._open_file()

    def _get_filename(self) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        return str(self.log_dir / f"{self.prefix}_{today}.jsonl")

    def _open_file(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.current_date:
            if self.file:
                self.file.close()
            filename = self._get_filename()
            self.file = open(filename, 'a', encoding='utf-8')
            self.current_date = today

    def emit(self, record: logging.LogRecord):
        self._open_file()
        msg = self.format(record)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
        super().close()


class RequestIdFilter(logging.Filter):
    """为日志记录添加 request_id"""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = str(uuid.uuid4())[:8]
        return True


def setup_logging(log_level: str = "INFO",
                  log_to_file: bool = True,
                  log_to_console: bool = True) -> logging.Logger:
    """
    配置结构化日志。

    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger("firefly")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    # 添加 request_id filter
    req_filter = RequestIdFilter()

    # 控制台输出（人类可读格式）
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "[%(levelname)s] %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(req_filter)
        logger.addHandler(console_handler)

    # 文件输出（JSON 格式，按日轮转）
    if log_to_file:
        file_handler = DailyRotatingFileHandler(LOG_DIR, "firefly")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(req_filter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """获取 firefly logger（如果未配置则自动初始化）"""
    logger = logging.getLogger("firefly")
    if not logger.handlers:
        return setup_logging()
    return logger


# 方便使用的快捷方法
def log_chat_request(request_id: str, user_message: str,
                    response: str, rag_used: bool = True,
                    validation_score: int = None,
                    generation_time_ms: int = None):
    """记录聊天请求"""
    logger = get_logger()
    extra = {
        "user_message": user_message[:200],
        "response": response[:200],
        "rag_used": rag_used,
    }
    if validation_score is not None:
        extra["validation_score"] = validation_score
    if generation_time_ms is not None:
        extra["generation_time_ms"] = generation_time_ms

    logger.info("chat_request", extra={"type": "chat", **extra})


def log_error(request_id: str, error: Exception, context: dict = None):
    """记录错误"""
    logger = get_logger()
    logger.error(
        f"Error: {str(error)}",
        exc_info=True,
        extra={"type": "error", "context": context or {}},
    )
