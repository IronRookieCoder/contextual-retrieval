"""
工具模块

提供日志管理、重试机制、性能计时器和自定义异常类。
"""

import functools
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar


# ==================== 异常类 ====================

class ContextualRetrievalError(Exception):
    """基础异常类"""
    pass


class ConfigurationError(ContextualRetrievalError):
    """配置错误"""
    pass


class APIError(ContextualRetrievalError):
    """API 调用错误"""
    pass


class DatabaseError(ContextualRetrievalError):
    """数据库错误"""
    pass


class ValidationError(ContextualRetrievalError):
    """数据验证错误"""
    pass


# ==================== 日志管理 ====================

class Logger:
    """
    统一日志管理器

    提供格式化的日志输出，支持不同日志级别。

    参数:
        name: 日志器名称
        level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        format_string: 日志格式字符串

    示例:
        >>> logger = Logger("my_app")
        >>> logger.logger.info("Application started")
    """

    def __init__(
        self,
        name: str,
        level: str = "INFO",
        format_string: Optional[str] = None
    ):
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # 避免重复添加 handler
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(getattr(logging, level.upper()))
            formatter = logging.Formatter(format_string)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_level(self, level: str) -> None:
        """设置日志级别"""
        self.logger.setLevel(getattr(logging, level.upper()))
        for handler in self.logger.handlers:
            handler.setLevel(getattr(logging, level.upper()))


# ==================== 重试机制 ====================

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple = (Exception,),
    logger: Optional[logging.Logger] = None,
) -> Callable:
    """
    指数退避重试装饰器

    参数:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        exceptions: 需要重试的异常类型元组
        logger: 日志记录器（可选）

    返回:
        装饰器函数

    示例:
        >>> @retry_with_backoff(max_retries=3, exceptions=(APIError,))
        ... def call_api():
        ...     # API 调用逻辑
        ...     pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        if logger:
                            logger.error(
                                f"函数 {func.__name__} 在 {max_retries} 次尝试后失败"
                            )
                        raise

                    # 计算延迟时间（指数退避）
                    delay = min(base_delay * (2 ** attempt), max_delay)

                    if logger:
                        logger.warning(
                            f"函数 {func.__name__} 第 {attempt + 1}/{max_retries} "
                            f"次尝试失败: {str(e)}，{delay:.2f}秒后重试"
                        )

                    time.sleep(delay)

            return None  # 理论上不会到达这里

        return wrapper

    return decorator


# ==================== 性能计时器 ====================

@contextmanager
def Timer(name: str = "操作", logger: Optional[logging.Logger] = None):
    """
    性能计时器上下文管理器

    参数:
        name: 计时器名称
        logger: 日志记录器（可选）

    示例:
        >>> with Timer("数据加载"):
        ...     load_data()
        数据加载 耗时: 1.23秒
    """
    start_time = time.time()

    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        message = f"{name} 耗时: {elapsed_time:.2f}秒"

        if logger:
            logger.info(message)
        else:
            print(message)


class TimerProfiler:
    """
    计时性能分析器

    用于累积统计多个操作的耗时。

    示例:
        >>> profiler = TimerProfiler()
        >>> with profiler.profile("操作1"):
        ...     do_something()
        >>> print(profiler.get_summary())
    """

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self._current_start: Optional[float] = None
        self._current_name: Optional[str] = None

    @contextmanager
    def profile(self, name: str):
        """计时上下文管理器"""
        self._current_name = name
        self._current_start = time.time()

        try:
            yield
        finally:
            if self._current_name and self._current_start is not None:
                elapsed = time.time() - self._current_start

                if self._current_name not in self.timings:
                    self.timings[self._current_name] = []

                self.timings[self._current_name].append(elapsed)
                self._current_name = None
                self._current_start = None

    def get_summary(self) -> str:
        """获取计时摘要"""
        if not self.timings:
            return "无计时数据"

        lines = ["计时摘要:", "-" * 60]

        for name, times in sorted(self.timings.items()):
            count = len(times)
            total = sum(times)
            avg = total / count if count > 0 else 0

            lines.append(
                f"{name:30s} "
                f"调用次数: {count:4d}, "
                f"总耗时: {total:8.2f}s, "
                f"平均: {avg:8.2f}s"
            )

        lines.append("-" * 60)
        return "\n".join(lines)

    def reset(self) -> None:
        """重置计时数据"""
        self.timings.clear()


# ==================== 其他工具函数 ====================

def chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    将列表分割成指定大小的块

    参数:
        lst: 输入列表
        chunk_size: 块大小

    返回:
        包含子列表的列表

    示例:
        >>> chunks([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误

    参数:
        a: 被除数
        b: 除数
        default: 除数为零时的默认值

    返回:
        除法结果或默认值

    示例:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0, default=0.0)
        0.0
    """
    return a / b if b != 0 else default
