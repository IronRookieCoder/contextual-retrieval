"""
配置管理模块

提供单例模式的配置管理，支持从环境变量和 .env 文件加载配置。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .utils import ConfigurationError


@dataclass
class Config:
    """
    配置类（单例模式）

    支持从环境变量和 .env 文件加载配置，并提供配置验证功能。

    环境变量:
        ANTHROPIC_API_KEY: Anthropic Claude API 密钥
        VOYAGE_API_KEY: Voyage AI API 密钥
        COHERE_API_KEY: Cohere API 密钥
        ELASTICSEARCH_URL: Elasticsearch 服务地址

    示例:
        >>> config = Config.from_env()
        >>> config.validate()
    """

    # API 密钥（必需）
    ANTHROPIC_API_KEY: str
    VOYAGE_API_KEY: str
    COHERE_API_KEY: str

    # 服务地址（可选）
    ELASTICSEARCH_URL: str = "http://localhost:9200"

    # 模型配置
    ANTHROPIC_MODEL: str = "claude-haiku-4-5"
    VOYAGE_MODEL: str = "voyage-2"
    COHERE_RERANK_MODEL: str = "rerank-english-v3.0"

    # 处理参数
    EMBEDDING_BATCH_SIZE: int = 128
    MAX_PARALLEL_THREADS: int = 5
    DEFAULT_K: int = 20

    # 混合搜索权重
    SEMANTIC_WEIGHT: float = 0.8
    BM25_WEIGHT: float = 0.2

    # 重排序参数
    RERANK_RECALL_MULTIPLIER: int = 10
    RERANK_RATE_LIMIT_DELAY: float = 0.1

    # 目录配置
    DATA_DIR: Path = field(default_factory=lambda: Path("./data"))
    VECTOR_DB_DIR: Path = field(default_factory=lambda: Path("./data/vector_dbs"))

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 重试配置
    MAX_RETRIES: int = 3
    RETRY_BASE_DELAY: float = 1.0
    RETRY_MAX_DELAY: float = 60.0

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """
        从环境变量创建配置实例

        参数:
            env_file: .env 文件路径（可选）

        返回:
            Config 实例

        示例:
            >>> config = Config.from_env(".env.production")
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        return cls(
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""),
            VOYAGE_API_KEY=os.getenv("VOYAGE_API_KEY", ""),
            COHERE_API_KEY=os.getenv("COHERE_API_KEY", ""),
            ELASTICSEARCH_URL=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
            ANTHROPIC_MODEL=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            VOYAGE_MODEL=os.getenv("VOYAGE_MODEL", "voyage-2"),
            COHERE_RERANK_MODEL=os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
            EMBEDDING_BATCH_SIZE=int(os.getenv("EMBEDDING_BATCH_SIZE", "128")),
            MAX_PARALLEL_THREADS=int(os.getenv("MAX_PARALLEL_THREADS", "5")),
            DEFAULT_K=int(os.getenv("DEFAULT_K", "20")),
            SEMANTIC_WEIGHT=float(os.getenv("SEMANTIC_WEIGHT", "0.8")),
            BM25_WEIGHT=float(os.getenv("BM25_WEIGHT", "0.2")),
            RERANK_RECALL_MULTIPLIER=int(os.getenv("RERANK_RECALL_MULTIPLIER", "10")),
            RERANK_RATE_LIMIT_DELAY=float(os.getenv("RERANK_RATE_LIMIT_DELAY", "0.1")),
            DATA_DIR=Path(os.getenv("DATA_DIR", "./data")),
            VECTOR_DB_DIR=Path(os.getenv("VECTOR_DB_DIR", "./data/vector_dbs")),
            LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
            MAX_RETRIES=int(os.getenv("MAX_RETRIES", "3")),
            RETRY_BASE_DELAY=float(os.getenv("RETRY_BASE_DELAY", "1.0")),
            RETRY_MAX_DELAY=float(os.getenv("RETRY_MAX_DELAY", "60.0")),
        )

    def validate(self) -> bool:
        """
        验证配置完整性

        返回:
            bool: 验证是否通过

        抛出:
            ConfigurationError: 当必需的配置项缺失时
        """
        if not self.ANTHROPIC_API_KEY:
            raise ConfigurationError("ANTHROPIC_API_KEY is required")

        if not self.VOYAGE_API_KEY:
            raise ConfigurationError("VOYAGE_API_KEY is required")

        if not self.COHERE_API_KEY:
            raise ConfigurationError("COHERE_API_KEY is required")

        # 验证权重和为 1
        if abs(self.SEMANTIC_WEIGHT + self.BM25_WEIGHT - 1.0) > 0.01:
            raise ConfigurationError(
                f"权重和必须为 1.0，当前为 {self.SEMANTIC_WEIGHT + self.BM25_WEIGHT}"
            )

        # 验证参数范围
        if self.EMBEDDING_BATCH_SIZE <= 0:
            raise ConfigurationError("EMBEDDING_BATCH_SIZE must be positive")

        if self.MAX_PARALLEL_THREADS <= 0:
            raise ConfigurationError("MAX_PARALLEL_THREADS must be positive")

        if self.DEFAULT_K <= 0:
            raise ConfigurationError("DEFAULT_K must be positive")

        return True

    def ensure_directories(self) -> None:
        """创建必要的目录"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    def __str__(self) -> str:
        """返回配置的字符串表示（隐藏敏感信息）"""
        return f"""Config(
    ANTHROPIC_MODEL={self.ANTHROPIC_MODEL},
    VOYAGE_MODEL={self.VOYAGE_MODEL},
    COHERE_RERANK_MODEL={self.COHERE_RERANK_MODEL},
    ELASTICSEARCH_URL={self.ELASTICSEARCH_URL},
    EMBEDDING_BATCH_SIZE={self.EMBEDDING_BATCH_SIZE},
    MAX_PARALLEL_THREADS={self.MAX_PARALLEL_THREADS},
    DEFAULT_K={self.DEFAULT_K},
    SEMANTIC_WEIGHT={self.SEMANTIC_WEIGHT},
    BM25_WEIGHT={self.BM25_WEIGHT},
    DATA_DIR={self.DATA_DIR},
    VECTOR_DB_DIR={self.VECTOR_DB_DIR},
    LOG_LEVEL={self.LOG_LEVEL},
)"""
