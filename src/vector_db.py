"""
向量数据库模块

提供基础的向量存储和相似度搜索功能，使用 Jina AI 生成嵌入。
"""

import json
import os
import pickle
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
import requests
from tqdm import tqdm

from .config import Config
from .utils import APIError, DatabaseError, Logger, Timer, retry_with_backoff


class VectorDB(ABC):
    """
    向量数据库抽象基类

    定义了向量数据库的通用接口，包括数据加载、搜索和持久化。
    """

    @abstractmethod
    def load_data(self, dataset: List[Dict[str, Any]]) -> None:
        """
        加载数据集并生成嵌入

        参数:
            dataset: 数据集列表，每个元素包含 doc_id, chunks 等字段
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """
        搜索最相似的文档

        参数:
            query: 查询文本
            k: 返回结果数量

        返回:
            包含 metadata 和 similarity 的结果列表
        """
        pass

    @abstractmethod
    def save_db(self) -> None:
        """保存数据库到磁盘"""
        pass

    @abstractmethod
    def load_db(self) -> None:
        """从磁盘加载数据库"""
        pass


class VectorDBImpl(VectorDB):
    """
    向量数据库实现类

    使用 Jina AI 生成嵌入，余弦相似度进行搜索，
    支持 LRU 查询缓存和 pickle 持久化。

    参数:
        name: 数据库名称
        config: 配置对象（可选）

    示例:
        >>> db = VectorDBImpl("my_db")
        >>> db.load_data(dataset)
        >>> results = db.search("查询文本", k=10)
    """

    # Jina AI Embeddings API endpoint
    JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(self, name: str, config: Config = None):
        self.name = name
        self.config = config or Config.from_env()

        # Jina API 认证头
        self._jina_headers = {
            "Authorization": f"Bearer {self.config.JINA_API_KEY}",
            "Content-Type": "application/json",
        }

        # 数据存储
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

        # 查询缓存（LRU）
        self.query_cache: Dict[str, np.ndarray] = {}

        # 数据库文件路径
        self.db_path = os.path.join(
            self.config.VECTOR_DB_DIR,
            self.name,
            "vector_db.pkl"
        )

        # 日志
        self.logger = Logger(f"VectorDB.{name}")


    def _call_jina_embeddings(
        self,
        texts: List[str],
        task: str = "retrieval.passage"
    ) -> List[np.ndarray]:
        """
        调用 Jina AI Embeddings API

        参数:
            texts: 待嵌入的文本列表
            task: 任务类型（retrieval.passage 或 retrieval.query）

        返回:
            嵌入向量列表

        抛出:
            APIError: API 调用失败时
        """
        payload = {
            "model": self.config.JINA_EMBEDDING_MODEL,
            "input": texts,
            "task": task,
        }

        response = requests.post(
            self.JINA_EMBEDDINGS_URL,
            headers=self._jina_headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            raise APIError(
                f"Jina Embeddings API 返回错误 {response.status_code}: "
                f"{response.text[:500]}"
            )

        data = response.json()
        # Jina 返回格式: {"data": [{"embedding": [...], "index": 0}, ...]}
        embeddings = [
            np.array(item["embedding"])
            for item in sorted(data["data"], key=lambda x: x["index"])
        ]
        return embeddings

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def load_data(self, dataset: List[Dict[str, Any]]) -> None:
        """
        加载数据集并生成嵌入

        如果数据已经加载过，则跳过。如果磁盘上有缓存文件，
        则从磁盘加载。否则，生成新的嵌入并保存。

        参数:
            dataset: 数据集列表

        抛出:
            DatabaseError: 数据加载失败时
        """
        # 检查是否已加载
        if self.embeddings and self.metadata:
            self.logger.logger.info("向量数据库已加载，跳过数据加载")
            return

        # 尝试从磁盘加载
        if os.path.exists(self.db_path):
            self.logger.logger.info("从磁盘加载向量数据库")
            self.load_db()
            return

        # 收集待嵌入的文本和元数据
        texts_to_embed: List[str] = []
        metadata: List[Dict[str, Any]] = []
        total_chunks = sum(len(doc["chunks"]) for doc in dataset)

        with tqdm(total=total_chunks, desc="收集文档块") as pbar:
            for doc in dataset:
                for chunk in doc["chunks"]:
                    texts_to_embed.append(chunk["content"])
                    metadata.append({
                        "doc_id": doc["doc_id"],
                        "original_uuid": doc.get("original_uuid", ""),
                        "chunk_id": chunk["chunk_id"],
                        "original_index": chunk["original_index"],
                        "content": chunk["content"],
                    })
                    pbar.update(1)

        # 生成嵌入
        self._embed_and_store(texts_to_embed, metadata)

        # 保存到磁盘
        self.save_db()

        self.logger.logger.info(
            f"向量数据库加载并保存完成。处理的总块数: {len(texts_to_embed)}"
        )

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def _embed_and_store(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        生成嵌入并存储

        参数:
            texts: 待嵌入的文本列表
            metadata: 对应的元数据列表
        """
        batch_size = self.config.EMBEDDING_BATCH_SIZE
        all_embeddings = []

        with tqdm(total=len(texts), desc="生成嵌入") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                with Timer(f"嵌入批次 {i // batch_size + 1}"):
                    batch_embeddings = self._call_jina_embeddings(
                        batch, task="retrieval.passage"
                    )

                all_embeddings.extend(batch_embeddings)
                pbar.update(len(batch))

        self.embeddings = all_embeddings
        self.metadata = metadata

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """
        搜索最相似的文档

        参数:
            query: 查询文本
            k: 返回结果数量

        返回:
            包含 metadata 和 similarity 的结果列表

        抛出:
            DatabaseError: 数据库未加载时
        """
        if not self.embeddings:
            raise DatabaseError("向量数据库未加载，请先调用 load_data()")

        # 检查查询缓存
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
            self.logger.logger.debug(f"查询缓存命中: {query[:50]}...")
        else:
            # 生成查询嵌入
            query_embedding = self._call_jina_embeddings(
                [query], task="retrieval.query"
            )[0]

            # 更新缓存（使用 LRU）
            if len(self.query_cache) >= 1000:
                # 移除最旧的缓存项
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]

            self.query_cache[query] = query_embedding

        # 计算余弦相似度（点积）
        similarities = np.dot(self.embeddings, query_embedding)

        # 获取 top-k 索引
        top_indices = np.argsort(similarities)[::-1][:k]

        # 构建结果
        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)

        return top_results

    def save_db(self) -> None:
        """保存数据库到磁盘"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # 准备数据
        data = {
            "embeddings": [
                emb.tolist() if isinstance(emb, np.ndarray) else emb
                for emb in self.embeddings
            ],
            "metadata": self.metadata,
            "query_cache": {
                query: emb.tolist() if isinstance(emb, np.ndarray) else emb
                for query, emb in self.query_cache.items()
            },
        }

        # 保存到 pickle 文件
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)

        self.logger.logger.info(f"向量数据库已保存到: {self.db_path}")

    def load_db(self) -> None:
        """
        从磁盘加载数据库

        抛出:
            DatabaseError: 文件不存在或加载失败时
        """
        if not os.path.exists(self.db_path):
            raise DatabaseError(
                f"向量数据库文件不存在: {self.db_path}。"
                "请使用 load_data() 创建新数据库。"
            )

        with open(self.db_path, "rb") as f:
            data = pickle.load(f)

        self.embeddings = [
            np.array(emb) if isinstance(emb, list) else emb
            for emb in data["embeddings"]
        ]
        self.metadata = data["metadata"]
        self.query_cache = {
            query: np.array(emb) if isinstance(emb, list) else emb
            for query, emb in data.get("query_cache", {}).items()
        }

        self.logger.logger.info(
            f"向量数据库已加载。嵌入数: {len(self.embeddings)}, "
            f"元数据数: {len(self.metadata)}"
        )

    def clear_cache(self) -> None:
        """清除查询缓存"""
        self.query_cache.clear()
        self.logger.logger.info("查询缓存已清除")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息

        返回:
            包含统计信息的字典
        """
        return {
            "name": self.name,
            "num_embeddings": len(self.embeddings),
            "embedding_dim": len(self.embeddings[0]) if self.embeddings else 0,
            "cache_size": len(self.query_cache),
            "db_path": self.db_path,
        }


# 向后兼容的别名
VectorDB = VectorDBImpl
