"""
BM25 搜索模块

使用 Elasticsearch 实现 BM25 关键词搜索，支持多字段查询。
"""

from typing import Any, Dict, List

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from .config import Config
from .utils import DatabaseError, Logger, Timer, retry_with_backoff


class ElasticsearchBM25:
    """
    Elasticsearch BM25 搜索引擎

    提供高性能的关键词搜索功能，支持批量索引和多字段查询。

    参数:
        index_name: 索引名称
        config: 配置对象（可选）
        es_url: Elasticsearch URL（可选，默认使用配置值）

    示例:
        >>> es_bm25 = ElasticsearchBM25("my_index")
        >>> es_bm25.index_documents(documents)
        >>> results = es_bm25.search("查询文本", k=10)
    """

    def __init__(
        self,
        index_name: str = "contextual_bm25_index",
        config: Config = None,
        es_url: str = None
    ):
        self.config = config or Config.from_env()
        self.es_url = es_url or self.config.ELASTICSEARCH_URL
        self.index_name = index_name

        # 初始化 Elasticsearch 客户端
        self.es_client = Elasticsearch(self.es_url)

        # 日志
        self.logger = Logger(f"ElasticsearchBM25.{index_name}")

        # 创建索引
        self.create_index()

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def create_index(self) -> None:
        """
        创建 Elasticsearch 索引

        如果索引已存在，则跳过创建。
        """
        if self.es_client.indices.exists(index=self.index_name):
            self.logger.logger.info(f"索引已存在: {self.index_name}")
            return

        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False,
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "english"
                    },
                    "contextualized_content": {
                        "type": "text",
                        "analyzer": "english"
                    },
                    "doc_id": {
                        "type": "keyword",
                        "index": False
                    },
                    "chunk_id": {
                        "type": "keyword",
                        "index": False
                    },
                    "original_index": {
                        "type": "integer",
                        "index": False
                    },
                }
            },
        }

        self.es_client.indices.create(
            index=self.index_name,
            settings=index_settings["settings"],
            mappings=index_settings["mappings"],
        )

        self.logger.logger.info(f"创建索引: {self.index_name}")

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        批量索引文档

        参数:
            documents: 文档列表，每个文档应包含:
                - original_content: 原始内容
                - contextualized_content: 上下文描述
                - doc_id: 文档 ID
                - chunk_id: 块 ID
                - original_index: 原始索引

        返回:
            成功索引的文档数量

        抛出:
            DatabaseError: 索引失败时
        """
        # 准备批量操作
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "content": doc.get("original_content", ""),
                    "contextualized_content": doc.get("contextualized_content", ""),
                    "doc_id": doc.get("doc_id", ""),
                    "chunk_id": doc.get("chunk_id", ""),
                    "original_index": doc.get("original_index", 0),
                },
            }
            for doc in documents
        ]

        # 执行批量索引
        with Timer(f"索引 {len(actions)} 个文档"):
            success, failed = bulk(self.es_client, actions)

        # 刷新索引
        self.es_client.indices.refresh(index=self.index_name)

        if failed > 0:
            self.logger.logger.warning(
                f"批量索引完成，但 {failed} 个文档失败"
            )

        self.logger.logger.info(
            f"成功索引 {success} 个文档到 {self.index_name}"
        )

        return success

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def search(
        self,
        query: str,
        k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        搜索相关文档

        参数:
            query: 查询文本
            k: 返回结果数量

        返回:
            结果列表，每个结果包含:
                - doc_id: 文档 ID
                - original_index: 原始索引
                - content: 内容
                - contextualized_content: 上下文描述
                - score: BM25 分数

        抛出:
            DatabaseError: 搜索失败时
        """
        # 刷新索引以确保最新数据可搜索
        self.es_client.indices.refresh(index=self.index_name)

        # 执行多字段搜索
        response = self.es_client.search(
            index=self.index_name,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                    "type": "best_fields",
                }
            },
            size=k,
        )

        # 解析结果
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "doc_id": hit["_source"]["doc_id"],
                "original_index": hit["_source"]["original_index"],
                "content": hit["_source"]["content"],
                "contextualized_content": hit["_source"].get(
                    "contextualized_content", ""
                ),
                "score": hit["_score"],
            })

        self.logger.logger.debug(
            f"搜索 '{query[:50]}...' 返回 {len(results)} 个结果"
        )

        return results

    def delete_index(self) -> None:
        """
        删除索引

        警告: 此操作不可逆！
        """
        if self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.delete(index=self.index_name)
            self.logger.logger.info(f"已删除索引: {self.index_name}")
        else:
            self.logger.logger.warning(
                f"索引不存在，无法删除: {self.index_name}"
            )

    def get_index_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息

        返回:
            包含统计信息的字典
        """
        if not self.es_client.indices.exists(index=self.index_name):
            return {"exists": False}

        stats = self.es_client.indices.stats(index=self.index_name)

        return {
            "exists": True,
            "index_name": self.index_name,
            "document_count": stats["indices"][self.index_name]["total"]["docs"]["count"],
            "store_size": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
        }
