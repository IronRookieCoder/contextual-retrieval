"""
重排序器模块

使用 Jina AI Rerank API 对搜索结果进行精细排序。
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import requests

from .config import Config
from .utils import APIError, Logger, retry_with_backoff


class Reranker(ABC):
    """重排序器抽象基类"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        对文档进行重排序

        参数:
            query: 查询文本
            documents: 文档列表
            top_n: 返回的 top-n 结果数量

        返回:
            重排序后的结果列表
        """
        pass


class JinaReranker(Reranker):
    """
    Jina AI 重排序器

    使用 Jina Rerank API 对搜索结果进行精细排序。

    参数:
        model: 模型名称（可选，默认使用配置值）
        rate_limit_delay: 速率限制延迟（秒）
        config: 配置对象（可选）

    示例:
        >>> reranker = JinaReranker()
        >>> results = reranker.rerank(query, documents, top_n=10)
        >>> for r in results:
        ...     print(f"索引: {r['index']}, 分数: {r['score']:.4f}")
    """

    # Jina AI Reranker API endpoint
    JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"

    def __init__(
        self,
        model: str = None,
        rate_limit_delay: float = None,
        config: Config = None,
    ):
        self.config = config or Config.from_env()
        self.model = model or self.config.JINA_RERANKER_MODEL
        self.rate_limit_delay = (
            rate_limit_delay or self.config.RERANK_RATE_LIMIT_DELAY
        )

        # Jina API 认证头
        self._jina_headers = {
            "Authorization": f"Bearer {self.config.JINA_API_KEY}",
            "Content-Type": "application/json",
        }

        # 日志
        self.logger = Logger(f"JinaReranker.{self.model}")

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        对文档进行重排序

        参数:
            query: 查询文本
            documents: 文档列表（字符串列表）
            top_n: 返回的 top-n 结果数量

        返回:
            重排序后的结果列表，每个结果包含:
                - index: 原始文档索引
                - score: 相关性分数
                - text: 文档文本（可选）

        抛出:
            APIError: API 调用失败时

        示例:
            >>> reranker = JinaReranker()
            >>> results = reranker.rerank(
            ...     "如何实现认证?",
            ...     ["文档1", "文档2", "文档3"],
            ...     top_n=2
            ... )
            >>> len(results)
            2
        """
        if not documents:
            return []

        if top_n > len(documents):
            self.logger.logger.warning(
                f"请求的 top_n ({top_n}) 大于文档数量 ({len(documents)})，"
                f"将使用所有文档进行重排序"
            )
            top_n = len(documents)

        # 调用 Jina Rerank API
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }

        response = requests.post(
            self.JINA_RERANK_URL,
            headers=self._jina_headers,
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            raise APIError(
                f"Jina Rerank API 返回错误 {response.status_code}: "
                f"{response.text[:500]}"
            )

        data = response.json()

        # 速率限制
        time.sleep(self.rate_limit_delay)

        # 解析结果: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        results = []
        for r in data["results"]:
            results.append({
                "index": r["index"],
                "score": r["relevance_score"],
                "text": documents[r["index"]] if r["index"] < len(documents) else None,
            })

        self.logger.logger.debug(
            f"重排序 '{query[:50]}...' 返回 {len(results)} 个结果"
        )

        return results

    def rerank_search_results(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        top_n: int = 10,
        text_field: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        对搜索结果进行重排序

        参数:
            query: 查询文本
            search_results: 搜索结果列表
            top_n: 返回的 top-n 结果数量
            text_field: 用于提取文本的字段名

        返回:
            重排序后的搜索结果列表

        示例:
            >>> results = db.search("查询", k=50)
            >>> reranked = reranker.rerank_search_results(
            ...     "查询",
            ...     results,
            ...     top_n=10
            ... )
        """
        # 提取文档文本
        documents = []
        for result in search_results:
            if "metadata" in result:
                # 向量数据库结果
                metadata = result["metadata"]
                text = metadata.get(text_field, metadata.get("content", ""))
            else:
                # 其他格式
                text = result.get(text_field, "")

            # 如果有上下文描述，则合并
            if "metadata" in result and "contextualized_content" in result["metadata"]:
                contextualized = result["metadata"]["contextualized_content"]
                if contextualized:
                    text = f"{text}\n\nContext: {contextualized}"

            documents.append(text)

        # 执行重排序
        rerank_results = self.rerank(query, documents, top_n=top_n)

        # 构建重排序后的结果
        final_results = []
        for r in rerank_results:
            original_result = search_results[r["index"]]

            # 创建新的结果对象
            reranked_result = {
                **original_result,
                "rerank_score": r["score"],
            }

            final_results.append(reranked_result)

        return final_results

    def rerank_with_over_retrieval(
        self,
        query: str,
        vector_db,
        k: int = 10,
        recall_multiplier: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        过检索 + 重排序

        参数:
            query: 查询文本
            vector_db: 向量数据库实例
            k: 最终返回结果数量
            recall_multiplier: 过检索倍数

        返回:
            重排序后的结果列表

        示例:
            >>> results = reranker.rerank_with_over_retrieval(
            ...     "查询文本",
            ...     vector_db,
            ...     k=10,
            ...     recall_multiplier=10
            ... )
        """
        # 过检索
        recall_k = k * recall_multiplier

        # 检查过检索数量
        if recall_k < k * 5:
            self.logger.logger.warning(
                f"过检索倍数 ({recall_multiplier}) 可能不足，"
                f"当前检索 {recall_k} 个候选，推荐至少 {k * 10} 个"
            )

        search_results = vector_db.search(query, k=recall_k)

        # 重排序
        reranked_results = self.rerank_search_results(
            query,
            search_results,
            top_n=k
        )

        return reranked_results


# 向后兼容的别名
CohereReranker = JinaReranker
