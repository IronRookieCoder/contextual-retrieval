"""
上下文向量数据库模块

扩展基础向量数据库，使用 Claude 生成块级上下文描述，
通过提示缓存优化成本。
"""

import json
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import anthropic
import numpy as np
import voyageai
from tqdm import tqdm

from .config import Config
from .utils import Logger, Timer, retry_with_backoff
from .vector_db import VectorDBImpl


@dataclass
class TokenCounters:
    """
    Token 使用统计

    属性:
        input: 输入 token 总数
        output: 输出 token 总数
        cache_read: 从缓存读取的 token 数
        cache_creation: 写入缓存的 token 数
    """
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_creation: int = 0

    def total_input(self) -> int:
        """总输入 token 数（包括缓存）"""
        return self.input + self.cache_read + self.cache_creation

    def savings_percentage(self) -> float:
        """缓存节省百分比"""
        total = self.total_input()
        return (self.cache_read / total * 100) if total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"TokenCounters(input={self.input}, output={self.output}, "
            f"cache_read={self.cache_read}, cache_creation={self.cache_creation}, "
            f"savings={self.savings_percentage():.2f}%)"
        )


class ContextualVectorDB(VectorDBImpl):
    """
    上下文向量数据库

    使用 Claude 生成块级上下文描述，通过提示缓存优化成本。

    关键特性:
        - 提示缓存：60-80% 成本节省
        - 并行处理：多线程加速上下文生成
        - Token 统计：实时跟踪成本

    参数:
        name: 数据库名称
        config: 配置对象（可选）

    示例:
        >>> db = ContextualVectorDB("my_contextual_db")
        >>> db.load_data(dataset, parallel_threads=5)
        >>> print(db.token_counts)
        >>> results = db.search("查询文本", k=10)
    """

    # 上下文生成提示模板
    DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

    CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

    def __init__(self, name: str, config: Config = None):
        super().__init__(name, config)

        # 初始化 Anthropic 客户端
        self.anthropic_client = anthropic.Anthropic(
            api_key=self.config.ANTHROPIC_API_KEY
        )

        # Token 统计
        self.token_counts = TokenCounters()
        self.token_lock = threading.Lock()

        # 数据库文件路径
        self.db_path = os.path.join(
            self.config.VECTOR_DB_DIR,
            self.name,
            "contextual_vector_db.pkl"
        )

        # 更新日志名称
        self.logger = Logger(f"ContextualVectorDB.{name}")

    def situate_context(
        self,
        doc: str,
        chunk: str
    ) -> Tuple[str, anthropic.types.Usage]:
        """
        使用 Claude 生成块的上下文描述

        参数:
            doc: 完整文档内容
            chunk: 块内容

        返回:
            (上下文描述, token 使用情况)

        示例:
            >>> context, usage = db.situate_context(doc, chunk)
            >>> print(context)
            >>> print(usage)
        """
        response = self.anthropic_client.messages.create(
            model=self.config.ANTHROPIC_MODEL,
            max_tokens=1000,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": self.CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                    },
                ],
            }],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )

        return response.content[0].text, response.usage

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def load_data(
        self,
        dataset: List[Dict[str, Any]],
        parallel_threads: int = None
    ) -> None:
        """
        加载数据集并生成上下文嵌入

        参数:
            dataset: 数据集列表
            parallel_threads: 并行线程数（默认使用配置值）

        抛出:
            Exception: 数据加载失败时
        """
        # 检查是否已加载
        if self.embeddings and self.metadata:
            self.logger.logger.info("向量数据库已加载，跳过数据加载")
            return

        # 尝试从磁盘加载
        if os.path.exists(self.db_path):
            self.logger.logger.info("从磁盘加载上下文向量数据库")
            self.load_db()
            return

        # 设置并行线程数
        if parallel_threads is None:
            parallel_threads = self.config.MAX_PARALLEL_THREADS

        # 收集待嵌入的文本和元数据
        texts_to_embed: List[str] = []
        metadata: List[Dict[str, Any]] = []
        total_chunks = sum(len(doc["chunks"]) for doc in dataset)

        self.logger.logger.info(
            f"处理 {total_chunks} 个块，使用 {parallel_threads} 个线程"
        )

        def process_chunk(doc: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
            """处理单个块，生成上下文"""
            try:
                # 生成上下文
                contextualized_text, usage = self.situate_context(
                    doc["content"],
                    chunk["content"]
                )

                # 线程安全地更新 token 计数
                with self.token_lock:
                    self.token_counts.input += usage.input_tokens
                    self.token_counts.output += usage.output_tokens
                    self.token_counts.cache_read += usage.cache_read_input_tokens
                    self.token_counts.cache_creation += usage.cache_creation_input_tokens

                return {
                    "text_to_embed": f"{chunk['content']}\n\n{contextualized_text}",
                    "metadata": {
                        "doc_id": doc["doc_id"],
                        "original_uuid": doc.get("original_uuid", ""),
                        "chunk_id": chunk["chunk_id"],
                        "original_index": chunk["original_index"],
                        "original_content": chunk["content"],
                        "contextualized_content": contextualized_text,
                    },
                }
            except Exception as e:
                self.logger.logger.error(
                    f"处理块失败 (doc_id={doc.get('doc_id')}, "
                    f"chunk_id={chunk.get('chunk_id')}): {str(e)}"
                )
                # 降级：使用原始内容
                return {
                    "text_to_embed": chunk["content"],
                    "metadata": {
                        "doc_id": doc["doc_id"],
                        "original_uuid": doc.get("original_uuid", ""),
                        "chunk_id": chunk["chunk_id"],
                        "original_index": chunk["original_index"],
                        "original_content": chunk["content"],
                        "contextualized_content": "",
                    },
                }

        # 并行处理所有块
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []

            # 提交所有任务
            for doc in dataset:
                for chunk in doc["chunks"]:
                    futures.append(executor.submit(process_chunk, doc, chunk))

            # 收集结果
            for future in tqdm(
                as_completed(futures),
                total=total_chunks,
                desc="生成上下文"
            ):
                result = future.result()
                texts_to_embed.append(result["text_to_embed"])
                metadata.append(result["metadata"])

        # 生成嵌入
        self._embed_and_store(texts_to_embed, metadata)

        # 保存到磁盘
        self.save_db()

        # 打印统计信息
        total_tokens = self.token_counts.total_input()
        self.logger.logger.info(
            f"上下文向量数据库加载并保存完成。处理的总块数: {len(texts_to_embed)}"
        )
        self.logger.logger.info(f"总输入 token（不含缓存）: {self.token_counts.input}")
        self.logger.logger.info(f"总输出 token: {self.token_counts.output}")
        self.logger.logger.info(f"写入缓存的 token: {self.token_counts.cache_creation}")
        self.logger.logger.info(f"从缓存读取的 token: {self.token_counts.cache_read}")
        self.logger.logger.info(
            f"提示缓存节省: {self.token_counts.savings_percentage():.2f}% "
            "的输入 token 来自缓存"
        )
        self.logger.logger.info("从缓存读取的 token 享受 90% 折扣！")

    def get_token_stats(self) -> Dict[str, Any]:
        """
        获取 token 使用统计

        返回:
            包含详细统计信息的字典
        """
        return {
            "input_tokens": self.token_counts.input,
            "output_tokens": self.token_counts.output,
            "cache_read_tokens": self.token_counts.cache_read,
            "cache_creation_tokens": self.token_counts.cache_creation,
            "total_input_tokens": self.token_counts.total_input(),
            "cache_savings_percentage": self.token_counts.savings_percentage(),
        }

    def reset_token_counts(self) -> None:
        """重置 token 计数器"""
        self.token_counts = TokenCounters()
        self.logger.logger.info("Token 计数器已重置")
