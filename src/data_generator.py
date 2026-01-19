"""
数据生成器模块

生成示例代码库文档和评估查询集，用于测试和演示。
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from .config import Config
from .utils import Logger


class DataGenerator:
    """
    数据生成器

    生成模拟的代码库文档和评估查询集，用于测试和演示。

    参数:
        config: 配置对象（可选）

    示例:
        >>> generator = DataGenerator()
        >>> dataset = generator.generate_dataset(num_docs=10)
        >>> queries = generator.generate_queries(dataset, num_queries=20)
    """

    # 代码主题模板
    CODE_TEMPLATES = {
        "authentication": [
            "用户认证系统实现",
            "JWT Token 生成和验证",
            "OAuth2 登录流程",
            "密码哈希和验证",
            "多因素认证",
        ],
        "database": [
            "数据库连接池管理",
            "ORM 映射配置",
            "数据库迁移脚本",
            "查询优化器",
            "事务处理",
        ],
        "api": [
            "RESTful API 设计",
            "GraphQL 端点实现",
            "API 版本控制",
            "速率限制中间件",
            "请求验证",
        ],
        "testing": [
            "单元测试框架",
            "集成测试套件",
            "Mock 对象生成",
            "测试数据工厂",
            "覆盖率报告",
        ],
        "security": [
            "XSS 防护",
            "CSRF 令牌验证",
            "SQL 注入防护",
            "加密解密工具",
            "安全审计日志",
        ],
        "performance": [
            "缓存策略实现",
            "异步任务队列",
            "连接池优化",
            "内存管理",
            "性能监控",
        ],
    }

    # 函数名称模板
    FUNCTION_NAMES = [
        "authenticate_user",
        "validate_token",
        "hash_password",
        "execute_query",
        "create_transaction",
        "handle_request",
        "parse_input",
        "format_response",
        "log_event",
        "cache_result",
    ]

    def __init__(self, config: Config = None):
        self.config = config or Config.from_env()
        self.logger = Logger("DataGenerator")
        self.random = random.Random(42)  # 固定种子以保证可重复性

    def generate_dataset(
        self,
        num_docs: int = 10,
        chunks_per_doc: int = 5,
        chunk_size: int = 500
    ) -> List[Dict[str, Any]]:
        """
        生成模拟数据集

        参数:
            num_docs: 文档数量
            chunks_per_doc: 每个文档的块数
            chunk_size: 每个块的字符数

        返回:
            数据集列表

        示例:
            >>> dataset = generator.generate_dataset(num_docs=10, chunks_per_doc=5)
            >>> len(dataset)
            10
        """
        dataset = []

        for doc_id in range(num_docs):
            # 随机选择主题
            theme = self.random.choice(list(self.CODE_TEMPLATES.keys()))
            topics = self.CODE_TEMPLATES[theme]
            topic = self.random.choice(topics)

            # 生成文档内容
            doc_content = self._generate_document_content(
                theme, topic, chunks_per_doc, chunk_size
            )

            # 分块
            chunks = self._split_into_chunks(
                doc_content, chunks_per_doc, chunk_size
            )

            doc = {
                "doc_id": f"doc_{doc_id:04d}",
                "original_uuid": f"uuid_{doc_id:04d}",
                "content": doc_content,
                "theme": theme,
                "topic": topic,
                "chunks": chunks,
            }

            dataset.append(doc)

        self.logger.logger.info(
            f"生成数据集: {num_docs} 个文档, "
            f"{sum(len(doc['chunks']) for doc in dataset)} 个块"
        )

        return dataset

    def _generate_document_content(
        self,
        theme: str,
        topic: str,
        num_chunks: int,
        chunk_size: int
    ) -> str:
        """生成单个文档的内容"""
        sections = []

        # 文档头部
        header = f"# {topic}\n\n"
        header += f"本文档描述了 {topic} 的实现。\n\n"
        sections.append(header)

        # 生成多个代码块
        for i in range(num_chunks):
            section = self._generate_code_section(theme, topic, i)
            sections.append(section)

        return "\n".join(sections)

    def _generate_code_section(
        self,
        theme: str,
        topic: str,
        index: int
    ) -> str:
        """生成代码块"""
        function_name = self.random.choice(self.FUNCTION_NAMES)
        section = f"## {function_name}_{index}\n\n"

        # 添加注释
        section += f"/**\n * {function_name} 函数实现\n"
        section += f" * 这是 {topic} 的第 {index + 1} 个部分\n"
        section += f" * 包含核心逻辑和错误处理\n */\n\n"

        # 生成伪代码
        section += f"def {function_name}_{index}(request):\n"
        section += f"    # 验证输入参数\n"
        section += f"    if not validate(request):\n"
        section += f"        raise ValidationError('Invalid input')\n\n"

        section += f"    # 处理核心逻辑\n"
        section += f"    result = process_{theme}(request)\n\n"

        section += f"    # 返回结果\n"
        section += f"    return format_response(result)\n"

        return section

    def _split_into_chunks(
        self,
        content: str,
        num_chunks: int,
        chunk_size: int
    ) -> List[Dict[str, Any]]:
        """将内容分割成块"""
        chunks = []
        lines = content.split("\n")

        current_chunk = []
        current_size = 0
        chunk_index = 0

        for line in lines:
            line_size = len(line)

            if current_size + line_size > chunk_size and current_chunk:
                # 保存当前块
                chunk_content = "\n".join(current_chunk)
                chunks.append({
                    "chunk_id": f"chunk_{chunk_index:04d}",
                    "original_index": chunk_index,
                    "content": chunk_content,
                })
                chunk_index += 1

                # 开始新块
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # 添加最后一个块
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            chunks.append({
                "chunk_id": f"chunk_{chunk_index:04d}",
                "original_index": chunk_index,
                "content": chunk_content,
            })

        return chunks

    def generate_queries(
        self,
        dataset: List[Dict[str, Any]],
        num_queries: int = 20
    ) -> List[Dict[str, Any]]:
        """
        生成评估查询集

        参数:
            dataset: 数据集
            num_queries: 查询数量

        返回:
            查询列表

        示例:
            >>> queries = generator.generate_queries(dataset, num_queries=20)
            >>> len(queries)
            20
        """
        queries = []

        for i in range(num_queries):
            # 随机选择一个文档和块作为黄金标准
            doc = self.random.choice(dataset)
            chunk = self.random.choice(doc["chunks"])

            # 生成查询
            query = self._generate_query(doc, chunk)

            query_item = {
                "query": query,
                "golden_chunk_uuids": [(doc["original_uuid"], chunk["original_index"])],
                "golden_documents": [doc],
                "golden_chunks": [chunk],
            }

            queries.append(query_item)

        self.logger.logger.info(f"生成查询集: {num_queries} 个查询")

        return queries

    def _generate_query(
        self,
        doc: Dict[str, Any],
        chunk: Dict[str, Any]
    ) -> str:
        """基于文档和块生成查询"""
        query_templates = [
            f"How to implement {doc['topic']}?",
            f"实现 {doc['topic']} 的方法",
            f"What is the purpose of {doc['topic']}?",
            f"{doc['topic']} 的核心逻辑是什么？",
            f"Explain the {doc['theme']} implementation",
            f"如何使用 {doc['theme']} 相关功能？",
            f"Show me the code for {doc['topic']}",
            f"{doc['topic']} 的代码示例",
        ]

        return self.random.choice(query_templates)

    def save_dataset(
        self,
        dataset: List[Dict[str, Any]],
        output_path: str = None
    ) -> str:
        """
        保存数据集到文件

        参数:
            dataset: 数据集
            output_path: 输出路径（可选）

        返回:
            保存的文件路径
        """
        if output_path is None:
            output_path = os.path.join(
                self.config.DATA_DIR,
                "sample_dataset.json"
            )

        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存到 JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        self.logger.logger.info(f"数据集已保存到: {output_path}")

        return output_path

    def save_queries(
        self,
        queries: List[Dict[str, Any]],
        output_path: str = None
    ) -> str:
        """
        保存查询集到 JSONL 文件

        参数:
            queries: 查询列表
            output_path: 输出路径（可选）

        返回:
            保存的文件路径
        """
        if output_path is None:
            output_path = os.path.join(
                self.config.DATA_DIR,
                "sample_queries.jsonl"
            )

        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存到 JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for query in queries:
                json.dump(query, f, ensure_ascii=False)
                f.write("\n")

        self.logger.logger.info(f"查询集已保存到: {output_path}")

        return output_path

    def load_dataset(self, input_path: str) -> List[Dict[str, Any]]:
        """
        从文件加载数据集

        参数:
            input_path: 输入路径

        返回:
            数据集列表
        """
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        self.logger.logger.info(
            f"从 {input_path} 加载数据集: {len(dataset)} 个文档"
        )

        return dataset

    def load_queries(self, input_path: str) -> List[Dict[str, Any]]:
        """
        从文件加载查询集

        参数:
            input_path: 输入路径

        返回:
            查询列表
        """
        queries = []

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))

        self.logger.logger.info(
            f"从 {input_path} 加载查询集: {len(queries)} 个查询"
        )

        return queries
