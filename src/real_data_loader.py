"""
真实文档加载器模块

从文件系统加载真实文档，支持 Markdown 感知的智能分块和基于内容的查询生成。
输出的数据集和查询格式与 DataGenerator 完全兼容，可直接用于 VectorDBImpl、
ContextualVectorDB 和 Evaluator。

用法:
    >>> from src.real_data_loader import DocumentLoader
    >>> loader = DocumentLoader()
    >>> documents = loader.load_documents("./docs")
    >>> for doc in documents:
    ...     doc["chunks"] = loader.chunk_document(doc)
    >>> chunks_per_doc = {d["doc_id"]: d["chunks"] for d in documents}
    >>> queries = loader.generate_queries(documents, chunks_per_doc)
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import Config
from .utils import Logger


class DocumentLoader:
    """
    文档加载器

    从文件系统加载真实文档，支持智能分块和基于内容的查询生成。
    输出的数据格式与 DataGenerator 兼容，可直接用于后续的索引和评估。

    参数:
        config: 配置对象（可选）
        include_ext: 要包含的文件扩展名集合
        exclude_files: 要排除的文件名集合
        min_chunk_size: 最小块大小（字符数）
        max_chunk_size: 最大块大小（字符数）

    示例:
        >>> loader = DocumentLoader()
        >>> docs = loader.load_documents("D:/code/my-project")
        >>> for doc in docs:
        ...     doc["chunks"] = loader.chunk_document(doc)
        >>> loader.save_dataset(docs, "my_eval")
    """

    # 默认包含的文件扩展名
    DEFAULT_INCLUDE_EXT: Set[str] = {".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml"}
    # 默认排除的文件
    DEFAULT_EXCLUDE_FILES: Set[str] = {"CLAUDE.md", "GEMINI.md", "AGENTS.md"}
    # 默认块大小限制（字符数）
    DEFAULT_MIN_CHUNK_SIZE: int = 100
    DEFAULT_MAX_CHUNK_SIZE: int = 1500

    # 查询模板
    QUERY_TEMPLATES: List[str] = [
        "什么是{topic}？",
        "How to implement {topic}?",
        "{topic} 的核心内容是什么？",
        "Explain {topic}",
        "{topic} 的关键发现",
        "What is {topic}?",
        "介绍一下 {topic}",
        "{topic} 的技术方案",
    ]

    def __init__(
        self,
        config: Optional[Config] = None,
        include_ext: Optional[Set[str]] = None,
        exclude_files: Optional[Set[str]] = None,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
    ):
        self.config = config or Config.from_env()

        self.include_ext = include_ext or self.DEFAULT_INCLUDE_EXT
        self.exclude_files = exclude_files or self.DEFAULT_EXCLUDE_FILES
        self.min_chunk_size = min_chunk_size or self.DEFAULT_MIN_CHUNK_SIZE
        self.max_chunk_size = max_chunk_size or self.DEFAULT_MAX_CHUNK_SIZE

        self.logger = Logger("DocumentLoader")
        self._rng = random.Random(42)  # 固定种子保证可重复性

    # ==================== 文档加载 ====================

    def load_documents(self, data_dir: str) -> List[Dict[str, Any]]:
        """
        从目录加载所有符合条件的文档

        递归遍历目录，读取文件内容，过滤扩展名和排除列表，
        返回与 DataGenerator.generate_dataset() 兼容的文档列表。

        参数:
            data_dir: 文档目录路径

        返回:
            文档列表，每个文档包含 doc_id, uuid, original_uuid, file_path,
            file_name, content, theme, topic 字段（chunks 需后续通过 chunk_document 添加）

        示例:
            >>> docs = loader.load_documents("./my_docs")
            >>> len(docs)
            15
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"文档目录不存在: {data_dir}")

        documents: List[Dict[str, Any]] = []
        doc_id = 0

        for root, dirs, files in os.walk(data_path):
            # 跳过隐藏目录和常见非源码目录
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".")
                and d not in ("venv", ".venv", "node_modules", "__pycache__",
                              "site-packages", "bin", "lib", "include")
            ]

            for fname in sorted(files):
                ext = Path(fname).suffix.lower()
                if ext not in self.include_ext:
                    continue
                if fname in self.exclude_files:
                    continue

                fpath = Path(root) / fname
                try:
                    content = fpath.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    self.logger.logger.warning(f"跳过不可读文件: {fpath}")
                    continue

                if len(content) < 50:  # 跳过过短文件
                    continue

                rel_path = str(fpath.relative_to(data_path)).replace("\\", "/")

                documents.append({
                    "doc_id": f"doc_{doc_id:04d}",
                    "uuid": f"uuid_{doc_id:04d}",
                    "original_uuid": f"uuid_{doc_id:04d}",
                    "file_path": rel_path,
                    "file_name": fname,
                    "content": content,
                    "theme": self._detect_theme(content, fname),
                    "topic": fname,
                })
                doc_id += 1

        self.logger.logger.info(f"从 {data_dir} 加载 {len(documents)} 个文档")
        return documents

    def _detect_theme(self, content: str, fname: str) -> str:
        """
        检测文档主题

        根据文件名和内容前 200 个字符的关键词，判断文档所属主题类别。

        参数:
            content: 文档内容
            fname: 文件名

        返回:
            主题类别字符串: evaluation / planning / template / analysis / improvement / framework / general
        """
        fname_lower = fname.lower()
        head = content[:200].lower()

        if any(k in fname_lower for k in ("eval", "评价", "评测")) or \
           any(k in head for k in ("评价", "评测")):
            return "evaluation"
        if any(k in fname_lower for k in ("roadmap", "规划")) or \
           any(k in head for k in ("规划", "roadmap")):
            return "planning"
        if any(k in fname_lower for k in ("template", "模板")) or \
           any(k in head for k in ("模板", "template")):
            return "template"
        if any(k in fname_lower for k in ("report", "洞察", "报告")) or \
           any(k in head for k in ("洞察", "报告", "分析")):
            return "analysis"
        if "improvement" in fname_lower or "改进" in head:
            return "improvement"
        if "harness" in fname_lower or "框架" in head:
            return "framework"
        return "general"

    # ==================== 文本分块 ====================

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        对文档进行智能分块

        按 Markdown 二级标题 (##) 分割文档，对超长段落进一步按三级标题 (###)
        或空行分割。输出与 DataGenerator 兼容的块列表。

        参数:
            doc: 文档字典，需包含 content 字段

        返回:
            块列表，每块包含 chunk_id, index, original_index, content 字段

        示例:
            >>> chunks = loader.chunk_document(doc)
            >>> len(chunks)
            12
        """
        content = doc["content"]
        chunks: List[Dict[str, Any]] = []

        # 按 ## 标题分割（Markdown 二级标题）
        sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)

        for i, section in enumerate(sections):
            section = section.strip()
            if not section or len(section) < self.min_chunk_size:
                continue

            if len(section) > self.max_chunk_size:
                sub_chunks = self._split_large_section(section, i)
                chunks.extend(sub_chunks)
            else:
                chunks.append(self._make_chunk(section, i))

        return chunks

    def _split_large_section(
        self,
        section: str,
        section_idx: int
    ) -> List[Dict[str, Any]]:
        """
        将超长段落分割为多个块

        依次尝试：
        1. 按 ### 三级标题分割
        2. 按空行 (\n\n) 分割并合并小段落

        参数:
            section: 段落内容
            section_idx: 段落在文档中的索引

        返回:
            块列表
        """
        # 先尝试按 ### 三级标题分割
        subsections = re.split(r"(?=^### )", section, flags=re.MULTILINE)
        if len(subsections) > 1:
            result = []
            for j, sub in enumerate(subsections):
                sub = sub.strip()
                if len(sub) >= self.min_chunk_size:
                    result.append({
                        "chunk_id": f"chunk_{section_idx:04d}_{j:04d}",
                        "index": section_idx,
                        "original_index": section_idx,
                        "content": sub,
                    })
            if result:
                return result

        # 按空行分段
        paras = [p.strip() for p in re.split(r"\n\n+", section) if p.strip()]
        if len(paras) <= 2:
            return [self._make_chunk(section, section_idx)]

        # 合并小段落至 max_chunk_size 大小
        merged = []
        current = ""
        for p in paras:
            if len(current) + len(p) < self.max_chunk_size:
                current += "\n\n" + p if current else p
            else:
                if current and len(current) >= self.min_chunk_size:
                    merged.append(current)
                current = p
        if current and len(current) >= self.min_chunk_size:
            merged.append(current)

        return [self._make_chunk(m, section_idx) for m in merged]

    @staticmethod
    def _make_chunk(content: str, idx: int) -> Dict[str, Any]:
        """创建单个块"""
        return {
            "chunk_id": f"chunk_{idx:04d}",
            "index": idx,
            "original_index": idx,
            "content": content,
        }

    # ==================== 查询生成 ====================

    def generate_queries(
        self,
        documents: List[Dict[str, Any]],
        chunks_per_doc: Dict[str, List[Dict[str, Any]]],
        num_per_doc: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        基于文档真实内容生成评估查询

        为每个文档生成最多 num_per_doc 个查询，查询主题从块的第一行
        （通常是 Markdown 标题）提取，模板随机选择以增加多样性。

        参数:
            documents: 文档列表（需包含 doc_id, uuid, topic, chunks 等字段）
            chunks_per_doc: 文档 ID 到块列表的映射
            num_per_doc: 每个文档生成的查询数（默认 3）

        返回:
            查询列表，每项包含 query, golden_chunk_uuids, golden_documents,
            golden_chunks 字段，与 DataGenerator.generate_queries() 格式兼容

        示例:
            >>> chunks_map = {d["doc_id"]: d["chunks"] for d in docs}
            >>> queries = loader.generate_queries(docs, chunks_map, num_per_doc=2)
            >>> len(queries)
            26
        """
        queries: List[Dict[str, Any]] = []

        for doc in documents:
            doc_chunks = chunks_per_doc.get(doc["doc_id"], [])
            if not doc_chunks:
                continue

            num_queries = min(len(doc_chunks), num_per_doc)
            topic = doc.get("topic", "").replace(".md", "").replace(".py", "")

            for i in range(num_queries):
                chunk = doc_chunks[i % len(doc_chunks)]

                # 从块内容的第一行提取具体主题（通常是标题）
                first_line = chunk["content"].strip().split("\n")[0]
                first_line = re.sub(r"^#+\s*", "", first_line).strip()

                specific_topic = (
                    first_line
                    if first_line and len(first_line) < 80 and first_line != topic
                    else topic
                )

                template = self._rng.choice(self.QUERY_TEMPLATES)
                query = template.format(topic=specific_topic)

                queries.append({
                    "query": query,
                    "golden_chunk_uuids": [(doc["uuid"], chunk["original_index"])],
                    "golden_documents": [doc],
                    "golden_chunks": [chunk],
                })

        self.logger.logger.info(f"生成 {len(queries)} 个查询")
        return queries

    # ==================== 持久化 ====================

    def save_dataset(
        self,
        dataset: List[Dict[str, Any]],
        name: str = "real_eval",
    ) -> str:
        """
        保存数据集到 JSON 文件

        参数:
            dataset: 数据集列表
            name: 数据集名称（用于文件名）

        返回:
            保存的文件路径

        示例:
            >>> path = loader.save_dataset(dataset, "my_eval")
            >>> print(path)
            data/real_eval/my_eval_dataset.json
        """
        output_dir = self.config.DATA_DIR / "real_eval"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{name}_dataset.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        self.logger.logger.info(f"数据集已保存到: {output_path}")
        return str(output_path)

    def save_queries(
        self,
        queries: List[Dict[str, Any]],
        name: str = "real_eval",
    ) -> str:
        """
        保存查询集到 JSONL 文件

        参数:
            queries: 查询列表
            name: 查询集名称（用于文件名）

        返回:
            保存的文件路径

        示例:
            >>> path = loader.save_queries(queries, "my_eval")
            >>> print(path)
            data/real_eval/my_eval_queries.jsonl
        """
        output_dir = self.config.DATA_DIR / "real_eval"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{name}_queries.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for q in queries:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        self.logger.logger.info(f"查询集已保存到: {output_path}")
        return str(output_path)

    def load_dataset(self, name: str = "real_eval") -> List[Dict[str, Any]]:
        """
        从文件加载数据集

        参数:
            name: 数据集名称

        返回:
            数据集列表
        """
        input_path = self.config.DATA_DIR / "real_eval" / f"{name}_dataset.json"
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        self.logger.logger.info(f"从 {input_path} 加载数据集: {len(dataset)} 个文档")
        return dataset

    def load_queries(self, name: str = "real_eval") -> List[Dict[str, Any]]:
        """
        从文件加载查询集

        参数:
            name: 查询集名称

        返回:
            查询列表
        """
        input_path = self.config.DATA_DIR / "real_eval" / f"{name}_queries.jsonl"
        queries = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))
        self.logger.logger.info(f"从 {input_path} 加载查询集: {len(queries)} 个查询")
        return queries

    # ==================== 工具方法 ====================

    def process_directory(
        self,
        data_dir: str,
        num_per_doc: int = 3,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        一站式处理目录：加载 → 分块 → 生成查询

        这是 load_documents() + chunk_document() + generate_queries() 的便捷组合调用。

        参数:
            data_dir: 文档目录路径
            num_per_doc: 每文档查询数

        返回:
            (dataset, queries) 元组

        示例:
            >>> dataset, queries = loader.process_directory("./my_docs")
            >>> print(f"{len(dataset)} 个文档, {len(queries)} 个查询")
        """
        documents = self.load_documents(data_dir)
        chunks_per_doc: Dict[str, List[Dict[str, Any]]] = {}

        dataset = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            if chunks:
                doc["chunks"] = chunks
                chunks_per_doc[doc["doc_id"]] = chunks
                dataset.append(doc)

        queries = self.generate_queries(dataset, chunks_per_doc, num_per_doc)

        total_chunks = sum(len(c) for c in chunks_per_doc.values())
        total_chars = sum(len(doc["content"]) for doc in documents)
        self.logger.logger.info(
            f"目录处理完成: {len(dataset)} 个文档, "
            f"{total_chunks} 个块, "
            f"{total_chars:,} 字符, "
            f"{len(queries)} 个查询"
        )

        return dataset, queries
