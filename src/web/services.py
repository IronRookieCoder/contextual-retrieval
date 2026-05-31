import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

from elasticsearch import Elasticsearch

from src.bm25_search import ElasticsearchBM25
from src.config import Config
from src.contextual_db import ContextualVectorDB
from src.data_generator import DataGenerator
from src.hybrid_search import HybridSearchEngine
from src.real_data_loader import DocumentLoader
from src.reranking import JinaReranker
from src.vector_db import VectorDBImpl
from src.evaluation import Evaluator
from .schemas import ConfigStatus, EvaluationTable, IndexSummary, SearchResultView


class WebServiceError(Exception):
    """Expected validation or workflow error that should render in the UI."""


SAFE_INDEX_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def redact_secrets(message: str, secrets: Iterable[str]) -> str:
    redacted = message
    sorted_secrets = sorted((s for s in secrets if s and len(s) >= 4), key=len, reverse=True)
    for secret in sorted_secrets:
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def validate_index_name(name: str) -> str:
    value = (name or "").strip()
    if not value or not SAFE_INDEX_RE.fullmatch(value):
        raise WebServiceError("索引名只能包含字母、数字、下划线和连字符。")
    return value


def validate_hybrid_weights(
    semantic_weight: float,
    bm25_weight: float,
) -> Tuple[float, float]:
    if semantic_weight <= 0 or bm25_weight <= 0:
        raise WebServiceError("混合搜索权重必须为正数。")
    if abs(semantic_weight + bm25_weight - 1.0) > 0.01:
        raise WebServiceError("混合搜索权重之和必须为 1.0。")
    return semantic_weight, bm25_weight


def validate_positive_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WebServiceError(f"{label}必须是正整数。") from exc
    if parsed <= 0:
        raise WebServiceError(f"{label}必须是正整数。")
    return parsed


def parse_k_values(raw: str) -> List[int]:
    parts = [part for part in re.split(r"[\s,]+", (raw or "").strip()) if part]
    if not parts:
        raise WebServiceError("k 值不能为空。")

    values: List[int] = []
    for part in parts:
        values.append(validate_positive_int(part, "k 值"))
    return values


def get_config_status(
    config: Config,
    es_client_factory: Optional[Callable[[str], object]] = None,
) -> ConfigStatus:
    warnings: List[str] = []
    deepseek_configured = bool(config.DEEPSEEK_API_KEY)
    jina_configured = bool(config.JINA_API_KEY)

    if not deepseek_configured:
        warnings.append("DEEPSEEK_API_KEY 未配置：上下文索引不可用。")
    if not jina_configured:
        warnings.append("JINA_API_KEY 未配置：向量索引、搜索和重排不可用。")

    factory = es_client_factory or (lambda url: Elasticsearch(url))
    elasticsearch_available = False
    client = None
    try:
        client = factory(config.ELASTICSEARCH_URL)
        client.info()
        elasticsearch_available = True
    except Exception:
        warnings.append("Elasticsearch 不可用：混合搜索和混合评估将被禁用。")
    finally:
        if client is not None and es_client_factory is None:
            client.close()

    return ConfigStatus(
        deepseek_configured=deepseek_configured,
        jina_configured=jina_configured,
        elasticsearch_url=config.ELASTICSEARCH_URL,
        elasticsearch_available=elasticsearch_available,
        deepseek_model=config.DEEPSEEK_MODEL,
        deepseek_base_url=config.DEEPSEEK_BASE_URL,
        jina_embedding_model=config.JINA_EMBEDDING_MODEL,
        jina_reranker_model=config.JINA_RERANKER_MODEL,
        data_dir=config.DATA_DIR,
        vector_db_dir=config.VECTOR_DB_DIR,
        warnings=warnings,
    )


def load_dataset_from_path(dataset_path: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists() or not path.is_file():
        raise WebServiceError(f"数据集文件不存在: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise WebServiceError(f"数据集 JSON 无法解析: {path}") from exc
    if not isinstance(data, list):
        raise WebServiceError("数据集 JSON 必须是文档列表。")
    return data


def prepare_sample_data(
    config,
    num_docs: int,
    chunks_per_doc: int,
    num_queries: int,
    generator_cls: Type[DataGenerator] = DataGenerator,
) -> Dict[str, Any]:
    generator = generator_cls(config)
    dataset = generator.generate_dataset(
        num_docs=num_docs,
        chunks_per_doc=chunks_per_doc,
    )
    queries = generator.generate_queries(dataset, num_queries=num_queries)
    dataset_path = generator.save_dataset(dataset)
    queries_path = generator.save_queries(queries)
    return {
        "dataset_path": dataset_path,
        "queries_path": queries_path,
        "documents": len(dataset),
        "chunks": sum(len(doc.get("chunks", [])) for doc in dataset),
        "queries": len(queries),
    }


def process_real_directory(
    config,
    data_dir: str,
    name: str,
    queries_per_doc: int,
    loader_cls: Type[DocumentLoader] = DocumentLoader,
) -> Dict[str, Any]:
    path = Path(data_dir)
    if not path.exists() or not path.is_dir():
        raise WebServiceError(f"文档目录不存在: {path}")
    safe_name = validate_index_name(name)
    loader = loader_cls(config)
    dataset, queries = loader.process_directory(str(path), num_per_doc=queries_per_doc)
    dataset_path = loader.save_dataset(dataset, safe_name)
    queries_path = loader.save_queries(queries, safe_name)
    return {
        "dataset_path": dataset_path,
        "queries_path": queries_path,
        "documents": len(dataset),
        "chunks": sum(len(doc.get("chunks", [])) for doc in dataset),
        "queries": len(queries),
    }


def build_index(
    config,
    name: str,
    method: str,
    dataset_path: str,
    parallel_threads: int,
    base_db_cls: Type[VectorDBImpl] = VectorDBImpl,
    contextual_db_cls: Type[ContextualVectorDB] = ContextualVectorDB,
) -> IndexSummary:
    safe_name = validate_index_name(name)
    if method not in {"base", "contextual"}:
        raise WebServiceError("索引方法必须是 base 或 contextual。")

    dataset = load_dataset_from_path(dataset_path)

    db = base_db_cls(safe_name, config) if method == "base" else contextual_db_cls(safe_name, config)
    existed_before = os.path.exists(db.db_path)
    if method == "contextual":
        db.load_data(dataset, parallel_threads=parallel_threads)
    else:
        db.load_data(dataset)

    stats = db.get_stats()
    token_stats = db.get_token_stats() if method == "contextual" else None
    return IndexSummary(
        name=stats["name"],
        method=method,
        num_embeddings=stats["num_embeddings"],
        embedding_dim=stats["embedding_dim"],
        cache_size=stats["cache_size"],
        db_path=stats["db_path"],
        loaded_from_disk=existed_before,
        token_stats=token_stats,
    )


def normalize_search_results(results: List[Dict[str, Any]]) -> List[SearchResultView]:
    normalized: List[SearchResultView] = []
    for rank, result in enumerate(results, start=1):
        if "metadata" in result:
            metadata = result["metadata"]
            content = metadata.get("original_content") or metadata.get("content", "")
            normalized.append(
                SearchResultView(
                    rank=rank,
                    doc_id=metadata.get("doc_id", ""),
                    chunk_id=metadata.get("chunk_id", str(metadata.get("original_index", ""))),
                    content=content,
                    contextualized_content=metadata.get("contextualized_content", ""),
                    similarity=result.get("similarity"),
                    rerank_score=result.get("rerank_score"),
                )
            )
            continue

        chunk = result.get("chunk", result)
        normalized.append(
            SearchResultView(
                rank=rank,
                doc_id=chunk.get("doc_id", ""),
                chunk_id=chunk.get("chunk_id", str(chunk.get("original_index", ""))),
                content=chunk.get("original_content") or chunk.get("content", ""),
                contextualized_content=chunk.get("contextualized_content", ""),
                score=result.get("score", chunk.get("score")),
                rerank_score=result.get("rerank_score"),
                from_semantic=bool(result.get("from_semantic", False)),
                from_bm25=bool(result.get("from_bm25", False)),
            )
        )
    return normalized


def run_search(
    config,
    query: str,
    index_name: str,
    method: str,
    k: int,
    semantic_weight: float,
    bm25_weight: float,
    rerank: bool,
    recall_multiplier: int,
    base_db_cls: Type[VectorDBImpl] = VectorDBImpl,
    contextual_db_cls: Type[ContextualVectorDB] = ContextualVectorDB,
    bm25_cls: Type[ElasticsearchBM25] = ElasticsearchBM25,
    hybrid_engine_cls: Type[HybridSearchEngine] = HybridSearchEngine,
    reranker_cls: Type[JinaReranker] = JinaReranker,
) -> List[SearchResultView]:
    safe_name = validate_index_name(index_name)
    if not query.strip():
        raise WebServiceError("查询不能为空。")
    if method not in {"base", "contextual", "hybrid"}:
        raise WebServiceError("搜索方法必须是 base、contextual 或 hybrid。")

    if method == "base":
        db = base_db_cls(safe_name, config)
    else:
        db = contextual_db_cls(safe_name, config)

    try:
        db.load_db()
    except Exception as exc:
        raise WebServiceError(f"无法加载索引 {safe_name}: {exc}") from exc

    if rerank and method in {"base", "contextual"}:
        results = reranker_cls(config=config).rerank_with_over_retrieval(
            query,
            db,
            k=k,
            recall_multiplier=recall_multiplier,
        )
        return normalize_search_results(results)

    if method == "hybrid":
        validate_hybrid_weights(semantic_weight, bm25_weight)
        bm25 = bm25_cls(f"{safe_name}_bm25_web", config)
        try:
            bm25.index_documents(db.metadata)
            engine = hybrid_engine_cls(
                db,
                bm25,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                config=config,
            )
            results = engine.search(query, k=k)
            if rerank:
                documents = []
                for item in results:
                    chunk = item.get("chunk", {})
                    documents.append(chunk.get("original_content") or chunk.get("content", ""))
                reranked = reranker_cls(config=config).rerank(query, documents, top_n=k)
                results = [{**results[item["index"]], "rerank_score": item["score"]} for item in reranked]
            return normalize_search_results(results)
        finally:
            try:
                bm25.delete_index()
            except Exception:
                pass

    results = db.search(query, k=k)
    return normalize_search_results(results)


def format_evaluation_table(
    method_name: str,
    results: Dict[int, Dict[str, Any]],
    report: str,
) -> EvaluationTable:
    rows = []
    for k in sorted(results.keys()):
        item = results[k]
        rows.append(
            {
                "k": k,
                "pass_at_k": item["pass_at_k"],
                "precision": item["precision"],
                "recall": item["recall"],
                "mrr": item["mrr"],
                "valid_queries": item["valid_queries"],
            }
        )
    return EvaluationTable(method_name=method_name, rows=rows, report=report)


def run_evaluation(
    config,
    index_name: str,
    method: str,
    queries_path: str,
    k_values: List[int],
    semantic_weight: float,
    bm25_weight: float,
    evaluator_cls: Type[Evaluator] = Evaluator,
    base_db_cls: Type[VectorDBImpl] = VectorDBImpl,
    contextual_db_cls: Type[ContextualVectorDB] = ContextualVectorDB,
    bm25_cls: Type[ElasticsearchBM25] = ElasticsearchBM25,
    hybrid_engine_cls: Type[HybridSearchEngine] = HybridSearchEngine,
) -> EvaluationTable:
    safe_name = validate_index_name(index_name)
    path = Path(queries_path)
    if not path.exists() or not path.is_file():
        raise WebServiceError(f"查询文件不存在: {path}")
    if method not in {"base", "contextual", "hybrid"}:
        raise WebServiceError("评估方法必须是 base、contextual 或 hybrid。")

    evaluator = evaluator_cls(config)
    queries = evaluator.load_queries(str(path))

    if method == "base":
        db = base_db_cls(safe_name, config)
        db.load_db()
        results = evaluator.evaluate(
            queries,
            lambda query, k: db.search(query, k=k),
            k_values=k_values,
            method_name=method,
        )
    elif method == "contextual":
        db = contextual_db_cls(safe_name, config)
        db.load_db()
        results = evaluator.evaluate(
            queries,
            lambda query, k: db.search(query, k=k),
            k_values=k_values,
            method_name=method,
        )
    else:
        validate_hybrid_weights(semantic_weight, bm25_weight)
        db = contextual_db_cls(safe_name, config)
        db.load_db()
        bm25 = bm25_cls(f"{safe_name}_bm25_eval_web", config)
        try:
            bm25.index_documents(db.metadata)
            engine = hybrid_engine_cls(
                db,
                bm25,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                config=config,
            )
            results = evaluator.evaluate_hybrid(
                queries,
                engine,
                k_values=k_values,
                method_name="hybrid",
            )
        finally:
            try:
                bm25.delete_index()
            except Exception:
                pass

    report = evaluator.generate_report(results, method)
    return format_evaluation_table(method, results, report)
