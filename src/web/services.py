import re
from typing import Callable, Iterable, List, Optional, Tuple

from elasticsearch import Elasticsearch

from .schemas import ConfigStatus


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
    config,
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
    try:
        factory(config.ELASTICSEARCH_URL).info()
        elasticsearch_available = True
    except Exception:
        warnings.append("Elasticsearch 不可用：混合搜索和混合评估将被禁用。")

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
