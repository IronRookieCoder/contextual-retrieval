import re
from typing import Iterable, List, Tuple


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
