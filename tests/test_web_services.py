import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.web.services import (
    WebServiceError,
    get_config_status,
    load_dataset_from_path,
    parse_k_values,
    prepare_sample_data,
    redact_secrets,
    validate_hybrid_weights,
    validate_index_name,
    validate_positive_int,
)


def test_validate_index_name_accepts_safe_names():
    assert validate_index_name("demo_contextual-01") == "demo_contextual-01"


@pytest.mark.parametrize("name", ["", "../bad", "bad/name", "bad name", "中文"])
def test_validate_index_name_rejects_unsafe_names(name):
    with pytest.raises(WebServiceError, match="索引名"):
        validate_index_name(name)


def test_validate_hybrid_weights_requires_sum_to_one():
    assert validate_hybrid_weights(0.8, 0.2) == (0.8, 0.2)

    with pytest.raises(WebServiceError, match="权重"):
        validate_hybrid_weights(0.8, 0.3)


def test_parse_k_values_accepts_space_or_comma_separated_values():
    assert parse_k_values("5 10,20") == [5, 10, 20]


def test_parse_k_values_rejects_non_positive_values():
    with pytest.raises(WebServiceError, match="k"):
        parse_k_values("5 0")


def test_validate_positive_int():
    assert validate_positive_int("5", "线程数") == 5

    with pytest.raises(WebServiceError, match="线程数"):
        validate_positive_int("0", "线程数")


def test_redact_secrets_replaces_secret_values():
    message = "request failed with sk-real-secret and jina-real-secret"

    redacted = redact_secrets(message, ["sk-real-secret", "jina-real-secret", ""])

    assert "sk-real-secret" not in redacted
    assert "jina-real-secret" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_overlapping_keys_longer_first():
    """Longer secrets must be redacted even when they contain shorter substrings."""
    redacted = redact_secrets("token sk-real-secret expired", ["sk-real", "sk-real-secret"])
    assert "sk-real-secret" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_empty_iterable():
    """Empty iterable should return the original message unchanged."""
    message = "some message with a token"
    redacted = redact_secrets(message, [])
    assert redacted == message


def test_redact_secrets_skips_short_secrets():
    """Secrets shorter than 4 characters must be ignored."""
    message = "a b c"
    redacted = redact_secrets(message, ["a", "ab", "abc"])
    assert redacted == "a b c"


class FakeESClient:
    def info(self):
        return {"cluster_name": "test"}


class BrokenESClient:
    def info(self):
        raise RuntimeError("connection refused")


def test_get_config_status_reports_available_services():
    config = SimpleNamespace(
        DEEPSEEK_API_KEY="sk-deepseek",
        JINA_API_KEY="jina-key",
        ELASTICSEARCH_URL="http://localhost:9200",
        DEEPSEEK_MODEL="deepseek-chat",
        DEEPSEEK_BASE_URL="https://api.deepseek.com",
        JINA_EMBEDDING_MODEL="jina-embeddings-v3",
        JINA_RERANKER_MODEL="jina-reranker-v2-base-multilingual",
        DATA_DIR=Path("data"),
        VECTOR_DB_DIR=Path("data/vector_dbs"),
    )

    status = get_config_status(config, es_client_factory=lambda url: FakeESClient())

    assert status.deepseek_configured is True
    assert status.jina_configured is True
    assert status.elasticsearch_available is True
    assert status.warnings == []


def test_get_config_status_reports_missing_keys_and_es_warning():
    config = SimpleNamespace(
        DEEPSEEK_API_KEY="",
        JINA_API_KEY="",
        ELASTICSEARCH_URL="http://localhost:9200",
        DEEPSEEK_MODEL="deepseek-chat",
        DEEPSEEK_BASE_URL="https://api.deepseek.com",
        JINA_EMBEDDING_MODEL="jina-embeddings-v3",
        JINA_RERANKER_MODEL="jina-reranker-v2-base-multilingual",
        DATA_DIR=Path("data"),
        VECTOR_DB_DIR=Path("data/vector_dbs"),
    )

    status = get_config_status(config, es_client_factory=lambda url: BrokenESClient())

    assert status.deepseek_configured is False
    assert status.jina_configured is False
    assert status.elasticsearch_available is False
    assert any("DEEPSEEK_API_KEY 未配置" in w for w in status.warnings)
    assert any("JINA_API_KEY 未配置" in w for w in status.warnings)
    assert any("Elasticsearch 不可用" in w for w in status.warnings)



class FakeGenerator:
    def __init__(self, config):
        self.config = config

    def generate_dataset(self, num_docs, chunks_per_doc):
        return [{"doc_id": "doc_0001", "content": "content", "chunks": [{"content": "chunk"}]}]

    def generate_queries(self, dataset, num_queries):
        return [{"query": "test", "golden_documents": dataset}]

    def save_dataset(self, dataset, output_path=None):
        return "data/sample_dataset.json"

    def save_queries(self, queries, output_path=None):
        return "data/sample_queries.jsonl"


def test_load_dataset_from_path_validates_existing_json(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps([{"doc_id": "doc_1", "chunks": []}]), encoding="utf-8")

    dataset = load_dataset_from_path(str(dataset_path))

    assert dataset == [{"doc_id": "doc_1", "chunks": []}]


def test_load_dataset_from_path_rejects_missing_file(tmp_path):
    with pytest.raises(WebServiceError, match="数据集文件不存在"):
        load_dataset_from_path(str(tmp_path / "missing.json"))


def test_prepare_sample_data_calls_generator():
    result = prepare_sample_data(
        config=SimpleNamespace(),
        num_docs=2,
        chunks_per_doc=3,
        num_queries=4,
        generator_cls=FakeGenerator,
    )

    assert result["dataset_path"] == "data/sample_dataset.json"
    assert result["queries_path"] == "data/sample_queries.jsonl"
    assert result["documents"] == 1
    assert result["queries"] == 1
