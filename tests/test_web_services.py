import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.web.schemas import SearchResultView
from src.web.services import (
    WebServiceError,
    build_index,
    format_evaluation_table,
    get_config_status,
    load_dataset_from_path,
    normalize_search_results,
    parse_k_values,
    prepare_sample_data,
    process_real_directory,
    redact_secrets,
    run_search,
    validate_hybrid_weights,
    validate_index_name,
    validate_positive_int,
)


class FakeBaseDB:
    def __init__(self, name, config):
        self.name = name
        self.loaded_dataset = None
        self.db_path = "data/vector_dbs/demo/vector_db.pkl"

    def load_data(self, dataset):
        self.loaded_dataset = dataset

    def get_stats(self):
        return {
            "name": self.name,
            "num_embeddings": 2,
            "embedding_dim": 2048,
            "cache_size": 0,
            "db_path": "data/vector_dbs/demo/vector_db.pkl",
        }


class FakeContextualDB(FakeBaseDB):
    def load_data(self, dataset, parallel_threads=None):
        self.loaded_dataset = dataset
        self.parallel_threads = parallel_threads

    def get_token_stats(self):
        return {"input_tokens": 100, "output_tokens": 10, "total_input_tokens": 100}


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


class FakeDocumentLoader:
    def __init__(self, config):
        self.config = config

    def process_directory(self, dir_path, num_per_doc=3):
        return (
            [{"doc_id": "real_doc", "chunks": [{"content": "real chunk"}]}],
            [{"query": "test", "golden_documents": []}],
        )

    def save_dataset(self, dataset, name):
        return f"data/{name}_dataset.json"

    def save_queries(self, queries, name):
        return f"data/{name}_queries.jsonl"


def test_process_real_directory_calls_loader(tmp_path):
    data_dir = tmp_path / "real_docs"
    data_dir.mkdir()

    result = process_real_directory(
        config=SimpleNamespace(),
        data_dir=str(data_dir),
        name="real_eval",
        queries_per_doc=3,
        loader_cls=FakeDocumentLoader,
    )

    assert result["documents"] == 1
    assert result["chunks"] == 1
    assert result["queries"] == 1
    assert result["dataset_path"] == "data/real_eval_dataset.json"


def test_process_real_directory_rejects_missing_dir(tmp_path):
    with pytest.raises(WebServiceError, match="文档目录不存在"):
        process_real_directory(
            config=SimpleNamespace(),
            data_dir=str(tmp_path / "missing"),
            name="test",
            queries_per_doc=3,
            loader_cls=FakeDocumentLoader,
        )


def test_load_dataset_from_path_rejects_invalid_json(tmp_path):
    dataset_path = tmp_path / "bad.json"
    dataset_path.write_text("not valid json", encoding="utf-8")

    with pytest.raises(WebServiceError, match="JSON 无法解析"):
        load_dataset_from_path(str(dataset_path))


def test_load_dataset_from_path_rejects_non_list_json(tmp_path):
    dataset_path = tmp_path / "obj.json"
    dataset_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    with pytest.raises(WebServiceError, match="必须是文档列表"):
        load_dataset_from_path(str(dataset_path))



def test_build_index_returns_base_summary(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps([{"doc_id": "doc_1", "chunks": [{"content": "x"}]}]), encoding="utf-8")

    summary = build_index(
        config=SimpleNamespace(),
        name="demo_base",
        method="base",
        dataset_path=str(dataset_path),
        parallel_threads=5,
        base_db_cls=FakeBaseDB,
        contextual_db_cls=FakeContextualDB,
    )

    assert summary.name == "demo_base"
    assert summary.method == "base"
    assert summary.num_embeddings == 2
    assert summary.loaded_from_disk is False
    assert summary.token_stats is None


def test_build_index_returns_contextual_token_summary(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps([{"doc_id": "doc_1", "chunks": [{"content": "x"}]}]), encoding="utf-8")

    summary = build_index(
        config=SimpleNamespace(),
        name="demo_contextual",
        method="contextual",
        dataset_path=str(dataset_path),
        parallel_threads=3,
        base_db_cls=FakeBaseDB,
        contextual_db_cls=FakeContextualDB,
    )

    assert summary.name == "demo_contextual"
    assert summary.method == "contextual"
    assert summary.loaded_from_disk is False
    assert summary.token_stats["input_tokens"] == 100


def test_normalize_base_search_result():
    results = normalize_search_results([
        {
            "metadata": {
                "doc_id": "doc_1",
                "chunk_id": "chunk_1",
                "content": "hello",
            },
            "similarity": 0.91,
        }
    ])

    assert results[0].rank == 1
    assert results[0].doc_id == "doc_1"
    assert results[0].chunk_id == "chunk_1"
    assert results[0].content == "hello"
    assert results[0].similarity == 0.91


def test_normalize_hybrid_and_reranked_result():
    results = normalize_search_results([
        {
            "chunk": {
                "doc_id": "doc_2",
                "chunk_id": "chunk_2",
                "original_content": "hybrid",
                "contextualized_content": "context",
            },
            "score": 0.5,
            "rerank_score": 0.99,
            "from_semantic": True,
            "from_bm25": False,
        }
    ])

    assert results[0].doc_id == "doc_2"
    assert results[0].content == "hybrid"
    assert results[0].contextualized_content == "context"
    assert results[0].score == 0.5
    assert results[0].rerank_score == 0.99
    assert results[0].from_semantic is True


class FakeSearchDB:
    def __init__(self, name, config):
        self.name = name

    def load_db(self):
        self.loaded = True

    def search(self, query, k=20):
        return [
            {
                "metadata": {"doc_id": "doc_1", "chunk_id": "chunk_1", "content": query},
                "similarity": 0.8,
            }
        ][:k]


class FakeReranker:
    def __init__(self, config=None):
        self.config = config

    def rerank_with_over_retrieval(self, query, vector_db, k=10, recall_multiplier=10):
        return [
            {
                "metadata": {"doc_id": "doc_r", "chunk_id": "chunk_r", "content": query},
                "similarity": 0.7,
                "rerank_score": 0.99,
            }
        ]


def test_format_evaluation_table_creates_rows_and_report():
    table = format_evaluation_table(
        method_name="contextual",
        results={
            5: {
                "pass_at_k": 0.8,
                "precision": 0.4,
                "recall": 0.6,
                "mrr": 0.7,
                "valid_queries": 3,
            }
        },
        report="report text",
    )

    assert table.method_name == "contextual"
    assert table.rows[0]["k"] == 5
    assert table.rows[0]["pass_at_k"] == 0.8
    assert table.report == "report text"


def test_run_search_uses_reranker_when_enabled():
    results = run_search(
        config=SimpleNamespace(),
        query="auth",
        index_name="demo",
        method="contextual",
        k=1,
        semantic_weight=0.8,
        bm25_weight=0.2,
        rerank=True,
        recall_multiplier=5,
        base_db_cls=FakeSearchDB,
        contextual_db_cls=FakeSearchDB,
        reranker_cls=FakeReranker,
    )

    assert results[0].doc_id == "doc_r"
    assert results[0].rerank_score == 0.99
