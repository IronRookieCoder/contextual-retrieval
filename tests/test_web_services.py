import pytest

from src.web.services import (
    WebServiceError,
    parse_k_values,
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
