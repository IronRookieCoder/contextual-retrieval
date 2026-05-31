from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.web.app import create_app


def test_dashboard_renders():
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "Contextual Retrieval" in response.text
    assert "配置检查" in response.text


def test_data_route_renders_validation_error():
    from src.web import app as web_app

    client = TestClient(web_app.create_app())

    response = client.post("/data", data={"mode": "unknown"})

    assert response.status_code == 200
    assert "数据模式必须是 sample 或 real" in response.text


def test_index_route_renders_validation_error(monkeypatch):
    from src.web import app as web_app

    monkeypatch.setattr(web_app, "get_config_status", lambda config: None)
    client = TestClient(web_app.create_app())

    response = client.post(
        "/index",
        data={
            "name": "../bad",
            "method": "base",
            "dataset_path": "missing.json",
            "parallel_threads": "5",
        },
    )

    assert response.status_code == 200
    assert "索引名只能包含" in response.text


def test_search_route_renders_mock_results(monkeypatch):
    from src.web import app as web_app

    monkeypatch.setattr(web_app, "get_config_status", lambda config: None)
    monkeypatch.setattr(
        web_app,
        "run_search",
        lambda **kwargs: [
            SimpleNamespace(
                rank=1,
                doc_id="doc_1",
                chunk_id="chunk_1",
                content="result content",
                contextualized_content="",
                similarity=0.9,
                score=None,
                rerank_score=None,
            )
        ],
    )
    client = TestClient(web_app.create_app())

    response = client.post(
        "/search",
        data={
            "query": "auth",
            "index_name": "demo",
            "method": "base",
            "k": "1",
            "semantic_weight": "0.8",
            "bm25_weight": "0.2",
            "recall_multiplier": "10",
        },
    )

    assert response.status_code == 200
    assert "result content" in response.text
