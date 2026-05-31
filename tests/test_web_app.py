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
