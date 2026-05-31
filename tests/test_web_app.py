from fastapi.testclient import TestClient

from src.web.app import create_app


def test_dashboard_renders():
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "Contextual Retrieval" in response.text
    assert "配置检查" in response.text
