"""Автоматическая проверка API: python -m pytest -q test_api.py"""
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

EXAMPLE = {"city": "Москва", "total_square": 60.0, "rooms": 2, "floor": 5}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_get():
    response = client.get("/predict_get", params=EXAMPLE)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price_rub"] > 0
    assert data["currency"] == "RUB"
    assert data["features"] == EXAMPLE


def test_predict_post():
    response = client.post("/predict_post", json=EXAMPLE)
    assert response.status_code == 200
    assert response.json()["predicted_price_rub"] > 0


def test_get_and_post_same_prediction():
    response_get = client.get("/predict_get", params=EXAMPLE)
    response_post = client.post("/predict_post", json=EXAMPLE)
    assert response_get.json()["predicted_price_rub"] == response_post.json()["predicted_price_rub"]


def test_invalid_square_validation():
    response = client.get("/predict_get", params={**EXAMPLE, "total_square": -2})
    assert response.status_code == 422


def test_unknown_city_validation():
    response = client.post("/predict_post", json={**EXAMPLE, "city": "НесуществующийГород"})
    assert response.status_code == 422


def test_openapi_routes():
    schema = client.get("/openapi.json").json()
    for route in ("/health", "/predict_get", "/predict_post"):
        assert route in schema["paths"]
