"""The HTTP contract of POST /predict, using the tiny model from conftest."""
import pytest
from fastapi.testclient import TestClient

import src.api as api


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setattr(api, "MODEL_DIR", model_dir)
    with TestClient(api.app) as test_client:  # entering runs the lifespan, which loads the model
        yield test_client


def test_returns_label_margin_and_model_version(client):
    response = client.post("/predict", json={"text": "a brilliant, wonderful film"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"label", "sentiment", "margin", "model_version"}
    assert (body["label"], body["sentiment"]) == ("positive", 1)
    assert body["margin"] > 0
    assert len(body["model_version"]) == 12


def test_negative_review_has_a_negative_margin(client):
    body = client.post("/predict", json={"text": "an awful, boring film"}).json()
    assert (body["label"], body["sentiment"]) == ("negative", 0)
    assert body["margin"] < 0


@pytest.mark.parametrize("payload", [
    {},
    {"text": ""},
    {"text": "    "},
    {"text": "10/10 !!!"},
    {"text": 42},
    {"text": "a" * (api.MAX_TEXT_CHARS + 1)},
])
def test_rejects_invalid_input_with_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_unexpected_errors_return_500_without_details(client, monkeypatch):
    def fail(_texts):
        raise ValueError("internal detail that must not leak")

    monkeypatch.setattr(api.preprocessor, "transform", fail)
    response = client.post("/predict", json={"text": "a fine film"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}


def test_startup_stops_with_a_clear_message_without_model_files(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path / "empty")
    with pytest.raises(RuntimeError, match="run: python train.py"):
        with TestClient(api.app):
            pass
