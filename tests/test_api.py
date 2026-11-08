"""The HTTP contract of POST /predict and GET /model, using the tiny published Pipeline from conftest."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api as api


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setattr(api, "MODEL_DIR", model_dir)
    with TestClient(api.app) as test_client:  # entering runs the lifespan, which loads and validates the model
        yield test_client


def test_returns_label_margin_and_model_version(client, model_dir):
    response = client.post("/predict", json={"text": "a brilliant, wonderful film"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"label", "sentiment", "margin", "model_version"}
    assert (body["label"], body["sentiment"]) == ("positive", 1)
    assert body["margin"] > 0
    pointer = json.loads((model_dir / "serving.json").read_text(encoding="utf-8"))
    assert body["model_version"] == pointer["model_version"] and len(body["model_version"]) == 12


def test_negative_review_has_a_negative_margin(client):
    body = client.post("/predict", json={"text": "an awful, boring film"}).json()
    assert (body["label"], body["sentiment"]) == ("negative", 0)
    assert body["margin"] < 0


def test_model_endpoint_reports_the_validated_manifest(client):
    info = client.get("/model").json()
    assert info["candidate_id"] == "tiny-test-model" and info["schema_version"] == 1
    assert info["config"]["cleaning"] == "legacy-v1" and info["n_features"] > 0


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


def test_html_line_breaks_are_cleaned_as_in_training(client):
    """Serving applies the training cleaning: <br /> becomes a space, so it cannot add a 'br' token."""
    with_tags = client.post("/predict", json={"text": "a wonderful<br /><br />film"}).json()
    without = client.post("/predict", json={"text": "a wonderful film"}).json()
    assert with_tags["margin"] == without["margin"]


def test_unexpected_errors_return_500_without_details(client, monkeypatch):
    def fail(_texts):
        raise ValueError("internal detail that must not leak")

    monkeypatch.setattr(api.state.features, "transform", fail)
    response = client.post("/predict", json={"text": "a fine film"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}


def test_startup_stops_with_a_clear_message_without_model_files(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path / "empty")
    with pytest.raises(RuntimeError, match="run: python train.py"):
        with TestClient(api.app):
            pass


def test_startup_refuses_an_artefact_that_does_not_match_its_manifest(model_dir, monkeypatch):
    pointer = json.loads((model_dir / "serving.json").read_text(encoding="utf-8"))
    artefact = model_dir / pointer["artefact"] / "pipeline.joblib"
    artefact.write_bytes(artefact.read_bytes() + b"tampered")
    monkeypatch.setattr(api, "MODEL_DIR", model_dir)
    with pytest.raises(RuntimeError, match="Refusing to serve"):
        with TestClient(api.app):
            pass


@pytest.fixture
def scored_with(client, monkeypatch):
    """Make the loaded classifier return a chosen decision score, and record every call to it."""
    calls = []

    def install(score):
        def decision_function(X):
            calls.append(X.shape[0])
            return np.full(X.shape[0], score)

        monkeypatch.setattr(api.state.classifier, "decision_function", decision_function)
        return calls

    return install


@pytest.mark.parametrize("score, label", [
    (1e-9, "positive"),
    (-1e-9, "negative"),
    (0.0, "negative"),
    (-0.0, "negative"),
])
def test_label_follows_the_full_precision_score(client, scored_with, score, label):
    calls = scored_with(score)
    body = client.post("/predict", json={"text": "a brilliant film"}).json()
    assert (body["label"], body["sentiment"]) == (label, int(label == "positive"))
    assert body["margin"] == score  # not rounded: 1e-9 must not come back as 0.0
    assert calls == [1]


def test_a_valid_request_calls_the_decision_function_once(client, monkeypatch):
    calls = []
    real = api.state.classifier.decision_function

    def counting(X):
        calls.append(X.shape[0])
        return real(X)

    monkeypatch.setattr(api.state.classifier, "decision_function", counting)
    assert client.post("/predict", json={"text": "a brilliant, wonderful film"}).status_code == 200
    assert calls == [1]


@pytest.mark.parametrize("text", ["a", "I", "qwzx blorft zzyzx"])
def test_text_without_known_words_gets_422(client, scored_with, text):
    calls = scored_with(1.0)
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert (error["type"], error["loc"]) == ("no_recognized_vocabulary", ["body", "text"])
    assert "no recognized vocabulary" in error["msg"]
    assert calls == []  # nothing was scored
