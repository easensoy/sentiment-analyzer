"""SentimentModel: the score-to-label policy, predictions, and saving and loading."""
import numpy as np

from src.model import LABELS, SentimentModel


def test_predicts_clear_examples(fitted):
    preprocessor, model = fitted
    features = preprocessor.transform(["a brilliant, wonderful film", "an awful, boring film"])
    assert list(model.predict(features)) == [1, 0]


def test_score_sign_matches_the_prediction(fitted):
    preprocessor, model = fitted
    features = preprocessor.transform(["superb and delightful", "dull and tedious", "words it never saw"])
    scores = model.decision_function(features)
    assert list((scores > 0).astype(int)) == list(model.predict(features))


def test_scores_map_to_labels_by_the_application_policy(fitted, monkeypatch):
    preprocessor, model = fitted
    scores = np.array([1e-9, -1e-9, 0.0, -0.0])
    monkeypatch.setattr(model.model, "decision_function", lambda X: scores)
    labels, returned = model.predict_with_scores(preprocessor.transform(["film"] * 4))
    assert list(labels) == [1, 0, 0, 0]  # above 0 is positive; 0 and below are negative
    assert returned is scores


def test_save_and_load_give_identical_scores(tmp_path, fitted):
    preprocessor, model = fitted
    model.save(tmp_path / "model.joblib")
    restored = SentimentModel()
    restored.load(tmp_path / "model.joblib")
    features = preprocessor.transform(["a brilliant film", "a boring film", "words it never saw"])
    assert list(restored.predict(features)) == list(model.predict(features))
    assert np.allclose(restored.decision_function(features), model.decision_function(features))


def test_labels_map_both_classes():
    assert LABELS == {0: "negative", 1: "positive"}
