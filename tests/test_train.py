"""train.py end to end on a tiny CSV: the split, the published Pipeline and the saved metrics."""
import json
import string

import pandas as pd
import pytest

import train
from src.pipeline import load

TOKENS = [f"qx{letter}" for letter in string.ascii_lowercase[:20]]  # one made-up word per row


@pytest.fixture
def trained(tmp_path, sentences):
    positive, negative = sentences
    rows = [{"text": f"{(positive if i < 10 else negative)[i % 5]} {token}", "label": int(i < 10)}
            for i, token in enumerate(TOKENS)]
    data, out = tmp_path / "tiny.csv", tmp_path / "models"
    pd.DataFrame(rows).to_csv(data, index=False)
    train.main(["--data", str(data), "--out", str(out)])
    return out


def test_vectoriser_never_sees_held_out_rows(trained):
    pipeline, _ = load(trained)
    vocabulary = set(pipeline.named_steps["tfidf"].vocabulary_)
    held_out = {text.split()[-1] for text in pd.read_csv(trained / "test_predictions.csv")["text"]}
    assert held_out and not held_out & vocabulary
    assert set(TOKENS) - held_out <= vocabulary


def test_metrics_include_an_accuracy_interval(trained):
    metrics = json.loads((trained / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["rows"] == {"train": 16, "test": 4}
    low, high = metrics["accuracy_ci95"]
    assert 0 <= low <= metrics["accuracy"] <= high <= 1
    assert sum(map(sum, metrics["confusion_matrix"]["rows_true_cols_predicted"])) == 4


def test_publishes_a_validated_pipeline_with_its_training_record(trained):
    pipeline, manifest = load(trained)
    assert manifest["candidate_id"] == train.CANDIDATE_ID and manifest["config"] == train.CONFIG
    assert manifest["training"]["rows"] == 16 and manifest["training"]["seed"] == 42
    metrics = json.loads((trained / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_version"] == manifest["model_version"]
    predictions = pd.read_csv(trained / "test_predictions.csv")
    rescored = pipeline[-1].decision_function(pipeline[:-1].transform(predictions["text"].tolist()))
    assert list(rescored) == pytest.approx(list(predictions["score"]), abs=1e-12)


def test_rejects_a_csv_without_the_expected_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"review": ["fine"], "sentiment": [1]}).to_csv(bad, index=False)
    with pytest.raises(SystemExit):
        train.main(["--data", str(bad), "--out", str(tmp_path / "models")])
