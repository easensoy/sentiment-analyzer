"""Shared fixtures: a tiny, deterministic model, so tests run in milliseconds without models/."""
import os

import pytest

from src.model import SentimentModel
from src.pipeline import build_pipeline, publish
from src.preprocessor import TextPreprocessor

# Mirrored sentences: the shared words (film, story, acting...) appear in both classes,
# so only the sentiment words decide the label.
POSITIVE = [
    "a wonderful film with brilliant acting",
    "great story and superb performances",
    "loved it, a delightful and charming film",
    "excellent direction and a beautiful score",
    "fun, enjoyable and very entertaining",
]
NEGATIVE = [
    "a boring film with awful acting",
    "terrible story and wooden performances",
    "hated it, a dull and tedious film",
    "poor direction and a forgettable score",
    "bad, annoying and very disappointing",
]
CONFIG = {"cleaning": "legacy-v1", "ngram_range": [1, 1], "min_df": 1, "estimator": "svc", "C": 1.0}


@pytest.fixture
def sentences():
    return POSITIVE, NEGATIVE


@pytest.fixture
def fitted():
    """The legacy two-object model (TextPreprocessor + SentimentModel), kept for loading old artefacts."""
    preprocessor = TextPreprocessor()
    model = SentimentModel()
    model.train(preprocessor.fit_transform(POSITIVE + NEGATIVE), [1] * len(POSITIVE) + [0] * len(NEGATIVE))
    return preprocessor, model


@pytest.fixture
def tiny_pipeline():
    pipeline = build_pipeline(CONFIG)
    pipeline.fit(POSITIVE + NEGATIVE, [1] * len(POSITIVE) + [0] * len(NEGATIVE))
    return pipeline


@pytest.fixture
def model_dir(tmp_path, tiny_pipeline):
    """A models directory as train.py leaves it: an immutable artefact plus the serving.json pointer."""
    publish(tiny_pipeline, tmp_path / "models", candidate_id="tiny-test-model", config=CONFIG,
            training={"protocol": "unit test", "rows": 10})
    return tmp_path / "models"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """On GitHub Actions, also report each failure as an error annotation: the checks API exposes annotations
    publicly, whereas job logs need a signed-in account."""
    outcome = yield
    report = outcome.get_result()
    if report.failed and os.environ.get("GITHUB_ACTIONS") == "true":
        message = str(report.longrepr)[-3000:].replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        title = report.nodeid.replace("%", "%25").replace(":", "%3A").replace(",", "%2C")  # property escaping
        item.config.pluginmanager.get_plugin("terminalreporter").write_line(f"::error title={title}::{message}")
