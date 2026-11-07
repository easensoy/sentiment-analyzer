"""Shared fixtures: a tiny, deterministic model, so tests run in milliseconds without models/."""
import joblib
import pytest

from src.model import SentimentModel
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


@pytest.fixture
def sentences():
    return POSITIVE, NEGATIVE


@pytest.fixture
def fitted():
    preprocessor = TextPreprocessor()
    model = SentimentModel()
    model.train(preprocessor.fit_transform(POSITIVE + NEGATIVE), [1] * len(POSITIVE) + [0] * len(NEGATIVE))
    return preprocessor, model


@pytest.fixture
def model_dir(tmp_path, fitted):
    """The two files train.py writes, built from the tiny model."""
    preprocessor, model = fitted
    model.save(tmp_path / "model.joblib")
    joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")
    return tmp_path
