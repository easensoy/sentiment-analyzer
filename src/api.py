import hashlib
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .pipeline import LABELS, ArtefactError, labels_from_scores, load

logger = logging.getLogger(__name__)

# The published artefact lives under models/ (artefacts/<version>/ plus the serving.json pointer);
# SENTIMENT_MODELS_DIR points the server at another directory.
MODEL_DIR = Path(os.environ.get("SENTIMENT_MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))
MAX_TEXT_CHARS = 20_000  # the longest IMDB review is about 13,700 characters
# Same shape as FastAPI's own validation errors, so clients can handle every 422 alike.
NO_KNOWN_WORDS = {
    "type": "no_recognized_vocabulary",
    "loc": ["body", "text"],
    "msg": "Text contains no recognized vocabulary: none of its words are in the model's training "
           "vocabulary (single letters never are), so there is nothing to score",
}

# Filled at startup from one validated Pipeline: features = cleaning + TF-IDF, classifier = the linear model.
state = SimpleNamespace(features=None, classifier=None, manifest=None, model_version="not loaded")


def artefact_version(*paths: Path) -> str:
    """First 12 hex characters of the SHA-256 over files (the legacy two-file artefact's identifier)."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load and validate the published Pipeline: schema version, SHA-256 identity, steps, cleaning
    # version, classes, feature dimension and scikit-learn version must all match its manifest.
    try:
        pipeline, manifest = load(MODEL_DIR)
    except FileNotFoundError as exc:
        raise RuntimeError(f"No published model in {MODEL_DIR}; run: python train.py") from exc
    except ArtefactError as exc:
        raise RuntimeError(f"Refusing to serve the model in {MODEL_DIR}: {exc}") from exc
    state.features, state.classifier = pipeline[:-1], pipeline[-1]
    state.manifest, state.model_version = manifest, manifest["model_version"]
    yield


app = FastAPI(title="Sentiment Analyzer", lifespan=lifespan)


class TextInput(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS, description="An English review")

    @field_validator("text")
    @classmethod
    def has_english_letters(cls, value: str) -> str:
        # Every cleaning version keeps ASCII letters, so text without any leaves the model nothing to read.
        if not re.search(r"[A-Za-z]", value):
            raise ValueError("text must contain English letters")
        return value


class Prediction(BaseModel):
    label: Literal["negative", "positive"]
    sentiment: int = Field(description="0 = negative, 1 = positive; kept for existing clients")
    margin: float = Field(description="The linear classifier's signed decision score w·x + b, at full precision. "
                                      "Above 0 is positive; 0 or below is negative. Not a probability")
    model_version: str = Field(description="First 12 hex characters of the SHA-256 of the loaded Pipeline artefact")


class ModelInfo(BaseModel):
    model_version: str
    candidate_id: str
    schema_version: int
    config: dict
    n_features: int
    score_policy: str
    training: dict


def prediction_failed() -> HTTPException:
    logger.exception("prediction failed")
    return HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict", response_model=Prediction)
def predict_sentiment(input_data: TextInput) -> Prediction:
    """Classify an English review.

    The label comes from the linear classifier's signed decision score, which is returned at full
    precision: above 0 is positive, and 0 or below is negative. The score is not a probability.

    Returns 422 when the text is empty, longer than 20,000 characters or has no English letters,
    and when none of its words are in the model's vocabulary (a single letter, or only unseen
    words), because then there is nothing to score.
    """
    # A plain def, so FastAPI runs the CPU-bound scoring in a worker thread instead of on the event loop.
    try:
        features = state.features.transform([input_data.text])  # the same cleaning path as training
    except Exception:
        raise prediction_failed()
    if features.nnz == 0:
        raise HTTPException(status_code=422, detail=[NO_KNOWN_WORDS])
    try:
        scores = state.classifier.decision_function(features)  # one decision-function call
    except Exception:
        raise prediction_failed()
    sentiment = int(labels_from_scores(scores)[0])
    return Prediction(label=LABELS[sentiment], sentiment=sentiment, margin=float(scores[0]),
                      model_version=state.model_version)


@app.get("/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    """Identity and configuration of the served artefact, from its validated manifest."""
    m = state.manifest
    return ModelInfo(model_version=m["model_version"], candidate_id=m["candidate_id"],
                     schema_version=m["schema_version"], config=m["config"], n_features=m["n_features"],
                     score_policy=m["score_policy"], training=m["training"])
