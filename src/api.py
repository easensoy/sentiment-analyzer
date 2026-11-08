import hashlib
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from .model import LABELS, SentimentModel
from .preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MAX_TEXT_CHARS = 20_000  # the longest IMDB review is about 13,700 characters
# Same shape as FastAPI's own validation errors, so clients can handle every 422 alike.
NO_KNOWN_WORDS = {
    "type": "no_recognized_vocabulary",
    "loc": ["body", "text"],
    "msg": "Text contains no recognized vocabulary: none of its words are in the model's training "
           "vocabulary (single letters never are), so there is nothing to score",
}

model = SentimentModel()
preprocessor = TextPreprocessor()
model_version = "not loaded"


def artefact_version(*paths: Path) -> str:
    """First 12 hex characters of the SHA-256 over the model files, so each answer names its model."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the fitted vectoriser and SVC that train.py saved together.
    global preprocessor, model_version
    model_path = MODEL_DIR / "model.joblib"
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    try:
        model.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"No trained model in {MODEL_DIR}; run: python train.py") from exc
    model_version = artefact_version(model_path, preprocessor_path)
    yield


app = FastAPI(title="Sentiment Analyzer", lifespan=lifespan)


class TextInput(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS, description="An English review")

    @field_validator("text")
    @classmethod
    def has_english_letters(cls, value: str) -> str:
        # clean_text keeps only ASCII letters, so text without any leaves the model nothing to read.
        if not re.search(r"[A-Za-z]", value):
            raise ValueError("text must contain English letters")
        return value


class Prediction(BaseModel):
    label: Literal["negative", "positive"]
    sentiment: int = Field(description="0 = negative, 1 = positive; kept for existing clients")
    margin: float = Field(description="The linear SVM's signed decision score, at full precision. Above 0 is "
                                      "positive; 0 or below is negative. Not a probability")
    model_version: str = Field(description="First 12 hex characters of the SHA-256 of the loaded model files")


def prediction_failed() -> HTTPException:
    logger.exception("prediction failed")
    return HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict", response_model=Prediction)
def predict_sentiment(input_data: TextInput) -> Prediction:
    """Classify an English review.

    The label comes from the linear SVM's signed decision score, which is returned at full
    precision: above 0 is positive, and 0 or below is negative. The score is not a probability.

    Returns 422 when the text is empty, longer than 20,000 characters or has no English letters,
    and when none of its words are in the model's vocabulary (a single letter, or only unseen
    words), because then there is nothing to score.
    """
    # A plain def, so FastAPI runs the CPU-bound scoring in a worker thread instead of on the event loop.
    try:
        features = preprocessor.transform([input_data.text])
    except Exception:
        raise prediction_failed()
    if features.nnz == 0:
        raise HTTPException(status_code=422, detail=[NO_KNOWN_WORDS])
    try:
        labels, scores = model.predict_with_scores(features)  # one decision-function call
    except Exception:
        raise prediction_failed()
    sentiment = int(labels[0])
    return Prediction(label=LABELS[sentiment], sentiment=sentiment, margin=float(scores[0]),
                      model_version=model_version)
