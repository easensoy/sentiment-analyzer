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
    margin: float = Field(description="Signed SVM decision value: above 0 means positive. A distance, not a probability")
    model_version: str = Field(description="First 12 hex characters of the SHA-256 of the loaded model files")


@app.post("/predict", response_model=Prediction)
def predict_sentiment(input_data: TextInput) -> Prediction:
    # A plain def, so FastAPI runs the CPU-bound scoring in a worker thread instead of on the event loop.
    try:
        features = preprocessor.transform([input_data.text])
        sentiment = int(model.predict(features)[0])
        margin = float(model.decision_function(features)[0])
    except Exception:
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    return Prediction(label=LABELS[sentiment], sentiment=sentiment, margin=round(margin, 4),
                      model_version=model_version)
