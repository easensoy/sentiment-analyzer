from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import SentimentModel
from .preprocessor import TextPreprocessor

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

model = SentimentModel()
preprocessor = TextPreprocessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the fitted vectoriser and SVC that train.py saved together.
    global preprocessor
    try:
        model.load(MODEL_DIR / "model.joblib")
        preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    except FileNotFoundError as exc:
        raise RuntimeError(f"No trained model in {MODEL_DIR}; run: python train.py") from exc
    yield


app = FastAPI(lifespan=lifespan)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(input_data: TextInput):
    try:
        # Preprocess the input text
        vectorized_text = preprocessor.transform([input_data.text])
        # Make prediction
        prediction = model.predict(vectorized_text)
        return {"sentiment": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
