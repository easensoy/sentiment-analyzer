from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import SentimentModel
from .preprocessor import TextPreprocessor

app = FastAPI()
model = SentimentModel()
preprocessor = TextPreprocessor()

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
