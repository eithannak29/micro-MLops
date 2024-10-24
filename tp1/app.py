from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import load_model
import numpy as np
from transformers import pipeline

app = FastAPI()

model = load_model('regression.joblib')

translator = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')

class PredictionRequest(BaseModel):
    size: float
    nb_rooms: int
    garden: int

class TranslationRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: PredictionRequest):
    input_array = np.array([[request.size, request.nb_rooms, request.garden]])
    prediction = model.predict(input_array)
    rounded_prediction = round(prediction[0], 2)
    return {"prediction": rounded_prediction}

@app.post("/translate")
def translate(request: TranslationRequest):
    result = translator(request.text)
    return {"translation": result[0]['translation_text']}
