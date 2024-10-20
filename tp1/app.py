from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import load_model
import numpy as np

app = FastAPI()

model = load_model('regression.joblib')

class PredictionRequest(BaseModel):
    size: float
    nb_rooms: int
    garden: int

@app.post("/predict")
def predict(request: PredictionRequest):
    input_array = np.array([[request.size, request.nb_rooms, request.garden]])
    prediction = model.predict(input_array)
    rounded_prediction = round(prediction[0], 2)
    return {"prediction": rounded_prediction}
