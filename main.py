from fastapi import FastAPI
import pickle
from src.data_model import WaterData
import pandas as pd

app = FastAPI(
title="Water Potability Prediction API",
description="An API for predicting water potability using a trained machine learning model.",
)

with open(r"./water_potability_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Water Potability Prediction API!"}

@app.post("/predict")
def predict_potability(data: WaterData):
    input_data = pd.DataFrame({
        'ph': [data.ph],
        'Hardness': [data.Hardness],
        'Solids': [data.Solids],
        'Chloramines': [data.Chloramines],
        'Sulfate': [data.Sulfate],
        'Conductivity': [data.Conductivity],
        'Organic_carbon': [data.Organic_carbon],
        'Trihalomethanes': [data.Trihalomethanes],
        'Turbidity': [data.Turbidity]
        })

    prediction = model.predict(input_data)
    potability = "Potable" if prediction[0] == 1 else "Not Potable"

    return {"potability": potability}