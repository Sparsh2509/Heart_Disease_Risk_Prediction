from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# Load Model and Feature Names
model = joblib.load("randomforest_heart_model.joblib")
feature_names = joblib.load("feature_columns.joblib") 

app = FastAPI(title="Heart Disease Prediction API")


# Input Schema
class HeartData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int



# Routes

@app.get("/")
def root():
    return {"message": "Welcome to the Heart Disease Prediction API (No Scaler)"}


@app.post("/predict")
def predict(data: HeartData):
    # Convert input to dictionary
    d = data.dict()

    # Convert to DataFrame with same feature names as training
    input_df = pd.DataFrame([d], columns=feature_names)

    # Predict directly
    prediction = model.predict(input_df)[0]
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    return {
        "prediction": int(prediction),
        "result": result
    }
