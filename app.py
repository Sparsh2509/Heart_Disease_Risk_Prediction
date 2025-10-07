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


THRESHOLDS = {
    "chol": 240,      # High cholesterol
    "fbs": 120,       # High fasting sugar
    "thalach": 130,   # Low heart rate (danger zone if below)
    "trestbps": 140   # High blood pressure
}

# Routes
def root():
    return {"message": "Welcome to the Heart Disease Prediction API"}

@app.post("/predict")
def predict(data: HeartData):
    d = data.dict()

    # Convert to DataFrame (keeps feature names safe)
    input_df = pd.DataFrame([d], columns=feature_names)

    # Threshold-based Risk Flags   
    health_flags = []

    # Cholesterol check
    if d["chol"] > THRESHOLDS["chol"]:
        health_flags.append("High cholesterol level (chol > 240 mg/dl)")

    # Fasting blood sugar check
    if d["fbs"] > THRESHOLDS["fbs"]:
        health_flags.append("High fasting blood sugar (fbs > 120 mg/dl)")

    # Max heart rate check
    if d["thalach"] < THRESHOLDS["thalach"]:
        health_flags.append("Low max heart rate achieved (thalach < 130 bpm)")

    # Resting BP check
    if d["trestbps"] > THRESHOLDS["trestbps"]:
        health_flags.append("High resting blood pressure (trestbps > 140 mm Hg)")

    else:
        health_flags.append("All vitals within healthy range")

    # Predict using trained model
    prediction = model.predict(input_df)[0]
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    return {
        "prediction": int(prediction),
        "result": result,
        "health_flags": health_flags
    }