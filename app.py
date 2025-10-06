from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("rf_heart_model.joblib")
scaler = joblib.load("rf_scaler.joblib")

app = FastAPI(title="Heart Disease Prediction API")

# -------------------------------
# Input Schema
# -------------------------------
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

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Heart Disease Prediction API"}

@app.post("/predict")
def predict(data: HeartData):
    d = data.dict()

    # Derived features (same as training)
    fbs_flag = int(data.fbs > 120)
    restecg_flag = int(data.restecg != 0)

    # Maintain same order as training
    input_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal,
        fbs_flag, restecg_flag
    ]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    return {
        "prediction": int(prediction),
        "result": result
    }
