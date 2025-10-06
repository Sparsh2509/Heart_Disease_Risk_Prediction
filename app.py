from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and feature columns
model = joblib.load("rf_heart_model_no_scaler.joblib")
feature_columns = joblib.load("feature_columns_no_scaler.joblib")

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
    return {"message": "Welcome to the Heart Disease Prediction API"}

@app.post("/predict")
def predict(data: HeartData):
    # Convert input to dictionary
    data_dict = data.dict()
    
    # Ensure input follows the correct feature order
    input_data = np.array([[data_dict[col] for col in feature_columns]])
    
    # Predict
    prediction = model.predict(input_data)[0]
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    return {
        "prediction": int(prediction),
        "result": result
    }
