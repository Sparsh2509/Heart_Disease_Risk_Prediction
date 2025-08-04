from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("knn_heart_model.joblib")
scaler = joblib.load("knn_scaler.joblib")

app = FastAPI(title="Heart Disease Predictor API")

# Define input features structure
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Prediction API"}


@app.post("/predict")
def predict(data: HeartData):
    # Step 1: Extract features
    d = data.dict()

    # Step 2: Compute engineered features (flags)
    high_chol_flag = int(data.chol > 240)
    fbs_flag = int(data.fbs) 
    restecg_flag = int(data.restecg != 0)

    # Create final input array (make sure order matches training)
    input_data = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol,
                            data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak,
                            data.slope, data.ca, data.thal, high_chol_flag, fbs_flag, restecg_flag]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

    return {
        "prediction": int(prediction[0]),
        "result": result
    }