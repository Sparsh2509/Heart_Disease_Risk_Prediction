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
    high_chol_flag = int(d["chol"] > 240)
    fbs_flag = int(d["fbs"] > 120)
    restecg_flag = int(d["restecg"] != 0)

    # Step 3: Prepare final 16-feature array
    input_features = [
        d["age"], d["sex"], d["cp"], d["trestbps"], d["chol"],
        d["fbs"], d["restecg"], d["thalach"], d["exang"], d["oldpeak"],
        d["slope"], d["ca"], d["thal"],
        high_chol_flag, fbs_flag, restecg_flag
    ]
    input_array = np.array(input_features).reshape(1, -1)

    # Step 4: Standardize & Predict
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    # Step 5: Return result
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    return {
        "prediction": int(prediction),
        "result": result
    }