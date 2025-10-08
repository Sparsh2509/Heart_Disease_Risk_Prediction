from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("randomforest_heart_model.joblib")
feature_names = joblib.load("feature_columns.joblib")

app = FastAPI(title="Heart Disease Prediction API (Hybrid Model + Scoring)")


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


# Threshold scoring setup
def calculate_score(data: dict):
    score = 0
    flags = []

    if data["age"] > 50:
        score += 1
        flags.append("Older age (>50 years)")
    if data["cp"] >= 2:
        score += 1
        flags.append("Abnormal chest pain type (non-anginal/asymptomatic)")
    if data["trestbps"] > 140:
        score += 2
        flags.append("High resting blood pressure (>140 mm Hg)")
    if data["chol"] > 240:
        score += 2
        flags.append("High cholesterol level (>240 mg/dl)")
    if data["fbs"] == 1:
        score += 1
        flags.append("High fasting blood sugar (>120 mg/dl)")
    if data["restecg"] != 0:
        score += 1
        flags.append("Abnormal ECG result")
    if data["thalach"] < 130:
        score += 2
        flags.append("Low max heart rate (<130 bpm)")
    if data["exang"] == 1:
        score += 2
        flags.append("Exercise-induced angina detected")
    if data["oldpeak"] > 1.5:
        score += 2
        flags.append("Significant ST depression (>1.5)")
    if data["slope"] == 2:
        score += 1
        flags.append("Downsloping ST segment")
    if data["ca"] >= 1:
        score += 2
        flags.append("Major vessels affected (ca >= 1)")
    if data["thal"] != 0:
        score += 2
        flags.append("Abnormal thalassemia (thal != 0)")

    return score, flags


@app.get("/")
def root():
    return {"message": "Welcome to the Heart Disease Prediction API (Hybrid Model + Scoring)"}


@app.post("/predict")
def predict(data: HeartData):
    d = data.dict()

    # Convert to DataFrame with proper feature names
    input_df = pd.DataFrame([d], columns=feature_names)

    # Model Prediction
    prob = model.predict_proba(input_df)[0]
    disease_prob = round(prob[1] * 100, 2)
    no_disease_prob = round(prob[0] * 100, 2)

    # Rule-based Scoring
    score, flags = calculate_score(d)

    # Interpret score
    if score <= 3:
        score_risk = "Low Risk "
    elif 4 <= score <= 7:
        score_risk = "Moderate Risk"
    else:
        score_risk = "High Risk"


    # Message based on model probability
    if disease_prob < 30:
        ml_message = "Low risk — your heart health seems good"
    elif 30 <= disease_prob <= 60:
        ml_message = "Moderate risk — maintain a healthy lifestyle"
    else:
        ml_message = "High risk — please consult a cardiologist immediately"

    
    # Combine insights
    return {
        "ml_prediction": {
            "heart_disease_probability": f"{disease_prob}%",
            "no_disease_probability": f"{no_disease_prob}%",
            "ml_risk_message": ml_message
        },
        "score_prediction": {
            "risk_score": score,
            "risk_level": score_risk,
            "threshold_flags": flags if flags else ["All vitals within healthy range"]
        },
        "final_advice": "The ML model and scoring system together indicate your overall heart risk. "
                        "For accurate diagnosis, please consult a healthcare professional."
    }



