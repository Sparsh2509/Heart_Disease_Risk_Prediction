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

@app.post("/predict")
def predict(data: HeartData):
    d = data.dict()
    score = 0
    reasons = []

    # Threshold scoring rules
    if d["chol"] > 240:
        score += 2
        reasons.append("High cholesterol (>240 mg/dl)")
    if d["trestbps"] > 140:
        score += 2
        reasons.append("High resting BP (>140 mm Hg)")
    if d["thalach"] < 130:
        score += 2
        reasons.append("Low maximum heart rate (<130 bpm)")
    if d["fbs"] > 120:
        score += 1
        reasons.append("High fasting blood sugar (>120 mg/dl)")
    if d["age"] > 50:
        score += 1
        reasons.append("Older age (>50 years)")
    if d["cp"] >= 2:
        score += 1
        reasons.append("Abnormal chest pain pattern")
    if d["exang"] == 1:
        score += 2
        reasons.append("Exercise-induced angina detected")

    # Risk classification
    if score <= 2:
        risk_level = "Low Risk ❤️"
        advice = "Your vitals look good! Keep maintaining a healthy lifestyle."
    elif 3 <= score <= 5:
        risk_level = "Moderate Risk 💛"
        advice = "Some parameters are elevated. Regular exercise and a balanced diet are recommended."
    else:
        risk_level = "High Risk ❤️‍🔥"
        advice = "You have multiple high-risk indicators. Consider consulting a cardiologist soon."

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "reasons": reasons if reasons else ["All vitals in healthy range"],
        "advice": advice
    }
