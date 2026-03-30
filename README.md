# ❤️ Heart Disease Prediction

A machine learning-powered web API using FastAPI to predict the presence of heart disease based on patient health indicators and medical history.
 This project is based on the dataset by kaggle [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci?resource=download).

 ### Deployed Project Using API on Streamlit 

 Test APP HERE -
 [https://heartdiseaseriskprediction-qqg6vlyqzqp3vqczxab2ru.streamlit.app/](https://heartdiseaseriskprediction-qqg6vlyqzqp3vqczxab2ru.streamlit.app/)

---

## 🚀 Features

- Predicts **presence or absence of heart disease** (0 = No Disease, 1 = Disease)
- Uses a hybrid system:
  - 🎯 Machine Learning (Random Forest) — provides probability-based prediction
  - 🩺 Rule-Based Scoring — evaluates each input against known medical thresholds

- Organized input via five intuitive medical sections:
  - **Demographic Information** – age, sex
  - **Chest Pain & Vitals** – chest pain type, resting blood pressure, cholesterol
  - **Blood & Sugar Data** – fasting blood sugar, maximum heart rate
  - **ECG & Exercise Information** – resting ECG results, exercise-induced angina, ST depression
  - **Scan & Diagnostic Results** – number of major vessels colored, thalassemia type

---

## 🧠 Machine Learning Prediction (Random Forest Classifier)

- Predicts the probability of having heart disease using a trained Random Forest model.
- Outputs two probabilities:
  - Heart Disease Probability (%)
  - No Disease Probability (%)
- Generates clear, human-readable messages:
  - 🟢 Low risk → "Low risk — your heart health seems good"
  - 🟡 Moderate risk → "Moderate risk — maintain a healthy lifestyle"
  - 🔴 High risk → "High risk — please consult a cardiologist immediately"
- Achieves an average accuracy of ~90.20%.

---

## ⚙️ Rule-Based Risk Scoring System

- Each patient’s input is evaluated against known medical thresholds.
- Every abnormal feature adds to a cumulative health risk score, which helps explain why the model made its prediction.

### Scoring Logic
- age > 50 →       +1 point (Older age)
- cp ≥ 2 →         +1 point (Abnormal chest pain: non-anginal/asymptomatic)
- trestbps > 140 → +2 points (High resting blood pressure)
- chol > 240 →     +2 points (High cholesterol)
- fbs == 1 →       +1 point (High fasting blood sugar)
- restecg != 0 →   +1 point (Abnormal ECG result)
- thalach < 130 →  +2 points (Low max heart rate achieved)
- exang == 1 →     +2 points (Exercise-induced angina)
- oldpeak > 1.5 →  +2 points (Significant ST depression)
- slope == 2 →     +1 point (Downsloping ST segment)
- ca ≥ 1 →         +2 points (Major vessels affected)
- thal != 0 →      +2 points (Abnormal thalassemia)

---

### 💡 Score Interpretation
- 0–3 → 🟢 Low Risk: Heart appears healthy.
- 4–7 → 🟡 Moderate Risk: Maintain lifestyle; consider a routine check-up.
- ≥8 → 🔴 High Risk: Consult a cardiologist immediately.


- Returns a clear **prediction** (Heart Disease / No Heart Disease) with **confidence score of 90.20%**
- Built using a lightweight and scalable **FastAPI backend**

---

## 📂 Project Structure

```
Heart_Disease_Prediction/
├── app.py                                              # FastAPI backend logic
├── Heart_Disease_Predict_Model.py                      # Model training script
├── Heart_disease_cleveland_new.csv                     # Cleaned cleveland dataset used for training model
├── randomforest_heart_model.joblib                     # Trained Random forest classifier model
├── feature_columns.joblib                              # Saved feature order for clean prediction
├── requirements.txt                                    # Python package dependencies
├── streamlit_app.py                                    # Deployed App Using API On Streamlit
└── README.md                                           # Project documentation
```

---

## 🛠️ Installation

### Prerequisites:
- Python 3.13.7

### Setup:
```bash
# Repository Name
Heart_Disease_Risk_Prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate 

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Model Training

To train or retrain the model using the dataset:

```bash
python Heart_Disease_Predict_Model.py
```

This will generate the model and features order files:
- `randomforest_heart_model.joblib `
- `feature_columns.joblib`

---

## 🚦 Running the API

Start the FastAPI server:
```bash
uvicorn app:app --reload
```

Navigate to:
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- AWS EB API URL: [http://heart-disease-api-env.eba-vrubpnic.ap-south-1.elasticbeanstalk.com](https://http://heart-disease-api-env.eba-vrubpnic.ap-south-1.elasticbeanstalk.com)
- Render: [https://heart-disease-risk-prediction-hpkk.onrender.com](https://heart-disease-risk-prediction-hpkk.onrender.com)

---

## 📥 API Usage

### Endpoint:
```
POST /predict
```

### Request Body Example:
```json
{
  "age": 56,
  "sex": 1,
  "cp": 2,
  "trestbps": 150,
  "chol": 260,
  "fbs": 1,
  "restecg": 1,
  "thalach": 120,
  "exang": 1,
  "oldpeak": 2.5,
  "slope": 2,
  "ca": 2,
  "thal": 2
}

```

### Sample Response:
```json
{
    "ml_prediction": {
        "heart_disease_probability": "85.83%",
        "no_disease_probability": "14.17%",
        "ml_risk_message": "High risk — please consult a cardiologist immediately"
    },
    "score_prediction": {
        "risk_score": 19,
        "risk_level": "High Risk",
        "threshold_flags": [
            "Older age (>50 years)",
            "Abnormal chest pain type (non-anginal/asymptomatic)",
            "High resting blood pressure (>140 mm Hg)",
            "High cholesterol level (>240 mg/dl)",
            "High fasting blood sugar (>120 mg/dl)",
            "Abnormal ECG result",
            "Low max heart rate (<130 bpm)",
            "Exercise-induced angina detected",
            "Significant ST depression (>1.5)",
            "Downsloping ST segment",
            "Major vessels affected (ca >= 1)",
            "Abnormal thalassemia (thal != 0)"
        ]
    },
    "final_advice": "The ML model and scoring system together indicate your overall heart risk. For accurate diagnosis, please consult a healthcare professional."
}

```

---

## 🧠 Model Overview

- Algorithm: `Random Forest Classifier`
- Input Features: 13
- Target: `Target` (0 or 1)
- Evaluation: Accuracy ~ 90.20%

---

## 📘 Dataset Info

- Source: Heart Disease Cleveland UCI
- [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci?resource=download)

---

