# ❤️ Heart Disease Prediction

A machine learning-powered web API using FastAPI to predict the presence of heart disease based on patient health indicators and medical history.
 This project is based on the dataset by kaggle [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci?resource=download).

---

## 🚀 Features

- Predicts **presence or absence of heart disease** (0 = No Disease, 1 = Disease)
- Organized input via five intuitive medical sections:
  - **Demographic Information** – age, sex
  - **Chest Pain & Vitals** – chest pain type, resting blood pressure, cholesterol
  - **Blood & Sugar Data** – fasting blood sugar, maximum heart rate
  - **ECG & Exercise Information** – resting ECG results, exercise-induced angina, ST depression
  - **Scan & Diagnostic Results** – number of major vessels colored, thalassemia type
- Provides warning flags for abnormal health indicators:
  - ⚠️ High cholesterol                            (chol > 240 mg/dl)
  - ⚠️ High fasting blood sugar                    (fbs > 120 mg/dl)
  - ⚠️ Low maximum heart rate (exercise tolerance) (thalach < 130 bpm)
  - ⚠️ High resting blood pressure                 (trestbps > 140 mm Hg)
  - If not in any flag warning it return **All vitals within healthy range** 
- Trained using the **Random Forest Classifier** algorithm for accurate classification
- Returns a clear **prediction** (Heart Disease / No Heart Disease) with **confidence score of 90.20%**
- Built using a lightweight and scalable **FastAPI backend**

---

## 📂 Project Structure

```
Heart_Disease_Prediction/
├── app.py                                              # FastAPI backend logic
├── Heart_Disease_Predict_Model.py                      # Model training script
├── Heart_disease_cleveland_new.csv                     # Cleaned cleveland dataset used for training
├── randomforest_heart_model.joblib                     # Trained Random forest classifier model
├── feature_columns.joblib                              # Saved feature order for clean prediction
├── requirements.txt                                    # Python package dependencies
└── README.md                                           # Project documentation
```

---

## 🛠️ Installation

### Prerequisites:
- Python 3.9.5

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
  "age": 55,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 230,
  "fbs": 110,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.5,
  "slope": 1,
  "ca": 2,
  "thal": 1
}


```

### Sample Response:
```json
{
    "prediction": 0,
    "result": "No Heart Disease",
    "health_flags": [
        "All vitals within healthy range"
    ]
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

