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
- Trained using the **K-Nearest Neighbors (KNN)** algorithm for accurate classification
- Returns a clear **prediction** (Heart Disease / No Heart Disease) with **confidence score of 90.16%**
- Built using a lightweight and scalable **FastAPI backend**


---

## 📂 Project Structure

```
Heart_Disease_Prediction/
├── app.py                                              # FastAPI backend logic
├── Heart_Disease_Predict_Model.py                      # Model training script
├── Heart_disease_cleveland_new.csv                     # Cleaned cleveland dataset used for training
├── knn_heart_model.joblib                              # Trained KNN model
├── knn_scaler.joblib                                   # Scaler for more good fitting
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

This will generate the model and scaler files:
- `knn_heart_model.joblib`
- `knn_scaler.joblib`

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
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 2
}

```

### Sample Response:
```json
{
  "prediction": 1,
  "result": "Heart Disease Detected"
}

```

---

## 🧠 Model Overview

- Algorithm: `K-Nearest Neighbors (KNN)`
- Input Features: 13
- Target: `Target` (0 or 1)
- Evaluation: Accuracy ~ 90.16%

---

## 📘 Dataset Info

- Source: Heart Disease Cleveland UCI
- [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci?resource=download)

---

