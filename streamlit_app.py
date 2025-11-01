import streamlit as st
import requests
import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
API_URL = "http://heart-disease-api-env.eba-vrubpnic.ap-south-1.elasticbeanstalk.com/predict"  
# 👆 Replace with your deployed API URL

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("🩺 Heart Disease Prediction (Hybrid Model + Scoring)")
st.markdown("""
This app uses an ML model **and** a medical scoring system to estimate your risk of heart disease.  
Fill the form below to get a personalized report.
""")


# -----------------------------
# INPUT FORM
# -----------------------------
with st.form("heart_form"):
    st.subheader("🔹 Patient Health Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Female (0)", "Male (1)"])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=250, value=120)

    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl?", ["No (0)", "Yes (1)"])
        restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)

    with col3:
        exang = st.selectbox("Exercise Induced Angina?", ["No (0)", "Yes (1)"])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
        ca = st.selectbox("Major Vessels Colored (0–4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

    submitted = st.form_submit_button("🔍 Predict Risk")


# -----------------------------
# API CALL
# -----------------------------
if submitted:
    # Prepare request body
    input_data = {
        "age": age,
        "sex": 1 if "Male" in sex else 0,
        "cp": int(cp),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if "Yes" in fbs else 0,
        "restecg": int(restecg),
        "thalach": thalach,
        "exang": 1 if "Yes" in exang else 0,
        "oldpeak": oldpeak,
        "slope": int(slope),
        "ca": int(ca),
        "thal": int(thal)
    }

    with st.spinner("🧠 Analyzing your data..."):
        try:
            response = requests.post(API_URL, json=input_data)
            if response.status_code == 200:
                result = response.json()

                # ML Results
                ml = result["ml_prediction"]
                score = result["score_prediction"]
                summary = result.get("final_summary", {})

                st.success("✅ Analysis Complete")

                st.subheader("📊 ML Model Prediction")
                st.metric("Heart Disease Probability", ml["heart_disease_probability"])
                st.metric("No Disease Probability", ml["no_disease_probability"])
                st.write(f"**AI Insight:** {ml['ml_risk_message']}")

                st.subheader("🧮 Rule-Based Scoring System")
                st.metric("Risk Score", score["risk_score"])
                st.write(f"**Risk Level:** {score['risk_level']}")
                st.write("**Flagged Conditions:**")
                st.markdown(
                    "\n".join([f"- {flag}" for flag in score["threshold_flags"]])
                )

                if summary:
                    st.subheader("💡 Final Summary")
                    st.write(f"**Overall Status:** {summary.get('overall_status', 'N/A')}")
                    st.write(f"**Recommendation:** {summary.get('recommendation', 'N/A')}")

                st.info(result["final_advice"])

            else:
                st.error(f"API Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"❌ Error connecting to API: {e}")


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Developed by Sparsh | Powered by FastAPI + Streamlit + AWS 💻")
