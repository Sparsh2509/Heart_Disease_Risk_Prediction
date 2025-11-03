import streamlit as st
import requests


# CONFIGURATION

API_URL = "http://heart-disease-api-env.eba-vrubpnic.ap-south-1.elasticbeanstalk.com/predict"

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("🩺 Heart Disease Prediction (Hybrid Model + Scoring)")
st.markdown("""
This app uses both an **ML model** and a **medical scoring system** to estimate your risk of heart disease.  
Fill in the details below and click **Predict Risk**.
""")


# INPUT FORM

with st.form("heart_form"):
    st.subheader("🔹 Patient Details")

    age = st.number_input("Age (years)", 10, 120, 45)
    sex = st.radio("Sex", ["Female (0)", "Male (1)"])
    cp = st.selectbox(
        "Chest Pain Type",
        [
            "0 - Typical Angina",
            "1 - Atypical Angina",
            "2 - Non-anginal Pain",
            "3 - Asymptomatic"
        ]
    )
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 250, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar >120 mg/dl", ["No (0)", "Yes (1)"])
    restecg = st.selectbox(
        "Resting ECG Results",
        [
            "0 - Normal",
            "1 - ST-T Wave Abnormality",
            "2 - Left Ventricular Hypertrophy"
        ]
    )
    thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
    exang = st.radio("Exercise Induced Angina", ["No (0)", "Yes (1)"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", ["0 - Upsloping", "1 - Flat", "2 - Downsloping"])
    ca = st.selectbox("Major Vessels Colored (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["0 - Normal", "1 - Fixed Defect", "2 - Reversible Defect", "3 - Others"])

    submitted = st.form_submit_button("🔍 Predict Risk")


# API CALL

if submitted:
    input_data = {
        "age": age,
        "sex": 1 if "Male" in sex else 0,
        "cp": int(cp.split(" - ")[0]),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if "Yes" in fbs else 0,
        "restecg": int(restecg.split(" - ")[0]),
        "thalach": thalach,
        "exang": 1 if "Yes" in exang else 0,
        "oldpeak": oldpeak,
        "slope": int(slope.split(" - ")[0]),
        "ca": int(ca),
        "thal": int(thal.split(" - ")[0])
    }

    with st.spinner("🧠 Analyzing your data..."):
        try:
            response = requests.post(API_URL, json=input_data)
            if response.status_code == 200:
                result = response.json()

                ml = result["ml_prediction"]
                score = result["score_prediction"]
                summary = result.get("final_summary", {})

                st.success("✅ Analysis Complete")

                # ML Prediction
                st.subheader("📊 ML Model Prediction")
                st.metric("Heart Disease Probability", ml["heart_disease_probability"])
                st.metric("No Heart Disease Probability", ml["no_disease_probability"])
                st.write(f"**AI Insight:** {ml['ml_risk_message']}")

                # Scoring System 
                st.subheader("🧮 Medical Scoring System")
                st.metric("Risk Score", score["risk_score"])
                st.write(f"**Risk Level:** {score['risk_level']}")
                st.write("**Flagged Conditions:**")
                st.markdown("\n".join([f"- {flag}" for flag in score["threshold_flags"]]))

                # Summary
                if summary:
                    st.subheader("💡 Final Summary")
                    st.write(f"**Overall Status:** {summary.get('overall_status', 'N/A')}")
                    st.write(f"**Recommendation:** {summary.get('recommendation', 'N/A')}")

                st.info(result["final_advice"])

            else:
                st.error(f"API Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"❌ Error connecting to API: {e}")


