import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load Model and Scaler ---
model = joblib.load("models/titanic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.write("This interactive web app predicts whether a passenger survived the Titanic disaster using a trained Random Forest model.")

st.markdown("---")

# --- Input Fields ---
st.subheader("🧍 Passenger Details")

pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=6, value=0)
fare = st.number_input("Fare (Ticket Price)", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["S", "Q"])

# --- Data Preparation ---
sex_encoded = 1 if sex == "Male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked_Q': [embarked_Q],
    'Embarked_S': [embarked_S]
})

# Scale numerical columns
input_data[['Age', 'Fare']] = scaler.transform(input_data[['Age', 'Fare']])

# --- Prediction Button ---
if st.button("🔍 Predict Survival"):
    prediction = model.predict(input_data)
    survival = "🟢 Survived" if prediction[0] == 1 else "🔴 Did Not Survive"

    st.subheader("Prediction Result:")
    st.success(f"**{survival}**")

st.markdown("---")

# --- Embedded Notebook Section ---
st.subheader("📘 View Full Data Science Notebook")
st.write("You can explore the complete model training, preprocessing, and evaluation below:")

notebook_url = "https://nbviewer.org/github/tanishhbajaj/titanic-survival-prediction/blob/main/Titanic_model.ipynb"

st.components.v1.iframe(notebook_url, height=600, scrolling=True)
