import pickle
import streamlit as st
from data_preprocessing import prepare_prediction_input

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600&display=swap');

    .stApp, .stApp * { font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background-color: #f5f0e8; }

    .main .block-container { padding: 2.5rem 3rem; max-width: 720px; }

    h1 {
        font-weight: 600;
        color: #4a4540 !important;
        letter-spacing: -0.02em;
    }

    label, p, .stNumberInput label, .stSelectbox label {
        color: #4A4A4A !important;
    }

    .stNumberInput input,
    .stNumberInput > div > div,
    .stSelectbox > div > div {
        background-color: #faf8f5 !important;
        border: 1px solid #c4beb5 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    .stNumberInput input {
        color: #5A5A5A !important;
    }
    .stSelectbox > div > div > div,
    .stSelectbox [data-baseweb="select"] > div {
        color: #5A5A5A !important;
    }
    .stSelectbox [data-baseweb="select"] {
        color: #5A5A5A !important;
    }
    .stSelectbox [role="listbox"],
    .stSelectbox [role="option"] {
        background-color: #faf8f5 !important;
        color: #5A5A5A !important;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 1.25rem;
    }
    .stNumberInput input::placeholder {
        color: #9a958d !important;
    }
    [data-testid="stHelp"] {
        color: #6a6a6a !important;
    }

    div[data-testid="stButton"] > button {
        background-color: #8fa89a !important;
        color: #f8f6f2 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
        transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #9fb5a8 !important;
        box-shadow: 0 3px 8px rgba(0,0,0,0.08) !important;
    }

    .result-box {
        padding: 1.25rem;
        border-radius: 10px;
        background-color: #ebe7e0;
        margin-top: 1.25rem;
        font-size: 1.1rem;
        color: #4a4540;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)

st.title("Titanic Survival Prediction")

col1, col2 = st.columns(2)

FARE_OPTIONS = {
    "0–10": 5,
    "10–25": 17.5,
    "25–50": 37.5,
    "50–100": 75,
    "100+": 150,
}

with col1:
    age = st.number_input("Age", min_value=0, max_value=100, value=30, help="Enter age (e.g., 29)")
    sex = st.selectbox("Sex", ["male", "female"], help="Select gender")

with col2:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], help="Choose class (1, 2, 3)")
    fare_label = st.selectbox("Fare", list(FARE_OPTIONS.keys()), help="Select fare range")

if st.button("Predict Survival"):
    fare = FARE_OPTIONS[fare_label]
    X_pred = prepare_prediction_input(age, sex, pclass, fare)

    with open('titanic_model.pkl', 'rb') as f:
        model = pickle.load(f)

    pred = model.predict(X_pred)[0]
    result = "Survived" if pred == 1 else "Did Not Survive"
    st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
