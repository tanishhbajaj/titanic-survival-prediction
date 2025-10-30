# 🚢 Titanic Survival Prediction (Machine Learning + Streamlit)

An interactive web app that predicts whether a Titanic passenger would have survived — built with **Python, Scikit-learn, and Streamlit**.

---

## 🎯 Project Overview

This project applies **data science and machine learning** techniques to the famous [Titanic dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) to predict passenger survival based on key factors such as age, sex, passenger class, and fare.

It includes:
- Data cleaning and preprocessing (handling missing values, encoding categorical features)
- Model training with **Random Forest Classifier**
- Deployment using **Streamlit** for a live, interactive prediction experience
- Embedded **Jupyter Notebook** to view full preprocessing and model steps

---

## 🧠 Model Workflow

1. **Data Preprocessing**
   - Filled missing `Age` values with mean
   - Dropped `Cabin` column (too many missing values)
   - Encoded `Sex` and `Embarked`
2. **Feature Scaling**
   - Standardized numeric features (`Age`, `Fare`)
3. **Model Training**
   - Used Random Forest Classifier (`n_estimators=100`)
   - Achieved ~81% accuracy
4. **Deployment**
   - Streamlit app for real-time survival prediction

---

## 🧩 Files in this Repo

| File | Description |
|------|--------------|
| `app.py` | Streamlit web app |
| `train_model.py` | Model training & preprocessing script |
| `requirements.txt` | Python dependencies |
| `models/` | Saved model and scaler (`.pkl` files) |
| `data/` | Titanic dataset |
| `notebooks/` | Full Jupyter Notebook version |

---

## 🌐 View Full Notebook

Explore the complete model training, evaluation, and preprocessing steps here:

🔗 [Open Notebook on nbviewer](https://nbviewer.org/github/tanishhbajaj/titanic-survival-prediction/blob/main/notebooks/Titanic_model.ipynb)

---

## 🚀 Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/tanishhbajaj/titanic-survival-prediction.git
   cd titanic-survival-prediction
