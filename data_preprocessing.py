import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
FEATURE_COLUMNS = ["Age", "Sex", "Pclass", "Fare"]
SEX_ENCODING = {"male": 0, "female": 1}

def load_and_preprocess():
    df = pd.read_csv(TITANIC_URL)
    df = df[['Age', 'Sex', 'Pclass', 'Fare', 'Survived']]

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    df['Sex'] = df['Sex'].map(SEX_ENCODING)

    X = df[FEATURE_COLUMNS]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def prepare_prediction_input(age, sex, pclass, fare):
    sex_enc = SEX_ENCODING[sex]
    return pd.DataFrame([[age, sex_enc, pclass, fare]], columns=FEATURE_COLUMNS)
