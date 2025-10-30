# train_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs("models", exist_ok=True)

# load
train = pd.read_csv("https://raw.githubusercontent.com/tanishhbajaj/titanic-survival-prediction/refs/heads/main/train.csv")

# preprocessing (match your notebook exactly)
train = train.drop('Cabin', axis=1)
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex']).astype(int)
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True, dtype=int)

X = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']]
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# scaler for numeric features
scaler = StandardScaler()
X_train[['Age','Fare']] = scaler.fit_transform(X_train[['Age','Fare']])
X_test[['Age','Fare']] = scaler.transform(X_test[['Age','Fare']])

# train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# save model and scaler
joblib.dump(rf, "models/titanic_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Saved models/titanic_model.pkl and models/scaler.pkl")
