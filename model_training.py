import pickle
from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()

model = LogisticRegression(random_state=42, max_iter=500)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
