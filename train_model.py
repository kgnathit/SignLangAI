"""
Step 2: Train the sign-language recognition model.
"""

import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_FILE = "hand_data.csv"
MODEL_FILE = "sign_model.pkl"
LABEL_FILE = "label_map.json"

data = pd.read_csv(DATA_FILE)
X = data.drop(columns=["label"]).values
y = data["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print(classification_report(y_test, predictions))

joblib.dump(model, MODEL_FILE)
labels = sorted(list(set(y)))
with open(LABEL_FILE, "w") as file:
    json.dump({str(i): label for i, label in enumerate(labels)}, file)
print(f"Saved {MODEL_FILE} and {LABEL_FILE}")
