# backend/models/risk_analysis/train_risk_model.py

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load your dataset
data_path = "data/risk/sample_risk_dataset.csv"
df = pd.read_csv(data_path)

# Define features and label
X = df.drop(columns=["risk_profile"])
y = df["risk_profile"]

# One-hot encode categorical features
X = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Ensure the model directory exists
model_dir = "backend/models/risk_analysis"
os.makedirs(model_dir, exist_ok=True)

# Save the model
joblib.dump(model, os.path.join(model_dir, "risk_model.pkl"))
print("✅ Risk model saved successfully.")