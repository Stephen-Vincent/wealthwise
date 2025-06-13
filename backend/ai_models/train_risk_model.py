from imblearn.over_sampling import SMOTE
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report

# Load the data
data_path = os.path.join(os.path.dirname(__file__), "training_data", "risk_assessment_dataset.csv")
df = pd.read_csv(data_path)

# Define features and target
X = df.drop(columns=["risk_score"])

def bin_risk_score(score):
    if score <= 20:
        return "Low"
    elif score <= 60:
        return "Medium"
    else:
        return "High"

y = df["risk_score"].apply(bin_risk_score)

# List categorical and numerical features
categorical_features = ["investment_goal", "timeframe", "income"]
numerical_features = ["years_of_experience", "target_amount", "lump_sum_investment", "monthly_investment"]

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features),
    ]
)

# Create the pipeline
# model = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("classifier", RandomForestClassifier(
#         n_estimators=300,
#         max_depth=10,
#         min_samples_split=5,
#         min_samples_leaf=2,
#         class_weight='balanced',
#         random_state=42
#     ))
# ])

# Use Gradient Boosting Classifier instead
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Fit preprocessor on training data and transform
X_train_transformed = preprocessor.fit_transform(X_train)

# Apply SMOTE to the transformed data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

# Fit only the classifier on the resampled data
model.named_steps["classifier"].fit(X_train_resampled, y_train_resampled)

# Replace the classifier in the pipeline with the trained one
model.steps[-1] = ("classifier", model.named_steps["classifier"])

# Evaluate model
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)

# Predict risk labels and also get probabilities
y_proba = model.predict_proba(X_test)

# Map labels back to numerical risk scores (optional - here we use median values)
label_to_score = {"Low": 10, "Medium": 40, "High": 80}
risk_scores = [label_to_score[label] for label in y_pred]

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix visualization
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Show a few sample outputs
for i in range(min(5, len(y_pred))):
    print(f"Predicted Label: {y_pred[i]}, Estimated Risk Score: {risk_scores[i]}")

# Save the full pipeline (preprocessor + model)
model_path = os.path.join(os.path.dirname(__file__), "risk_model", "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully.")
