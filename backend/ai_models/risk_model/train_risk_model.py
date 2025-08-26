"""
train_risk_model.py — Train and persist the WealthWise risk scoring model.

Overview
--------
This script trains a supervised regression model that predicts a user's numeric
risk score (1–100) from questionnaire + financial features. The target
`risk_score` is taken directly from the dataset—no recalculation is performed.

Data
----
- Input CSV: ./training_data/complete_risk_dataset.csv
- Target (y): risk_score
- Features (X):
  • Categorical: loss_tolerance, panic_behavior, financial_behavior,
                 engagement_level, investment_goal, income
  • Numerical:   years_of_experience, timeframe, target_amount,
                 lump_sum_investment, monthly_investment

Pipeline
--------
- Preprocessing via ColumnTransformer:
  • OneHotEncoder(handle_unknown="ignore") for categorical columns
  • StandardScaler() for numerical columns
- Estimator: XGBRegressor (n_estimators=200, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, random_state=42)
- All steps are wrapped in a single scikit-learn Pipeline so that training and
  inference use identical transformations.

Training & Evaluation
---------------------
- Train/test split: 80/20, random_state=42
- Metrics (on the test set): MSE and R² (reported on predictions clamped to
  [1, 100] for safe display). Raw prediction range is printed for diagnostics.
- The script also:
  • Prints NaN/Inf counts for basic data health checks
  • Shows five sample predictions spanning the distribution
  • Buckets predicted scores into labels:
      Ultra Conservative (<30), Conservative (30–<50), Moderate (50–<70),
      Moderate Aggressive (70–<85), Aggressive (>=85)
  • Logs the label distribution for sanity-checking class balance
  • Runs two "extreme profile" predictions (very conservative vs very aggressive)
    to verify the model spans the risk spectrum

Output
------
- Persists the trained Pipeline (preprocessing + model) to:
    ./enhanced_model.pkl
  Consumers can load this pickle and call `predict()` directly with a DataFrame
  containing the same column schema as the training data.

Maintenance Notes
-----------------
- If the dataset schema changes (column names or categories), update the
  `categorical_features` and `numerical_features` lists accordingly.
- `RandomForestRegressor` and `math` were previously imported but unused and
  have been removed in this version.
"""
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# FIXED: Use correct path where data generator saves the file
data_path = os.path.join(os.path.dirname(__file__), "training_data", "complete_risk_dataset.csv")

print(f"📁 Loading data from: {data_path}")
df = pd.read_csv(data_path)

print(f"✅ Loaded training data: {len(df)} rows")
print(f"📊 Risk score range in data: {df['risk_score'].min():.1f} - {df['risk_score'].max():.1f}")
print(f"📈 Mean risk score in data: {df['risk_score'].mean():.1f}")

# The model's objective is to predict the pre-calculated risk scores
# DO NOT recalculate risk scores - use the ones from the dataset

# Target column: risk_score (use as-is from dataset)
y = df["risk_score"]

# Features (drop the target)
X = df.drop(columns=["risk_score"])

# Identify categorical and numerical features
categorical_features = [
    "loss_tolerance", "panic_behavior", "financial_behavior", 
    "engagement_level", "investment_goal", "income"
]
numerical_features = [
    "years_of_experience", "timeframe", "target_amount", 
    "lump_sum_investment", "monthly_investment"
]

print(f"🔍 Categorical features: {categorical_features}")
print(f"🔍 Numerical features: {numerical_features}")

# Preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features),
    ]
)

print("🚨 Checking for NaNs or Infs in training data:")
print(df.isnull().sum())
print(np.isinf(df.select_dtypes(include=[np.number])).sum())

# Enhanced model with better hyperparameters for full range
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=200,       # More trees for better fit
        learning_rate=0.05,     # Lower learning rate for stability
        max_depth=6,           # Deeper trees for complex patterns
        subsample=0.8,         # Prevent overfitting
        colsample_bytree=0.8,  # Feature sampling
        random_state=42
    ))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📚 Training set: {len(X_train)} samples")
print(f"🧪 Test set: {len(X_test)} samples")

# Fit the model
print("🔄 Training model...")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

# DO NOT clamp predictions - let the model use the full range
print("🧠 Raw predictions range:", y_pred.min(), "-", y_pred.max())

# Only clamp to valid bounds (1-100) if absolutely necessary
y_pred_clamped = np.clip(y_pred, 1, 100)

# UPDATED risk label function to match new ranges
def get_risk_label(score):
    if score < 30:
        return "Ultra Conservative"
    elif score < 50:
        return "Conservative" 
    elif score < 70:
        return "Moderate"
    elif score < 85:
        return "Moderate Aggressive"
    else:
        return "Aggressive"

# Print sample predictions across the range
print("\n🎯 Sample Predictions Across Risk Spectrum:")
test_indices = np.argsort(y_pred_clamped)  # Sort by predicted score
sample_indices = [
    test_indices[0],           # Lowest
    test_indices[len(test_indices)//4],    # 25th percentile
    test_indices[len(test_indices)//2],    # Median
    test_indices[3*len(test_indices)//4],  # 75th percentile
    test_indices[-1]           # Highest
]

for i, idx in enumerate(sample_indices):
    actual = y_test.iloc[idx]
    predicted = y_pred_clamped[idx]
    risk_label = get_risk_label(predicted)
    print(f"   Sample {i+1}: Actual={actual:.1f}, Predicted={predicted:.1f}, Label={risk_label}")

print(f"\n📊 Model Performance:")
print(f"   Mean Squared Error: {mean_squared_error(y_test, y_pred_clamped):.2f}")
print(f"   R² Score: {r2_score(y_test, y_pred_clamped):.3f}")
print(f"   Mean Predicted Risk Score: {np.mean(y_pred_clamped):.1f}")
print(f"   Risk Score Range: {np.min(y_pred_clamped):.1f} - {np.max(y_pred_clamped):.1f}")

# UPDATED prediction distribution with new ranges
pred_ultra_conservative = np.sum(y_pred_clamped < 30)
pred_conservative = np.sum((y_pred_clamped >= 30) & (y_pred_clamped < 50))
pred_moderate = np.sum((y_pred_clamped >= 50) & (y_pred_clamped < 70))
pred_moderate_aggressive = np.sum((y_pred_clamped >= 70) & (y_pred_clamped < 85))
pred_aggressive = np.sum(y_pred_clamped >= 85)

print(f"\n🎯 Prediction Distribution:")
print(f"   Ultra Conservative (1-30): {pred_ultra_conservative} ({pred_ultra_conservative/len(y_pred_clamped)*100:.1f}%)")
print(f"   Conservative (30-50): {pred_conservative} ({pred_conservative/len(y_pred_clamped)*100:.1f}%)")
print(f"   Moderate (50-70): {pred_moderate} ({pred_moderate/len(y_pred_clamped)*100:.1f}%)")
print(f"   Moderate Aggressive (70-85): {pred_moderate_aggressive} ({pred_moderate_aggressive/len(y_pred_clamped)*100:.1f}%)")
print(f"   Aggressive (85-100): {pred_aggressive} ({pred_aggressive/len(y_pred_clamped)*100:.1f}%)")

# Save model
model_path = os.path.join(os.path.dirname(__file__), "enhanced_model.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n✅ Enhanced risk model saved to: {model_path}")

# Test with sample extreme cases to verify full range
print(f"\n🧪 Testing Extreme Cases:")

# Ultra Conservative Test Case
ultra_conservative_test = pd.DataFrame([{
    'years_of_experience': 0,
    'loss_tolerance': 'sell_immediately',
    'panic_behavior': 'yes_always',
    'financial_behavior': 'save_all',
    'engagement_level': 'rarely',
    'investment_goal': 'emergency fund',
    'target_amount': 5000,
    'lump_sum_investment': 2000,
    'monthly_investment': 50,
    'timeframe': 1,
    'income': 'low'
}])

# Ultra Aggressive Test Case  
ultra_aggressive_test = pd.DataFrame([{
    'years_of_experience': 25,
    'loss_tolerance': 'buy_more',
    'panic_behavior': 'no_never',
    'financial_behavior': 'invest_all',
    'engagement_level': 'daily',
    'investment_goal': 'retirement',
    'target_amount': 2000000,
    'lump_sum_investment': 500000,
    'monthly_investment': 8000,
    'timeframe': 30,
    'income': 'high'
}])

ultra_conservative_pred = model.predict(ultra_conservative_test)[0]
ultra_aggressive_pred = model.predict(ultra_aggressive_test)[0]

print(f"   Ultra Conservative Profile: {ultra_conservative_pred:.1f} (Expected: 5-25)")
print(f"   Ultra Aggressive Profile: {ultra_aggressive_pred:.1f} (Expected: 85-95)")

if ultra_conservative_pred < 35 and ultra_aggressive_pred > 70:
    print("✅ Model successfully spans risk spectrum!")
else:
    print("⚠️  Model may need adjustment for full range coverage")

print(f"\n🎉 Training complete! Model ready for production use.")