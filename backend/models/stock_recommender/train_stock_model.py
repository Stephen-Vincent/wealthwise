import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hamming_loss, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Import volatility calculation
from volatility_detector import calculate_volatility

from utils import remap_ticker
# Ensure the ticker remapping utility is available

# Path to your dataset
data_path = 'backend/data/stocks_training_data.csv'


# Load data
df = pd.read_csv(data_path)

# Clean tickers by removing dots
df['recommended_stocks'] = df['recommended_stocks'].str.replace('.', '', regex=False)

import numpy as np
np.random.seed(42)
df['age'] = np.random.randint(20, 70, size=len(df))
df['risk_tolerance'] = np.random.choice(['low', 'medium', 'high'], size=len(df))
df['income_bracket'] = np.random.choice(['low', 'medium', 'high'], size=len(df))

# Clean and standardize 'goal' and 'timeframe_years' into categorical types
df['goal'] = df['goal'].fillna('general').str.lower().astype(str)
df['timeframe'] = df['timeframe_years'].fillna(5).apply(lambda x: 'short_term' if x <= 3 else 'mid_term' if x <= 7 else 'long_term').astype(str)

# Define a mapping from timeframe to estimated investment horizon in months
timeframe_map = {
    'short_term': 12,
    'mid_term': 36,
    'long_term': 60
}
df['investment_horizon'] = df['timeframe'].map(timeframe_map).fillna(36)

# Ensure all expected columns exist
expected_columns = ['initial_lump_sum', 'monthly_contribution']
for col in expected_columns:
    if col not in df.columns:
        df[col] = 0

 # Derived features
df['monthly_contribution_total'] = df['monthly_contribution'] * df['investment_horizon']
df['total_projected_portfolio'] = df['initial_lump_sum'] + df['monthly_contribution_total']

# Add a placeholder for volatility
df['volatility'] = 0.0


# Helper to validate ticker symbols
VALID_ONE_LETTER_TICKERS = {'F', 'T', 'K', 'X', 'O'}

def is_valid_ticker(ticker):
    return len(ticker) > 1 or ticker in VALID_ONE_LETTER_TICKERS

# Replace placeholder with real volatility values
for idx, row in df.iterrows():
    tickers = row['recommended_stocks']
    if isinstance(tickers, str):
        tickers = [remap_ticker(t.strip()) for t in tickers.split(',') if is_valid_ticker(remap_ticker(t.strip()))]
    elif isinstance(tickers, list):
        tickers = [remap_ticker(t.strip()) for t in tickers if is_valid_ticker(remap_ticker(t.strip()))]
    else:
        tickers = []

    # Optional: print warning for suspicious ticker sets
    if any(len(t) == 1 and t not in VALID_ONE_LETTER_TICKERS for t in tickers):
        print(f"⚠️  Unusual tickers at row {idx}: {tickers}")

    volatilities = [calculate_volatility(ticker) for ticker in tickers if ticker]
    df.at[idx, 'volatility'] = np.mean(volatilities) if volatilities else 0.0

# Binary flags
df['is_short_term'] = (df['investment_horizon'] <= 12).astype(int)
df['is_long_term'] = (df['investment_horizon'] > 36).astype(int)

# Ensure all required numeric columns exist
required_columns = ['experience', 'initial_lump_sum', 'monthly_contribution', 'target_value']
for col in required_columns:
    if col not in df.columns:
        df[col] = 0

# Normalize numeric columns first
numeric_features = [
    'volatility',
    'experience', 'initial_lump_sum', 'monthly_contribution', 'target_value',
    'investment_horizon', 'monthly_contribution_total',
    'total_projected_portfolio'
]
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Convert comma-separated stock strings to list
df['recommended_stocks'] = df['recommended_stocks'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

from collections import Counter

all_labels = [stock.strip() for sublist in df['recommended_stocks'] for stock in sublist]
label_counts = Counter(all_labels)

print("📊 Label frequencies:")
for label, count in label_counts.most_common():
    print(f"{label}: {count}")

min_count = 40
frequent_labels = {label for label, count in label_counts.items() if count >= min_count}

df['recommended_stocks'] = df['recommended_stocks'].apply(
    lambda labels: [label for label in labels if label in frequent_labels]
)
df = df[df['recommended_stocks'].map(len) > 0]

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['recommended_stocks'])

# One-hot encode categorical features
df_encoded = pd.get_dummies(
    df.drop('recommended_stocks', axis=1),
    columns=['goal', 'timeframe', 'risk_tolerance', 'income_bracket'],
    drop_first=True
)

X = df_encoded

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

# Handle infinite and NaN values in features
import numpy as np
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Initialize and train the multi-label classifier using OneVsRest strategy with XGBoost
base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model = OneVsRestClassifier(base_model)
model.fit(X_train, y_train)

# Predict probabilities for test set
y_pred_proba = model.predict_proba(X_test)
threshold = 0.3
y_pred = (y_pred_proba > threshold).astype(int)

from sklearn.metrics import precision_recall_curve
thresholds = []
for i in range(y_test.shape[1]):
    precision, recall, th = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = th[np.argmax(f1[:-1])]
    thresholds.append(best_threshold)

# Apply optimized thresholds per label
y_pred_custom = np.array([
    (y_pred_proba[:, i] > thresholds[i]).astype(int)
    for i in range(y_pred_proba.shape[1])
]).T

# Generate classification report
report = classification_report(y_test, y_pred_custom, target_names=mlb.classes_, output_dict=True)
import pandas as pd
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('backend/models/stock_recommender/classification_report.csv')
print(report_df)


# Print evaluation metrics
print("Hamming Loss:", hamming_loss(y_test, y_pred_custom))
print("Subset Accuracy:", accuracy_score(y_test, y_pred_custom))

# Save model and label binarizer together in one file
model_bundle_path = 'backend/models/stock_recommender/stock_recommender_model.pkl'
os.makedirs(os.path.dirname(model_bundle_path), exist_ok=True)
joblib.dump((model, mlb), model_bundle_path)
print("✅ Stock recommendation model and label binarizer saved together successfully.")

# Generate stock name map from trained classes
import yfinance as yf
import json

stock_name_map = {}

for ticker in mlb.classes_:
    try:
        info = yf.Ticker(ticker).info
        long_name = info.get('longName', ticker)
        stock_name_map[ticker] = long_name
    except Exception as e:
        print(f"⚠️ Could not fetch data for {ticker}: {e}")
        stock_name_map[ticker] = ticker  # Fallback to ticker name

name_map_path = 'backend/models/stock_recommender/stock_name_map.json'
with open(name_map_path, 'w') as f:
    json.dump(stock_name_map, f, indent=4)

print(f"✅ Generated and saved stock name map to {name_map_path}")

# The model and MultiLabelBinarizer are saved together for inference

# Plot F1 scores per label
f1_scores = report_df.loc[mlb.classes_, 'f1-score']
f1_scores.sort_values().plot(kind='barh', figsize=(10, 12))
plt.xlabel('F1 Score')
plt.title('F1 Score per Label')
plt.tight_layout()
plt.savefig('backend/models/stock_recommender/f1_scores.png')
plt.show()
