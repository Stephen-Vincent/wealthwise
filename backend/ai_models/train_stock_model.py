import os
import pandas as pd

def train_and_recommend(target_value: float, timeframe: int, risk_score: float):
    # Load the data
    data_path = os.path.join(os.path.dirname(__file__), "training_data", "portfolio_stock_recommendation_dataset.csv")
    df = pd.read_csv(data_path)

    print("Preview of data:")
    print(df.head())
    print("\nColumn names:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    import joblib

    df.dropna(inplace=True)

    # Use only the 3 input columns as features
    feature_columns = ["target_value", "timeframe", "risk_score"]
    target_column = "risk_score"  # We will predict risk_score here but it's also input for recommendation mapping

    if not all(col in df.columns for col in feature_columns + ["recommended_stocks"]):
        raise ValueError("Dataset missing required columns.")

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nR^2 Score:", r2_score(y_test, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    # Create dataframe from input parameters for prediction
    input_df = pd.DataFrame([{
        "target_value": target_value,
        "timeframe": timeframe,
        "risk_score": risk_score
    }])

    predicted_risk_score = model.predict(input_df)[0]

    # Find closest matching row by predicted risk_score
    closest_row = df.iloc[(df["risk_score"] - predicted_risk_score).abs().argsort()[:1]]
    recommended_stocks_str = closest_row["recommended_stocks"].values[0]
    recommended_stocks = [ticker.strip() for ticker in recommended_stocks_str.split(",")]

    model_path = os.path.join(os.path.dirname(__file__), "stock_model/portfolio_stock_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return recommended_stocks

if __name__ == "__main__":
    # For demonstration, pick the first row from dataset and pass its values
    data_path = os.path.join(os.path.dirname(__file__), "training_data", "portfolio_stock_recommendation_dataset.csv")
    df = pd.read_csv(data_path)
    first_row = df.iloc[0]
    stocks = train_and_recommend(
        target_value=first_row["target_value"],
        timeframe=int(first_row["timeframe"]),
        risk_score=float(first_row["risk_score"])
    )
    print("Recommended Stocks:", ", ".join(stocks))