# Add this test function to your portfolio_simulator.py

def test_stock_data_quality(tickers: List[str], timeframe: int) -> Dict[str, Any]:
    """
    Test the quality of downloaded stock data to identify issues.
    Call this before running simulations to catch data problems.
    """
    try:
        logger.info(f"🧪 Testing data quality for {tickers}")
        
        # Download data
        data = download_stock_data(tickers, timeframe)
        
        results = {
            "data_shape": data.shape,
            "columns": list(data.columns),
            "date_range": f"{data.index[0]} to {data.index[-1]}",
            "issues": []
        }
        
        # Check for missing data
        null_counts = data.isnull().sum()
        if null_counts.any():
            results["issues"].append(f"Missing data: {dict(null_counts[null_counts > 0])}")
        
        # Check for zero values
        zero_counts = (data == 0).sum()
        if zero_counts.any():
            results["issues"].append(f"Zero values: {dict(zero_counts[zero_counts > 0])}")
        
        # Check first row (used for normalization)
        first_row = data.iloc[0]
        if (first_row == 0).any():
            problematic = first_row[first_row == 0]
            results["issues"].append(f"Zero starting prices: {dict(problematic)}")
        
        # Check for extreme outliers
        for col in data.columns:
            daily_returns = data[col].pct_change().dropna()
            extreme_moves = daily_returns[abs(daily_returns) > 0.5]  # >50% daily moves
            if len(extreme_moves) > 0:
                results["issues"].append(f"{col} has {len(extreme_moves)} extreme daily moves")
        
        # Data quality score
        total_issues = len(results["issues"])
        results["quality_score"] = max(0, 100 - (total_issues * 20))
        results["quality_status"] = "GOOD" if total_issues == 0 else "ISSUES_FOUND"
        
        logger.info(f"📊 Data quality: {results['quality_status']} (score: {results['quality_score']})")
        for issue in results["issues"]:
            logger.warning(f"⚠️ {issue}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Data quality test failed: {e}")
        return {"error": str(e), "quality_status": "TEST_FAILED"}

# Call this in your simulate_portfolio function before running the simulation:
# data_quality = test_stock_data_quality(tickers, timeframe)
# if data_quality["quality_status"] != "GOOD":
#     logger.warning(f"⚠️ Data quality issues detected: {data_quality['issues']}")
#     # You might want to filter out problematic tickers or use fallback