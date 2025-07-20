# debug_recommender.py
# Save this as a separate file in the same directory as your main code

import sys
import os
import logging
from typing import List, Dict, Tuple
import yfinance as yf
from datetime import datetime, timedelta

# Add the path to your main code if needed
# sys.path.append('/path/to/your/main/code')

# Import your EnhancedStockRecommender class
# Replace 'enhanced_stock_recommender' with your actual file name (without .py)
from enhanced_stock_recommender import EnhancedStockRecommender

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_stock_recommender():
    """
    Debug the stock recommender to identify why same stocks are always returned
    """
    print("🔍 DEBUGGING STOCK RECOMMENDER")
    print("=" * 50)
    
    recommender = EnhancedStockRecommender()
    
    # Test multiple scenarios
    test_cases = [
        (50000, 10, 25, 5000, 200),    # Conservative
        (100000, 5, 65, 10000, 1000),  # Aggressive  
        (75000, 7, 45, 8000, 500),     # Moderate
    ]
    
    for i, (target, timeframe, risk, current, monthly) in enumerate(test_cases, 1):
        print(f"\n🧪 TEST CASE {i}")
        print(f"Target: £{target:,}, Timeframe: {timeframe}y, Risk: {risk}")
        print("-" * 30)
        
        # Step 1: Check risk category mapping
        risk_category = recommender.risk_score_to_category(risk)
        print(f"✓ Risk Category: {risk_category}")
        
        # Step 2: Test market regime detection
        try:
            market_regime = recommender.detect_market_regime()
            print(f"✓ Market Regime: {market_regime['regime']} (confidence: {market_regime['confidence']:.1%})")
        except Exception as e:
            print(f"❌ Market Regime Detection Failed: {e}")
            market_regime = None
        
        # Step 3: Check asset universe
        if risk_category in recommender.asset_universes:
            universe = recommender.asset_universes[risk_category]
            print(f"✓ Asset Universe Found: {list(universe.keys())}")
            
            # Show actual stocks in universe
            all_stocks = []
            for category, items in universe.items():
                if isinstance(items, list):
                    all_stocks.extend(items)
            print(f"  All stocks in universe: {all_stocks}")
        else:
            print(f"❌ No universe for {risk_category}")
        
        # Step 4: Test expanded universe
        try:
            expanded_universe = recommender._get_expanded_universe(risk_category)
            print(f"✓ Expanded Universe ({len(expanded_universe)} stocks): {expanded_universe}")
        except Exception as e:
            print(f"❌ Expanded Universe Failed: {e}")
        
        # Step 5: Test factor analysis on a few stocks
        test_stocks = ["VTI", "QQQ", "ARKK"]
        print(f"📊 Testing Factor Analysis:")
        
        for stock in test_stocks:
            try:
                factors = recommender.analyze_stock_factors(stock)
                print(f"  {stock}: Composite Score = {factors['composite']:.3f}")
            except Exception as e:
                print(f"  {stock}: ❌ Factor Analysis Failed: {e}")
        
        # Step 6: Test stock validation
        print("🔍 Testing Stock Validation:")
        validation_test_stocks = ["VTI", "INVALID_TICKER", "QQQ", "FAKE123"]
        valid_stocks = recommender.validate_and_filter_stocks(validation_test_stocks)
        print(f"  Input: {validation_test_stocks}")
        print(f"  Valid: {valid_stocks}")
        
        # Step 7: Get final recommendation with detailed logging
        try:
            print("🎯 Getting Final Recommendation...")
            final_recommendation = recommender.recommend_stocks(target, timeframe, risk, current, monthly)
            print(f"🎯 Final Recommendation: {final_recommendation}")
        except Exception as e:
            print(f"❌ Final Recommendation Failed: {e}")
        
        print()

def test_market_data_access():
    """Test if market data is accessible"""
    print("\n📡 TESTING MARKET DATA ACCESS")
    print("=" * 30)
    
    test_tickers = ["SPY", "VTI", "QQQ", "BND", "ARKK", "TSLA", "AAPL"]
    
    for ticker in test_tickers:
        try:
            # Test Yahoo Finance access
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="5d")  # Try shorter period
            
            if len(hist) > 0 and info.get('regularMarketPrice'):
                print(f"✓ {ticker}: OK (Price: ${info.get('regularMarketPrice', 'N/A')})")
            else:
                print(f"❌ {ticker}: No data available")
                
        except Exception as e:
            print(f"❌ {ticker}: Error - {str(e)[:100]}")

def check_asset_universes():
    """Check if asset universes contain valid tickers"""
    print("\n🌍 CHECKING ASSET UNIVERSES")
    print("=" * 30)
    
    recommender = EnhancedStockRecommender()
    
    for risk_level, universe in recommender.asset_universes.items():
        print(f"\n📈 {risk_level.upper()}:")
        
        total_stocks = 0
        valid_stocks = 0
        
        for category, tickers in universe.items():
            if isinstance(tickers, list):
                print(f"  {category}: {tickers}")
                total_stocks += len(tickers)
                
                # Test a few tickers from each category
                for ticker in tickers[:2]:  # Test first 2 from each category
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        if info.get('regularMarketPrice'):
                            print(f"    ✓ {ticker}: Valid")
                            valid_stocks += 1
                        else:
                            print(f"    ❌ {ticker}: No price data")
                    except Exception as e:
                        print(f"    ❌ {ticker}: Invalid - {str(e)[:50]}")
        
        print(f"  Summary: {valid_stocks}/{min(total_stocks, len([t for cat in universe.values() if isinstance(cat, list) for t in cat[:2]]))} tested stocks are valid")

def simple_test():
    """Simple test to isolate the issue"""
    print("\n🔬 SIMPLE ISOLATION TEST")
    print("=" * 25)
    
    recommender = EnhancedStockRecommender()
    
    # Test with one simple case
    target = 50000
    timeframe = 10 
    risk = 60  # Should be moderate_aggressive
    
    print(f"Input: £{target:,} in {timeframe} years, risk {risk}")
    
    # Check each step manually
    print(f"1. Risk category: {recommender.risk_score_to_category(risk)}")
    
    try:
        result = recommender.recommend_stocks(target, timeframe, risk)
        print(f"2. Final result: {result}")
    except Exception as e:
        print(f"2. Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Debug Analysis...")
    print("="*60)
    
    # Run tests in order
    simple_test()
    test_market_data_access() 
    check_asset_universes()
    debug_stock_recommender()
    
    print("\n" + "="*60)
    print("🔍 DEBUG COMPLETE")
    print("="*60)
    print("\n💡 If you see repeated failures:")
    print("  - Check internet connection")
    print("  - Verify yfinance is installed: pip install yfinance")
    print("  - Some tickers might be invalid/delisted")
    print("  - Try running with different risk scores")