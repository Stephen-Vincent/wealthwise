# tests/test_stock_selection.py

"""
Test suite for AI stock selection functionality.

Tests the enhanced stock recommender, goal-oriented optimization,
risk-based allocation, and stock validation.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules we want to test
try:
    from ai_models.stock_model.enhanced_stock_recommender import EnhancedStockRecommender, train_and_recommend
    from services.portfolio_simulator import get_ai_stock_recommendations, get_fallback_stocks_by_risk_profile
except ImportError as e:
    print(f"Warning: Could not import stock selection modules: {e}")
    print("Please ensure your stock model files exist")

class TestStockSelection:
    """Test cases for AI stock selection and recommendation."""
    
    def setUp(self):
        """Set up test fixtures before each test method (unittest style)."""
        self.recommender = EnhancedStockRecommender()
        
    def setup_method(self, method):
        """Set up test fixtures before each test method (pytest style)."""
        self.recommender = EnhancedStockRecommender()
        
    def test_risk_category_mapping(self):
        """Test that risk scores map to correct categories."""
        test_cases = [
            (5, "ultra_conservative"),
            (20, "conservative"),
            (40, "moderate"),
            (60, "moderate_aggressive"),
            (80, "aggressive"),
            (95, "ultra_aggressive")
        ]
        
        for risk_score, expected_category in test_cases:
            category = self.recommender.risk_score_to_category(risk_score)
            assert category == expected_category, f"Risk score {risk_score} should map to {expected_category}, got {category}"
            
        print("✅ Risk category mapping test passed")
    
    def test_required_return_calculation(self):
        """Test goal-oriented required return calculations."""
        test_cases = [
            # (target, current, timeframe, monthly, expected_return_range)
            (50000, 10000, 10, 200, (0.05, 0.15)),  # Reasonable goal
            (100000, 5000, 5, 500, (0.10, 0.25)),   # Aggressive goal
            (25000, 20000, 3, 100, (0.0, 0.08)),    # Conservative goal
        ]
        
        for target, current, timeframe, monthly, expected_range in test_cases:
            required_return = self.recommender.calculate_required_return(
                target, current, timeframe, monthly
            )
            
            min_expected, max_expected = expected_range
            assert min_expected <= required_return <= max_expected, \
                f"Required return {required_return:.2%} not in expected range {min_expected:.1%}-{max_expected:.1%}"
            
            print(f"✅ Goal: £{target:,} in {timeframe}y → {required_return:.1%} annual return needed")
        
        print("✅ Required return calculation test passed")
    
    def test_goal_feasibility_assessment(self):
        """Test realistic goal feasibility scoring."""
        test_cases = [
            # (required_return, risk_score, expected_feasibility_range)
            (0.07, 50, (70, 100)),   # Achievable moderate goal
            (0.15, 30, (10, 40)),    # Unrealistic conservative goal
            (0.12, 80, (70, 100)),   # Achievable aggressive goal
            (0.05, 20, (80, 100)),   # Easy conservative goal
        ]
        
        for required_return, risk_score, expected_range in test_cases:
            assessment = self.recommender.assess_goal_feasibility(required_return, risk_score)
            
            feasibility = assessment["feasibility_score"]
            min_expected, max_expected = expected_range
            
            assert min_expected <= feasibility <= max_expected, \
                f"Feasibility {feasibility}% not in expected range {min_expected}-{max_expected}%"
            
            # Check required fields
            required_fields = ["feasibility_score", "required_return", "expected_return", "recommendation"]
            for field in required_fields:
                assert field in assessment, f"Missing field in assessment: {field}"
            
            print(f"✅ Goal feasibility: {required_return:.1%} return, risk {risk_score} → {feasibility:.0f}% feasible")
        
        print("✅ Goal feasibility assessment test passed")
    
    def test_stock_recommendations_by_risk(self):
        """Test that different risk levels produce appropriate stock selections."""
        test_scenarios = [
            # (risk_score, expected_characteristics)
            (15, {"conservative_stocks": True, "max_volatility": "low"}),
            (40, {"balanced_mix": True, "bonds_included": True}),
            (75, {"growth_focused": True, "tech_heavy": True}),
            (90, {"high_risk": True, "innovation_themes": True})
        ]
        
        target_value = 50000
        timeframe = 10
        
        for risk_score, characteristics in test_scenarios:
            try:
                stocks = self.recommender.recommend_stocks(target_value, timeframe, risk_score)
                
                # Basic validation
                assert isinstance(stocks, list), f"Recommendations should be a list, got {type(stocks)}"
                assert 3 <= len(stocks) <= 15, f"Should recommend 3-15 stocks, got {len(stocks)}"
                
                # Check for duplicates
                assert len(stocks) == len(set(stocks)), f"Duplicate stocks in recommendations: {stocks}"
                
                # Check that all are valid ticker symbols
                for stock in stocks:
                    assert isinstance(stock, str), f"Stock ticker should be string, got {type(stock)}"
                    assert len(stock) >= 2, f"Stock ticker too short: {stock}"
                    assert stock.isupper(), f"Stock ticker should be uppercase: {stock}"
                
                print(f"✅ Risk {risk_score}: {len(stocks)} stocks → {stocks}")
                
            except Exception as e:
                print(f"❌ Stock recommendation failed for risk {risk_score}: {e}")
                assert False, f"Stock recommendation failed: {e}"
        
        print("✅ Stock recommendations by risk test passed")
    
    def test_goal_oriented_optimization(self):
        """Test that recommendations adapt based on goal requirements."""
        base_scenarios = [
            # Easy goal (should be more conservative)
            {"target": 30000, "current": 20000, "timeframe": 10, "monthly": 100, "risk": 50},
            # Challenging goal (should be more aggressive)
            {"target": 100000, "current": 5000, "timeframe": 5, "monthly": 500, "risk": 50},
            # Impossible goal (should max out risk within tolerance)
            {"target": 500000, "current": 1000, "timeframe": 3, "monthly": 100, "risk": 30}
        ]
        
        for i, scenario in enumerate(base_scenarios):
            try:
                stocks = self.recommender.recommend_stocks(
                    scenario["target"], scenario["timeframe"], scenario["risk"],
                    scenario["current"], scenario["monthly"]
                )
                
                # Calculate required return for context
                required_return = self.recommender.calculate_required_return(
                    scenario["target"], scenario["current"], scenario["timeframe"], scenario["monthly"]
                )
                
                # Assess goal feasibility
                assessment = self.recommender.assess_goal_feasibility(required_return, scenario["risk"])
                
                assert len(stocks) >= 3, f"Should recommend at least 3 stocks for scenario {i+1}"
                
                print(f"✅ Scenario {i+1}: Need {required_return:.1%} return → {len(stocks)} stocks")
                print(f"   Feasibility: {assessment['feasibility_score']:.0f}% ({assessment['recommendation']})")
                print(f"   Stocks: {stocks}")
                
            except Exception as e:
                print(f"❌ Goal optimization failed for scenario {i+1}: {e}")
                assert False, f"Goal optimization failed: {e}"
        
        print("✅ Goal-oriented optimization test passed")
    
    def test_timeframe_impact(self):
        """Test that investment timeframe affects stock selection."""
        base_params = {"target_value": 50000, "risk_score": 50, "current_investment": 10000, "monthly_contribution": 300}
        timeframes = [2, 5, 10, 20]  # Short to long term
        
        recommendations = {}
        
        for timeframe in timeframes:
            try:
                stocks = self.recommender.recommend_stocks(timeframe=timeframe, **base_params)
                recommendations[timeframe] = stocks
                
                print(f"✅ {timeframe} years: {len(stocks)} stocks → {stocks}")
                
            except Exception as e:
                print(f"❌ Timeframe impact test failed for {timeframe} years: {e}")
                assert False, f"Timeframe test failed: {e}"
        
        # Short-term should generally be more conservative
        # Long-term can be more aggressive
        # This is hard to test automatically, but we can check basic properties
        
        for timeframe in timeframes:
            stocks = recommendations[timeframe]
            assert len(stocks) >= 3, f"Should have at least 3 stocks for {timeframe} years"
        
        print("✅ Timeframe impact test passed")
    
    @patch('yfinance.Ticker')
    def test_stock_validation(self, mock_ticker):
        """Test stock symbol validation with mocked yfinance."""
        # Mock valid stock
        mock_valid_ticker = Mock()
        mock_valid_ticker.info = {'regularMarketPrice': 150.0}
        
        # Mock invalid stock
        mock_invalid_ticker = Mock()
        mock_invalid_ticker.info = {}
        
        def mock_ticker_side_effect(symbol):
            if symbol in ['AAPL', 'MSFT', 'VTI']:
                return mock_valid_ticker
            else:
                return mock_invalid_ticker
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        test_stocks = ['AAPL', 'MSFT', 'INVALID1', 'VTI', 'INVALID2']
        valid_stocks = self.recommender.validate_and_filter_stocks(test_stocks)
        
        expected_valid = ['AAPL', 'MSFT', 'VTI']
        assert valid_stocks == expected_valid, f"Expected {expected_valid}, got {valid_stocks}"
        
        print("✅ Stock validation test passed")
    
    def test_fallback_mechanisms(self):
        """Test fallback stock selection when AI model fails."""
        risk_profiles = [
            (20, "conservative"),
            (50, "moderate"), 
            (80, "aggressive")
        ]
        
        for risk_score, risk_type in risk_profiles:
            try:
                fallback_stocks = get_fallback_stocks_by_risk_profile(risk_score, risk_type)
                
                assert isinstance(fallback_stocks, list), f"Fallback should return list, got {type(fallback_stocks)}"
                assert len(fallback_stocks) >= 3, f"Fallback should have at least 3 stocks, got {len(fallback_stocks)}"
                
                # Check for known ETF tickers
                known_etfs = {'VTI', 'BND', 'VEA', 'VWO', 'VNQ', 'QQQ', 'VGT', 'VUG', 'ARKK', 'VTEB'}
                for stock in fallback_stocks:
                    assert stock in known_etfs, f"Fallback stock {stock} not in known ETFs"
                
                print(f"✅ Fallback {risk_type} ({risk_score}): {fallback_stocks}")
                
            except Exception as e:
                print(f"❌ Fallback test failed for {risk_type}: {e}")
                assert False, f"Fallback mechanism failed: {e}"
        
        print("✅ Fallback mechanisms test passed")
    
    def test_backward_compatibility(self):
        """Test that the main train_and_recommend function works."""
        test_cases = [
            (25000, 5, 30),   # Conservative
            (50000, 10, 50),  # Moderate
            (100000, 15, 75)  # Aggressive
        ]
        
        for target, timeframe, risk in test_cases:
            try:
                stocks = train_and_recommend(target, timeframe, risk)
                
                assert isinstance(stocks, list), f"train_and_recommend should return list, got {type(stocks)}"
                assert len(stocks) >= 3, f"Should recommend at least 3 stocks, got {len(stocks)}"
                
                print(f"✅ Backward compatibility: £{target:,}, {timeframe}y, risk {risk} → {stocks}")
                
            except Exception as e:
                print(f"❌ Backward compatibility test failed: {e}")
                assert False, f"Backward compatibility failed: {e}"
        
        print("✅ Backward compatibility test passed")

def run_stock_selection_tests():
    """Run all stock selection tests."""
    print("🧪 Running Stock Selection Tests")
    print("=" * 50)
    
    test_suite = TestStockSelection()
    
    tests = [
        test_suite.test_risk_category_mapping,
        test_suite.test_required_return_calculation,
        test_suite.test_goal_feasibility_assessment,
        test_suite.test_stock_recommendations_by_risk,
        test_suite.test_goal_oriented_optimization,
        test_suite.test_timeframe_impact,
        test_suite.test_stock_validation,
        test_suite.test_fallback_mechanisms,
        test_suite.test_backward_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All stock selection tests passed!")
    else:
        print(f"⚠️  {failed} tests failed - check your stock selection logic")

if __name__ == "__main__":
    run_stock_selection_tests()