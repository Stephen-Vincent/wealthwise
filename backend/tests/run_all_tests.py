# tests/run_all_tests.py

"""
Master test runner for all AI components.

Runs comprehensive tests for:
- Risk assessment
- Stock selection 
- AI summary generation
- Integration tests

Usage:
    python run_all_tests.py
    python run_all_tests.py --component risk
    python run_all_tests.py --component stock
    python run_all_tests.py --component summary
"""

import sys
import os
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_risk_tests():
    """Run risk assessment tests."""
    try:
        from test_risk import run_risk_tests
        run_risk_tests()
        return True
    except Exception as e:
        print(f"❌ Risk tests failed to run: {e}")
        return False

def run_stock_tests():
    """Run stock selection tests."""
    try:
        from test_stock_selection import run_stock_selection_tests
        run_stock_selection_tests()
        return True
    except Exception as e:
        print(f"❌ Stock selection tests failed to run: {e}")
        return False

def run_summary_tests():
    """Run AI summary tests."""
    try:
        from test_ai_summary import run_ai_summary_tests
        run_ai_summary_tests()
        return True
    except Exception as e:
        print(f"❌ AI summary tests failed to run: {e}")
        return False

def run_integration_tests():
    """Run integration tests across all components."""
    print("🔄 Running Integration Tests")
    print("=" * 50)
    
    integration_passed = 0
    integration_failed = 0
    
    # Test 1: End-to-end portfolio simulation
    try:
        print("🧪 Testing end-to-end portfolio simulation...")
        
        # Sample user input
        sample_input = {
            "years_of_experience": 5,
            "loss_tolerance": "wait_and_see",
            "panic_behavior": "no_never",
            "financial_behavior": "invest_all",
            "engagement_level": "monthly",
            "goal": "retirement",
            "target_value": 100000,
            "lump_sum": 10000,
            "monthly": 500,
            "timeframe": 10,
            "income_bracket": "medium",
            "user_id": 1
        }
        
        # Test risk assessment
        from services.risk_assessor import calculate_user_risk
        risk_profile = calculate_user_risk(sample_input)
        assert "risk_score" in risk_profile
        print(f"   ✅ Risk assessment: {risk_profile['risk_score']} ({risk_profile['risk_level']})")
        
        # Test stock selection
        from ai_models.stock_model.enhanced_stock_recommender import train_and_recommend
        stocks = train_and_recommend(
            sample_input["target_value"],
            sample_input["timeframe"], 
            risk_profile["risk_score"]
        )
        assert len(stocks) >= 3
        print(f"   ✅ Stock selection: {len(stocks)} stocks recommended")
        
        # Test simple summary generation
        from services.portfolio_simulator import generate_simple_summary
        mock_simulation_results = {
            "starting_value": 70000,
            "end_value": 120000,
            "portfolio_return": 0.08
        }
        
        mock_stocks_picked = [{"symbol": stock, "allocation": 1/len(stocks)} for stock in stocks]
        
        summary = generate_simple_summary(
            mock_stocks_picked,
            sample_input,
            risk_profile["risk_score"],
            risk_profile["risk_level"],
            mock_simulation_results
        )
        assert len(summary) > 50
        print(f"   ✅ Summary generation: {len(summary)} character summary")
        
        print("✅ End-to-end integration test passed")
        integration_passed += 1
        
    except Exception as e:
        print(f"❌ End-to-end integration test failed: {e}")
        integration_failed += 1
    
    # Test 2: Component compatibility
    try:
        print("🧪 Testing component compatibility...")
        
        # Check that risk scores from risk assessor work with stock recommender
        risk_scores = [15, 35, 55, 75, 95]
        
        for risk_score in risk_scores:
            try:
                stocks = train_and_recommend(50000, 10, risk_score)
                assert isinstance(stocks, list)
                assert len(stocks) >= 3
            except Exception as e:
                raise Exception(f"Stock recommender failed for risk score {risk_score}: {e}")
        
        print("   ✅ Risk score compatibility with stock recommender")
        
        # Check that stock selections work with summary generator
        sample_stocks = [
            {"symbol": "VTI", "allocation": 0.6},
            {"symbol": "BND", "allocation": 0.4}
        ]
        
        summary = generate_simple_summary(
            sample_stocks,
            {"goal": "test", "target_value": 50000, "timeframe": 10},
            50,
            "Moderate",
            {"starting_value": 30000, "end_value": 50000, "portfolio_return": 0.066}
        )
        
        assert "VTI" in summary and "BND" in summary
        print("   ✅ Stock selection compatibility with summary generator")
        
        print("✅ Component compatibility test passed")
        integration_passed += 1
        
    except Exception as e:
        print(f"❌ Component compatibility test failed: {e}")
        integration_failed += 1
    
    print("=" * 50)
    print(f"📊 Integration Test Results: {integration_passed} passed, {integration_failed} failed")
    
    return integration_failed == 0

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run AI component tests")
    parser.add_argument("--component", choices=["risk", "stock", "summary", "integration", "all"], 
                       default="all", help="Which component to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🤖 WealthWise AI Component Test Suite")
    print("=" * 60)
    print(f"📅 Test run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {}
    
    if args.component in ["risk", "all"]:
        print("\n")
        results["risk"] = run_risk_tests()
    
    if args.component in ["stock", "all"]:
        print("\n")
        results["stock"] = run_stock_tests()
    
    if args.component in ["summary", "all"]:
        print("\n") 
        results["summary"] = run_summary_tests()
    
    if args.component in ["integration", "all"]:
        print("\n")
        results["integration"] = run_integration_tests()
    
    # Summary report
    print("\n")
    print("📊 FINAL TEST REPORT")
    print("=" * 60)
    
    total_passed = sum(1 for success in results.values() if success)
    total_components = len(results)
    
    for component, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{component.upper():12} {status}")
    
    print("=" * 60)
    print(f"OVERALL: {total_passed}/{total_components} components passed")
    
    if total_passed == total_components:
        print("🎉 ALL TESTS PASSED! Your AI components are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)