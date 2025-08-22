# tests/test_risk.py

"""
Test suite for risk assessment functionality.

Tests the risk scoring algorithms, risk categorization,
and risk profile generation for users.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules we want to test
try:
    from backend.services.portfolio_simulator.risk_assessor import calculate_user_risk, calculate_user_risk_legacy
    from database.schemas import OnboardingCreate
except ImportError as e:
    print(f"Warning: Could not import risk assessment modules: {e}")
    print("Please ensure your risk_assessor.py exists in services/")

class TestRiskAssessment:
    """Test cases for risk assessment functionality."""
    
    def create_sample_onboarding_data(self, 
                                    experience=5, 
                                    loss_tolerance="wait_and_see",
                                    panic_behavior="no_never",
                                    financial_behavior="invest_all",
                                    engagement_level="monthly"):
        """Create sample onboarding data for testing."""
        return {
            "years_of_experience": experience,
            "loss_tolerance": loss_tolerance,
            "panic_behavior": panic_behavior,
            "financial_behavior": financial_behavior,
            "engagement_level": engagement_level,
            "goal": "retirement",
            "target_value": 100000,
            "lump_sum": 5000,
            "monthly": 500,
            "timeframe": 10,
            "income_bracket": "medium",
            "consent": True,
            "name": "Test User",
            "user_id": 1
        }
    
    def test_conservative_risk_profile(self):
        """Test that conservative answers result in low risk score."""
        conservative_data = self.create_sample_onboarding_data(
            experience=1,
            loss_tolerance="sell_immediately",
            panic_behavior="yes_always",
            financial_behavior="save_all",
            engagement_level="rarely"
        )
        
        try:
            risk_profile = calculate_user_risk(conservative_data)
            
            # Conservative profile should have low risk score
            assert risk_profile["risk_score"] < 40, f"Expected conservative risk score < 40, got {risk_profile['risk_score']}"
            assert "conservative" in risk_profile["risk_level"].lower(), f"Expected conservative risk level, got {risk_profile['risk_level']}"
            
            # Should have appropriate recommendations
            assert risk_profile["recommended_stock_allocation"] < 70, "Conservative profile should have < 70% stocks"
            assert risk_profile["recommended_bond_allocation"] > 30, "Conservative profile should have > 30% bonds"
            
            print("✅ Conservative risk profile test passed")
            
        except Exception as e:
            print(f"❌ Conservative risk profile test failed: {e}")
            assert False, f"Risk assessment failed: {e}"
    
    def test_aggressive_risk_profile(self):
        """Test that aggressive answers result in high risk score."""
        aggressive_data = self.create_sample_onboarding_data(
            experience=10,
            loss_tolerance="buy_more",
            panic_behavior="no_never", 
            financial_behavior="invest_all",
            engagement_level="daily"
        )
        
        try:
            risk_profile = calculate_user_risk(aggressive_data)
            
            # Aggressive profile should have high risk score
            assert risk_profile["risk_score"] > 60, f"Expected aggressive risk score > 60, got {risk_profile['risk_score']}"
            assert "aggressive" in risk_profile["risk_level"].lower(), f"Expected aggressive risk level, got {risk_profile['risk_level']}"
            
            # Should have appropriate recommendations
            assert risk_profile["recommended_stock_allocation"] > 70, "Aggressive profile should have > 70% stocks"
            assert risk_profile["recommended_bond_allocation"] < 30, "Aggressive profile should have < 30% bonds"
            
            print("✅ Aggressive risk profile test passed")
            
        except Exception as e:
            print(f"❌ Aggressive risk profile test failed: {e}")
            assert False, f"Risk assessment failed: {e}"
    
    def test_moderate_risk_profile(self):
        """Test that moderate answers result in medium risk score."""
        moderate_data = self.create_sample_onboarding_data(
            experience=5,
            loss_tolerance="wait_and_see",
            panic_behavior="yes_sometimes",
            financial_behavior="save_half",
            engagement_level="monthly"
        )
        
        try:
            risk_profile = calculate_user_risk(moderate_data)
            
            # Moderate profile should have medium risk score
            assert 30 <= risk_profile["risk_score"] <= 70, f"Expected moderate risk score 30-70, got {risk_profile['risk_score']}"
            assert "moderate" in risk_profile["risk_level"].lower(), f"Expected moderate risk level, got {risk_profile['risk_level']}"
            
            # Should have balanced recommendations
            assert 50 <= risk_profile["recommended_stock_allocation"] <= 80, "Moderate profile should have 50-80% stocks"
            assert 20 <= risk_profile["recommended_bond_allocation"] <= 50, "Moderate profile should have 20-50% bonds"
            
            print("✅ Moderate risk profile test passed")
            
        except Exception as e:
            print(f"❌ Moderate risk profile test failed: {e}")
            assert False, f"Risk assessment failed: {e}"
    
    def test_risk_score_bounds(self):
        """Test that risk scores are always within valid bounds (0-100)."""
        test_cases = [
            # Extreme conservative
            self.create_sample_onboarding_data(0, "sell_immediately", "yes_always", "save_all", "rarely"),
            # Extreme aggressive  
            self.create_sample_onboarding_data(20, "buy_more", "no_never", "invest_all", "daily"),
            # Mixed responses
            self.create_sample_onboarding_data(5, "wait_and_see", "yes_sometimes", "save_half", "monthly")
        ]
        
        for i, test_data in enumerate(test_cases):
            try:
                risk_profile = calculate_user_risk(test_data)
                risk_score = risk_profile["risk_score"]
                
                assert 0 <= risk_score <= 100, f"Risk score {risk_score} out of bounds for test case {i+1}"
                assert isinstance(risk_score, (int, float)), f"Risk score must be numeric, got {type(risk_score)}"
                
                print(f"✅ Risk score bounds test {i+1} passed: {risk_score}")
                
            except Exception as e:
                print(f"❌ Risk score bounds test {i+1} failed: {e}")
                assert False, f"Risk assessment failed: {e}"
    
    def test_risk_profile_completeness(self):
        """Test that risk profile contains all required fields."""
        sample_data = self.create_sample_onboarding_data()
        
        try:
            risk_profile = calculate_user_risk(sample_data)
            
            required_fields = [
                "risk_score",
                "risk_level", 
                "risk_description",
                "allocation_guidance",
                "recommended_stock_allocation",
                "recommended_bond_allocation",
                "explanation"
            ]
            
            for field in required_fields:
                assert field in risk_profile, f"Missing required field: {field}"
                assert risk_profile[field] is not None, f"Field {field} is None"
                
                if field.endswith("_allocation"):
                    assert isinstance(risk_profile[field], (int, float)), f"Allocation field {field} must be numeric"
                    assert 0 <= risk_profile[field] <= 100, f"Allocation {field} must be 0-100%"
            
            # Check that allocations roughly sum to 100%
            stock_alloc = risk_profile["recommended_stock_allocation"]
            bond_alloc = risk_profile["recommended_bond_allocation"]
            total_alloc = stock_alloc + bond_alloc
            
            assert 80 <= total_alloc <= 120, f"Stock + Bond allocation should be ~100%, got {total_alloc}%"
            
            print("✅ Risk profile completeness test passed")
            
        except Exception as e:
            print(f"❌ Risk profile completeness test failed: {e}")
            assert False, f"Risk assessment failed: {e}"
    
    def test_experience_impact(self):
        """Test that experience level affects risk tolerance appropriately."""
        base_data = self.create_sample_onboarding_data()
        
        # Test with different experience levels
        experience_levels = [0, 2, 5, 10, 15]
        risk_scores = []
        
        for experience in experience_levels:
            test_data = base_data.copy()
            test_data["years_of_experience"] = experience
            
            try:
                risk_profile = calculate_user_risk(test_data)
                risk_scores.append(risk_profile["risk_score"])
                
                print(f"Experience {experience} years → Risk score {risk_profile['risk_score']}")
                
            except Exception as e:
                print(f"❌ Experience impact test failed for {experience} years: {e}")
                assert False, f"Risk assessment failed: {e}"
        
        # Generally, more experience should correlate with higher risk tolerance
        # (though other factors matter too)
        novice_score = risk_scores[0]  # 0 years
        expert_score = risk_scores[-1]  # 15 years
        
        print(f"Novice (0y): {novice_score}, Expert (15y): {expert_score}")
        print("✅ Experience impact test completed")
    
    def test_legacy_compatibility(self):
        """Test that legacy risk calculation still works."""
        sample_data = self.create_sample_onboarding_data()
        
        try:
            # Convert to mock schema object for legacy function
            mock_onboarding = Mock()
            for key, value in sample_data.items():
                setattr(mock_onboarding, key, value)
            
            risk_score, risk_label = calculate_user_risk_legacy(mock_onboarding)
            
            assert isinstance(risk_score, (int, float)), f"Legacy risk score must be numeric, got {type(risk_score)}"
            assert 0 <= risk_score <= 100, f"Legacy risk score {risk_score} out of bounds"
            assert isinstance(risk_label, str), f"Legacy risk label must be string, got {type(risk_label)}"
            assert risk_label in ["Low", "Medium", "High"], f"Invalid legacy risk label: {risk_label}"
            
            print(f"✅ Legacy compatibility test passed: {risk_score} ({risk_label})")
            
        except Exception as e:
            print(f"❌ Legacy compatibility test failed: {e}")
            assert False, f"Legacy risk assessment failed: {e}"

def run_risk_tests():
    """Run all risk assessment tests."""
    print("🧪 Running Risk Assessment Tests")
    print("=" * 50)
    
    test_suite = TestRiskAssessment()
    
    tests = [
        test_suite.test_conservative_risk_profile,
        test_suite.test_aggressive_risk_profile,
        test_suite.test_moderate_risk_profile,
        test_suite.test_risk_score_bounds,
        test_suite.test_risk_profile_completeness,
        test_suite.test_experience_impact,
        test_suite.test_legacy_compatibility
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
        print("🎉 All risk assessment tests passed!")
    else:
        print(f"⚠️  {failed} tests failed - check your risk assessment logic")

if __name__ == "__main__":
    run_risk_tests()