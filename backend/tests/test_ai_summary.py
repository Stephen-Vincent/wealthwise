# tests/test_ai_summary.py

"""
Test suite for AI summary generation functionality.

Tests the AI analysis service, portfolio summaries,
educational content generation, and fallback mechanisms.
"""

import pytest
import sys
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Any

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules we want to test
try:
    from backend.services.portfolio_simulator.ai_analysis import AIAnalysisService
    from services.portfolio_simulator import generate_ai_enhanced_summary, generate_simple_summary
except ImportError as e:
    print(f"Warning: Could not import AI summary modules: {e}")
    print("Please ensure your ai_analysis.py exists in services/")

class TestAISummary:
    """Test cases for AI summary generation and analysis."""
    
    def setUp(self):
        """Set up test fixtures before each test method (unittest style)."""
        self.ai_service = AIAnalysisService()
        
    def setup_method(self, method):
        """Set up test fixtures before each test method (pytest style)."""
        self.ai_service = AIAnalysisService()
        
    def create_sample_portfolio_data(self):
        """Create sample portfolio data for testing."""
        return {
            "results": {
                "stocks_picked": [
                    {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "allocation": 0.4},
                    {"symbol": "BND", "name": "Vanguard Total Bond Market ETF", "allocation": 0.3},
                    {"symbol": "VEA", "name": "Vanguard FTSE Developed Markets ETF", "allocation": 0.2},
                    {"symbol": "VWO", "name": "Vanguard Emerging Markets ETF", "allocation": 0.1}
                ],
                "starting_value": 15000.0,
                "end_value": 45000.0,
                "return": 0.08,
                "target_reached": True,
                "timeline": {
                    "contributions": [
                        {"date": "2020-01-01", "value": 5000},
                        {"date": "2024-12-31", "value": 20000}
                    ],
                    "portfolio": [
                        {"date": "2020-01-01", "value": 5000},
                        {"date": "2024-12-31", "value": 45000}
                    ]
                }
            },
            "goal": "retirement",
            "target_value": 40000,
            "lump_sum": 5000,
            "monthly": 250,
            "timeframe": 5,
            "risk_score": 45,
            "risk_label": "Moderate"
        }
    
    def create_sample_user_data(self):
        """Create sample user data for testing."""
        return {
            "goal": "retirement",
            "target_value": 40000,
            "lump_sum": 5000,
            "monthly": 250,
            "timeframe": 5
        }
    
    def create_sample_simulation_results(self):
        """Create sample simulation results for testing."""
        return {
            "starting_value": 15000.0,
            "end_value": 45000.0,
            "portfolio_return": 0.08,
            "timeline": {
                "contributions": [
                    {"date": "2020-01-01", "value": 5000},
                    {"date": "2024-12-31", "value": 20000}
                ],
                "portfolio": [
                    {"date": "2020-01-01", "value": 5000},
                    {"date": "2024-12-31", "value": 45000}
                ]
            }
        }
    
    def test_simple_summary_generation(self):
        """Test the fallback simple summary generation."""
        stocks_picked = [
            {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "allocation": 0.4},
            {"symbol": "BND", "name": "Vanguard Total Bond Market ETF", "allocation": 0.3},
            {"symbol": "VEA", "name": "Vanguard FTSE Developed Markets ETF", "allocation": 0.3}
        ]
        
        user_data = self.create_sample_user_data()
        simulation_results = self.create_sample_simulation_results()
        
        try:
            summary = generate_simple_summary(
                stocks_picked, user_data, 45, "Moderate", simulation_results
            )
            
            # Basic validation
            assert isinstance(summary, str), f"Summary should be string, got {type(summary)}"
            assert len(summary) > 50, f"Summary too short: {len(summary)} characters"
            assert len(summary) < 2000, f"Summary too long: {len(summary)} characters"
            
            # Check for key information
            assert "retirement" in summary.lower(), "Summary should mention the goal"
            assert "moderate" in summary.lower(), "Summary should mention risk level"
            assert "£15,000" in summary or "15000" in summary, "Summary should mention starting value"
            assert "£45,000" in summary or "45000" in summary, "Summary should mention ending value"
            
            # Check for stock mentions
            for stock in stocks_picked:
                assert stock["symbol"] in summary, f"Summary should mention {stock['symbol']}"
            
            print("✅ Simple summary generation test passed")
            print(f"Sample summary: {summary[:200]}...")
            
        except Exception as e:
            print(f"❌ Simple summary generation test failed: {e}")
            assert False, f"Simple summary generation failed: {e}"
    
    @patch('aiohttp.ClientSession')
    async def test_ollama_connection_success(self, mock_session):
        """Test successful connection to Ollama service."""
        # Mock successful Ollama response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "response": "This is a test AI response about portfolio performance."
        })
        
        mock_session_instance = AsyncMock()
        mock_session_instance.post.return_value.__aenter__.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        try:
            response = await self.ai_service._get_ollama_response("Test prompt")
            
            assert isinstance(response, str), f"Response should be string, got {type(response)}"
            assert len(response) > 10, f"Response too short: {len(response)} characters"
            assert "portfolio" in response.lower(), "Response should be relevant to prompt"
            
            print("✅ Ollama connection success test passed")
            
        except Exception as e:
            print(f"❌ Ollama connection test failed: {e}")
            assert False, f"Ollama connection failed: {e}"
    
    @patch('aiohttp.ClientSession')
    async def test_ollama_connection_failure(self, mock_session):
        """Test handling of Ollama service failure."""
        # Mock failed connection
        mock_session_instance = AsyncMock()
        mock_session_instance.post.side_effect = Exception("Connection refused")
        mock_session.return_value = mock_session_instance
        
        try:
            with pytest.raises(Exception):
                await self.ai_service._get_ollama_response("Test prompt")
            
            print("✅ Ollama connection failure test passed")
            
        except Exception as e:
            print(f"❌ Ollama connection failure test failed: {e}")
            assert False, f"Ollama failure handling failed: {e}"
    
    async def test_portfolio_performance_analysis(self):
        """Test portfolio performance analysis functionality."""
        portfolio_data = self.create_sample_portfolio_data()
        
        try:
            # Test the analysis data preparation
            analysis_data = self.ai_service._prepare_analysis_data(portfolio_data)
            
            # Validate analysis data structure
            required_fields = ["total_value", "total_invested", "total_return", "timeframe", 
                             "risk_label", "goal", "target_value", "holdings"]
            
            for field in required_fields:
                assert field in analysis_data, f"Missing field in analysis data: {field}"
            
            # Validate calculations
            assert analysis_data["total_value"] == 45000, f"Expected total_value 45000, got {analysis_data['total_value']}"
            assert analysis_data["goal"] == "retirement", f"Expected goal 'retirement', got {analysis_data['goal']}"
            assert len(analysis_data["holdings"]) == 4, f"Expected 4 holdings, got {len(analysis_data['holdings'])}"
            
            print("✅ Portfolio performance analysis test passed")
            
        except Exception as e:
            print(f"❌ Portfolio performance analysis test failed: {e}")
            assert False, f"Portfolio analysis failed: {e}"
    
    async def test_risk_allocation_analysis(self):
        """Test risk and allocation analysis functionality."""
        portfolio_data = self.create_sample_portfolio_data()
        
        try:
            # Test risk data preparation
            risk_data = self.ai_service._prepare_risk_data(portfolio_data)
            
            # Validate risk data structure
            required_fields = ["risk_profile", "risk_score", "total_holdings", "largest_holding"]
            
            for field in required_fields:
                assert field in risk_data, f"Missing field in risk data: {field}"
            
            # Validate values
            assert risk_data["risk_profile"] == "Moderate", f"Expected Moderate risk, got {risk_data['risk_profile']}"
            assert risk_data["risk_score"] == 45, f"Expected risk score 45, got {risk_data['risk_score']}"
            assert risk_data["total_holdings"] == 4, f"Expected 4 holdings, got {risk_data['total_holdings']}"
            
            # Check largest holding
            largest = risk_data["largest_holding"]
            assert largest["symbol"] == "VTI", f"Expected VTI as largest holding, got {largest['symbol']}"
            assert largest["allocation"] == 0.4, f"Expected 0.4 allocation, got {largest['allocation']}"
            
            print("✅ Risk allocation analysis test passed")
            
        except Exception as e:
            print(f"❌ Risk allocation analysis test failed: {e}")
            assert False, f"Risk allocation analysis failed: {e}"
    
    def test_performance_prompt_generation(self):
        """Test generation of performance analysis prompts."""
        portfolio_data = self.create_sample_portfolio_data()
        analysis_data = self.ai_service._prepare_analysis_data(portfolio_data)
        
        try:
            prompt = self.ai_service._create_performance_prompt(analysis_data, None)
            
            # Validate prompt structure
            assert isinstance(prompt, str), f"Prompt should be string, got {type(prompt)}"
            assert len(prompt) > 100, f"Prompt too short: {len(prompt)} characters"
            
            # Check for key elements
            assert "£45,000" in prompt or "45000" in prompt, "Prompt should include portfolio value"
            assert "retirement" in prompt.lower(), "Prompt should include goal"
            assert "moderate" in prompt.lower(), "Prompt should include risk level"
            assert "explain" in prompt.lower(), "Prompt should request explanation"
            
            print("✅ Performance prompt generation test passed")
            print(f"Sample prompt snippet: {prompt[:200]}...")
            
        except Exception as e:
            print(f"❌ Performance prompt generation test failed: {e}")
            assert False, f"Prompt generation failed: {e}"
    
    def test_risk_prompt_generation(self):
        """Test generation of risk analysis prompts."""
        portfolio_data = self.create_sample_portfolio_data()
        risk_data = self.ai_service._prepare_risk_data(portfolio_data)
        
        try:
            prompt = self.ai_service._create_risk_prompt(risk_data)
            
            # Validate prompt structure
            assert isinstance(prompt, str), f"Prompt should be string, got {type(prompt)}"
            assert len(prompt) > 50, f"Prompt too short: {len(prompt)} characters"
            
            # Check for key elements
            assert "moderate" in prompt.lower(), "Prompt should include risk profile"
            assert "45" in prompt, "Prompt should include risk score"
            assert "diversification" in prompt.lower(), "Prompt should mention diversification"
            
            print("✅ Risk prompt generation test passed")
            
        except Exception as e:
            print(f"❌ Risk prompt generation test failed: {e}")
            assert False, f"Risk prompt generation failed: {e}"
    
    def test_fallback_analysis_generation(self):
        """Test fallback analysis when AI is unavailable."""
        portfolio_data = self.create_sample_portfolio_data()
        
        try:
            # Test positive return scenario
            fallback_analysis = self.ai_service._get_fallback_analysis(portfolio_data)
            
            assert isinstance(fallback_analysis, str), f"Fallback should be string, got {type(fallback_analysis)}"
            assert len(fallback_analysis) > 20, f"Fallback too short: {len(fallback_analysis)} characters"
            assert "grown" in fallback_analysis.lower(), "Should mention portfolio growth"
            
            # Test negative return scenario
            negative_portfolio = portfolio_data.copy()
            negative_portfolio["results"]["end_value"] = 10000  # Loss scenario
            
            negative_fallback = self.ai_service._get_fallback_analysis(negative_portfolio)
            assert "decrease" in negative_fallback.lower() or "normal" in negative_fallback.lower(), \
                "Should acknowledge portfolio decrease"
            
            print("✅ Fallback analysis generation test passed")
            
        except Exception as e:
            print(f"❌ Fallback analysis generation test failed: {e}")
            assert False, f"Fallback analysis failed: {e}"
    
    async def test_enhanced_summary_integration(self):
        """Test the enhanced AI summary generation integration."""
        stocks_picked = [
            {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "allocation": 0.4},
            {"symbol": "BND", "name": "Vanguard Total Bond Market ETF", "allocation": 0.6}
        ]
        
        user_data = self.create_sample_user_data()
        simulation_results = self.create_sample_simulation_results()
        
        try:
            # This will likely fail to connect to Ollama, but should fall back gracefully
            summary = await generate_ai_enhanced_summary(
                stocks_picked, user_data, 45, "Moderate", simulation_results
            )
            
            # Validate summary
            assert isinstance(summary, str), f"Summary should be string, got {type(summary)}"
            assert len(summary) > 30, f"Summary too short: {len(summary)} characters"
            
            # Should contain key information regardless of AI success/failure
            assert Any(word in summary.lower() for word in ["portfolio", "investment", "simulation"]), \
                "Summary should mention portfolio/investment concepts"
            
            print("✅ Enhanced summary integration test passed")
            print(f"Summary type: {'AI-generated' if len(summary) > 200 else 'Fallback'}")
            
        except Exception as e:
            print(f"❌ Enhanced summary integration test failed: {e}")
            assert False, f"Enhanced summary integration failed: {e}"
    
    def test_summary_content_quality(self):
        """Test the quality and completeness of generated summaries."""
        stocks_picked = [
            {"symbol": "VTI", "allocation": 0.5},
            {"symbol": "BND", "allocation": 0.3},
            {"symbol": "VEA", "allocation": 0.2}
        ]
        
        user_data = {
            "goal": "house deposit",
            "target_value": 50000,
            "lump_sum": 10000,
            "monthly": 500,
            "timeframe": 8
        }
        
        simulation_results = {
            "starting_value": 58000,
            "end_value": 75000,
            "portfolio_return": 0.06
        }
        
        try:
            summary = generate_simple_summary(
                stocks_picked, user_data, 60, "Moderate Aggressive", simulation_results
            )
            
            # Content quality checks
            quality_indicators = [
                "house deposit" in summary.lower(),  # Mentions goal
                "8 years" in summary or "timeframe" in summary.lower(),  # Mentions timeframe
                "£58,000" in summary or "58000" in summary,  # Starting value
                "£75,000" in summary or "75000" in summary,  # Ending value
                Any(stock["symbol"] in summary for stock in stocks_picked),  # Mentions stocks
                "moderate aggressive" in summary.lower() or "risk" in summary.lower()  # Risk level
            ]
            
            passed_checks = sum(quality_indicators)
            assert passed_checks >= 4, f"Summary failed {6-passed_checks} quality checks"
            
            # Check for educational tone
            educational_words = ["demonstrates", "investing", "diversified", "growth", "portfolio"]
            has_educational_tone = Any(word in summary.lower() for word in educational_words)
            assert has_educational_tone, "Summary should have educational tone"
            
            print("✅ Summary content quality test passed")
            print(f"Quality score: {passed_checks}/6 checks passed")
            
        except Exception as e:
            print(f"❌ Summary content quality test failed: {e}")
            assert False, f"Summary quality test failed: {e}"
    
    def test_summary_length_constraints(self):
        """Test that summaries meet length requirements."""
        test_scenarios = [
            # Different portfolio sizes and complexities
            {
                "stocks": [{"symbol": "VTI", "allocation": 1.0}],  # Simple portfolio
                "goal": "emergency fund",
                "complexity": "simple"
            },
            {
                "stocks": [  # Complex portfolio
                    {"symbol": "VTI", "allocation": 0.3},
                    {"symbol": "BND", "allocation": 0.2},
                    {"symbol": "VEA", "allocation": 0.2},
                    {"symbol": "VWO", "allocation": 0.1},
                    {"symbol": "VNQ", "allocation": 0.1},
                    {"symbol": "QQQ", "allocation": 0.1}
                ],
                "goal": "multi-generational wealth",
                "complexity": "complex"
            }
        ]
        
        for scenario in test_scenarios:
            user_data = self.create_sample_user_data()
            user_data["goal"] = scenario["goal"]
            simulation_results = self.create_sample_simulation_results()
            
            try:
                summary = generate_simple_summary(
                    scenario["stocks"], user_data, 50, "Moderate", simulation_results
                )
                
                # Length constraints
                assert 50 <= len(summary) <= 1500, \
                    f"Summary length {len(summary)} not in acceptable range (50-1500) for {scenario['complexity']} portfolio"
                
                # Should be readable (not too dense)
                word_count = len(summary.split())
                assert 10 <= word_count <= 300, \
                    f"Word count {word_count} not in acceptable range (10-300) for {scenario['complexity']} portfolio"
                
                print(f"✅ {scenario['complexity'].title()} portfolio summary: {len(summary)} chars, {word_count} words")
                
            except Exception as e:
                print(f"❌ Length constraints test failed for {scenario['complexity']} portfolio: {e}")
                assert False, f"Length constraints failed: {e}"
        
        print("✅ Summary length constraints test passed")

def run_ai_summary_tests():
    """Run all AI summary tests."""
    print("🧪 Running AI Summary Tests")
    print("=" * 50)
    
    test_suite = TestAISummary()
    
    # Synchronous tests
    sync_tests = [
        test_suite.test_simple_summary_generation,
        test_suite.test_performance_prompt_generation,
        test_suite.test_risk_prompt_generation,
        test_suite.test_fallback_analysis_generation,
        test_suite.test_summary_content_quality,
        test_suite.test_summary_length_constraints
    ]
    
    # Asynchronous tests
    async_tests = [
        test_suite.test_ollama_connection_success,
        test_suite.test_ollama_connection_failure,
        test_suite.test_portfolio_performance_analysis,
        test_suite.test_risk_allocation_analysis,
        test_suite.test_enhanced_summary_integration
    ]
    
    passed = 0
    failed = 0
    
    # Run synchronous tests
    for test in sync_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    # Run asynchronous tests
    for test in async_tests:
        try:
            asyncio.run(test())
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All AI summary tests passed!")
    else:
        print(f"⚠️  {failed} tests failed - check your AI summary logic")

if __name__ == "__main__":
    run_ai_summary_tests()