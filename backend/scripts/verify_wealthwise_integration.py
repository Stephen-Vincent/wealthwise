# scripts/verify_wealthwise_integration.py
"""
WealthWise Integration Verification Script

This script comprehensively tests your WealthWise integration to ensure:
1. All components are properly installed
2. Models can be trained if needed
3. API endpoints are working
4. Fallback systems are functioning
5. Database integration is working

Run this before deploying to production.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any, List
import traceback

# Add your project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WealthWiseVerifier:
    """Comprehensive verification of WealthWise integration"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "warnings": 0,
            "details": {}
        }
    
    async def run_all_tests(self):
        """Run comprehensive integration tests"""
        
        print("🚀 Starting WealthWise Integration Verification")
        print("=" * 60)
        
        # Test 1: Import verification
        await self.test_imports()
        
        # Test 2: WealthWise availability check
        await self.test_wealthwise_availability()
        
        # Test 3: Model training (if needed)
        await self.test_model_training()
        
        # Test 4: Portfolio simulation integration
        await self.test_portfolio_simulation()
        
        # Test 5: API endpoints
        await self.test_api_endpoints()
        
        # Test 6: Database integration
        await self.test_database_integration()
        
        # Test 7: Fallback systems
        await self.test_fallback_systems()
        
        # Test 8: SHAP explanations
        await self.test_shap_explanations()
        
        # Generate final report
        self.generate_report()
        
        return self.results
    
    async def test_imports(self):
        """Test if all required modules can be imported"""
        print("\n📦 Testing Imports...")
        
        try:
            # Test core imports
            from services.portfolio_simulator import simulate_portfolio
            from backend.services.portfolio_simulator.ai_analysis import AIAnalysisService
            print("✅ Core services imported successfully")
            
            # Test WealthWise imports
            try:
                from ai_models.stock_model.core.recommender import EnhancedStockRecommender
                from ai_models.stock_model.explainable_ai import SHAPExplainer
                from ai_models.stock_model.goal_optimization import GoalCalculator
                from ai_models.stock_model.utils import initialize_complete_system
                
                self.results["details"]["wealthwise_imports"] = "✅ Success"
                print("✅ WealthWise modules imported successfully")
                self._pass_test("imports")
                
            except ImportError as e:
                self.results["details"]["wealthwise_imports"] = f"❌ Failed: {str(e)}"
                print(f"⚠️ WealthWise import failed: {e}")
                print("   System will use fallback mode")
                self._warning("imports")
                
        except Exception as e:
            print(f"❌ Critical import failure: {e}")
            self.results["details"]["core_imports"] = f"❌ Failed: {str(e)}"
            self._fail_test("imports")
    
    async def test_wealthwise_availability(self):
        """Test WealthWise system availability"""
        print("\n🎯 Testing WealthWise Availability...")
        
        try:
            from services.portfolio_simulator import WEALTHWISE_AVAILABLE
            
            if WEALTHWISE_AVAILABLE:
                print("✅ WealthWise is available and ready")
                self.results["details"]["wealthwise_available"] = "✅ Available"
                self._pass_test("wealthwise_availability")
                
                # Test initialization
                try:
                    from ai_models.stock_model.utils import initialize_complete_system
                    init_result = initialize_complete_system({
                        'LOG_LEVEL': 'INFO',
                        'LOG_TO_FILE': False,
                        'ENABLE_PERFORMANCE_TRACKING': True
                    })
                    
                    if init_result['success']:
                        print("✅ WealthWise initialization successful")
                        self.results["details"]["wealthwise_init"] = "✅ Success"
                    else:
                        print(f"⚠️ WealthWise initialization warning: {init_result.get('error')}")
                        self.results["details"]["wealthwise_init"] = f"⚠️ Warning: {init_result.get('error')}"
                        self._warning("wealthwise_init")
                        
                except Exception as e:
                    print(f"❌ WealthWise initialization failed: {e}")
                    self.results["details"]["wealthwise_init"] = f"❌ Failed: {str(e)}"
                    self._fail_test("wealthwise_init")
            else:
                print("⚠️ WealthWise not available - system will use fallback mode")
                self.results["details"]["wealthwise_available"] = "⚠️ Not available - using fallback"
                self._warning("wealthwise_availability")
                
        except Exception as e:
            print(f"❌ Error checking WealthWise availability: {e}")
            self.results["details"]["wealthwise_available"] = f"❌ Error: {str(e)}"
            self._fail_test("wealthwise_availability")
    
    async def test_model_training(self):
        """Test model training if WealthWise is available"""
        print("\n🧠 Testing Model Training...")
        
        try:
            from services.portfolio_simulator import WEALTHWISE_AVAILABLE
            
            if not WEALTHWISE_AVAILABLE:
                print("⚠️ Skipping model training - WealthWise not available")
                self.results["details"]["model_training"] = "⚠️ Skipped - WealthWise not available"
                self._warning("model_training")
                return
            
            # Test SHAP model training
            try:
                from ai_models.stock_model.explainable_ai import SHAPExplainer
                
                shap_explainer = SHAPExplainer()
                
                if not shap_explainer.is_available():
                    print("🔄 Training SHAP model (this may take a few minutes)...")
                    success = shap_explainer.train_shap_model(num_samples=100)  # Small sample for testing
                    
                    if success:
                        print("✅ SHAP model trained successfully")
                        self.results["details"]["shap_training"] = "✅ Success"
                        self._pass_test("shap_training")
                    else:
                        print("❌ SHAP model training failed")
                        self.results["details"]["shap_training"] = "❌ Failed"
                        self._fail_test("shap_training")
                else:
                    print("✅ SHAP model already available")
                    self.results["details"]["shap_training"] = "✅ Already available"
                    self._pass_test("shap_training")
                    
            except Exception as e:
                print(f"❌ SHAP model training error: {e}")
                self.results["details"]["shap_training"] = f"❌ Error: {str(e)}"
                self._fail_test("shap_training")
                
        except Exception as e:
            print(f"❌ Model training test failed: {e}")
            self.results["details"]["model_training"] = f"❌ Error: {str(e)}"
            self._fail_test("model_training")
    
    async def test_portfolio_simulation(self):
        """Test portfolio simulation with sample data"""
        print("\n💼 Testing Portfolio Simulation...")
        
        try:
            from services.portfolio_simulator import simulate_portfolio
            from database.db import SessionLocal
            
            # Create test simulation input
            test_input = {
                "user_id": "test_user",
                "goal": "wealth building",
                "target_value": 50000,
                "lump_sum": 10000,
                "monthly": 500,
                "timeframe": 10,
                "years_of_experience": 5,
                "income_bracket": "medium",
                "risk_score": 50,
                "risk_label": "Medium"
            }
            
            # Test with database session
            db = SessionLocal()
            try:
                print("🔄 Running test simulation...")
                result = await simulate_portfolio(test_input, db)
                
                # Verify result structure
                required_fields = ['id', 'ai_summary', 'results', 'target_achieved']
                missing_fields = [field for field in required_fields if field not in result]
                
                if not missing_fields:
                    print("✅ Portfolio simulation completed successfully")
                    print(f"   Enhanced: {result.get('wealthwise_enhanced', False)}")
                    print(f"   SHAP Available: {result.get('has_shap_explanations', False)}")
                    
                    self.results["details"]["portfolio_simulation"] = {
                        "status": "✅ Success",
                        "enhanced": result.get('wealthwise_enhanced', False),
                        "shap_available": result.get('has_shap_explanations', False)
                    }
                    self._pass_test("portfolio_simulation")
                else:
                    print(f"❌ Missing required fields: {missing_fields}")
                    self.results["details"]["portfolio_simulation"] = f"❌ Missing fields: {missing_fields}"
                    self._fail_test("portfolio_simulation")
                    
            finally:
                db.close()
                
        except Exception as e:
            print(f"❌ Portfolio simulation test failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            self.results["details"]["portfolio_simulation"] = f"❌ Error: {str(e)}"
            self._fail_test("portfolio_simulation")
    
    async def test_api_endpoints(self):
        """Test API endpoints are properly configured"""
        print("\n🌐 Testing API Endpoints...")
        
        try:
            # Import router to check endpoint configuration
            from api.routers.ai_analysis import router
            
            # Check if new endpoints are available
            expected_endpoints = [
                "/ai/simulate",
                "/ai/simulation/{simulation_id}/shap-visualization",
                "/ai/simulation/{simulation_id}/news-analysis"
            ]
            
            # Get all routes from the router
            router_paths = [route.path for route in router.routes]
            
            missing_endpoints = []
            for endpoint in expected_endpoints:
                # Check for pattern match (handling path parameters)
                endpoint_base = endpoint.split('/{')[0]
                if not any(endpoint_base in path for path in router_paths):
                    missing_endpoints.append(endpoint)
            
            if not missing_endpoints:
                print("✅ All expected API endpoints are configured")
                self.results["details"]["api_endpoints"] = "✅ All endpoints configured"
                self._pass_test("api_endpoints")
            else:
                print(f"⚠️ Missing endpoints: {missing_endpoints}")
                self.results["details"]["api_endpoints"] = f"⚠️ Missing: {missing_endpoints}"
                self._warning("api_endpoints")
                
        except Exception as e:
            print(f"❌ API endpoint test failed: {e}")
            self.results["details"]["api_endpoints"] = f"❌ Error: {str(e)}"
            self._fail_test("api_endpoints")
    
    async def test_database_integration(self):
        """Test database integration for enhanced features"""
        print("\n🗃️ Testing Database Integration...")
        
        try:
            from database.db import SessionLocal
            from database import models
            
            db = SessionLocal()
            try:
                # Test if we can query simulations
                simulations = db.query(models.Simulation).limit(1).all()
                print("✅ Database connection successful")
                
                # Check if enhanced fields are being stored
                if simulations:
                    sim = simulations[0]
                    has_enhanced_data = bool(sim.results and sim.results.get("wealthwise_enhanced"))
                    
                    if has_enhanced_data:
                        print("✅ Enhanced simulation data found in database")
                        self.results["details"]["database_integration"] = "✅ Enhanced data available"
                    else:
                        print("⚠️ No enhanced simulation data found yet")
                        self.results["details"]["database_integration"] = "⚠️ No enhanced data yet"
                        self._warning("database_integration")
                else:
                    print("ℹ️ No simulations in database yet")
                    self.results["details"]["database_integration"] = "ℹ️ No simulations yet"
                
                self._pass_test("database_integration")
                
            finally:
                db.close()
                
        except Exception as e:
            print(f"❌ Database integration test failed: {e}")
            self.results["details"]["database_integration"] = f"❌ Error: {str(e)}"
            self._fail_test("database_integration")
    
    async def test_fallback_systems(self):
        """Test fallback systems work when WealthWise is unavailable"""
        print("\n🔄 Testing Fallback Systems...")
        
        try:
            # Test fallback stock selection
            from services.portfolio_simulator import get_fallback_stocks_by_risk_profile
            
            test_stocks = get_fallback_stocks_by_risk_profile(50, "Medium")
            
            if test_stocks and len(test_stocks) > 0:
                print(f"✅ Fallback stock selection working: {test_stocks}")
                self.results["details"]["fallback_systems"] = "✅ Working correctly"
                self._pass_test("fallback_systems")
            else:
                print("❌ Fallback stock selection failed")
                self.results["details"]["fallback_systems"] = "❌ Failed"
                self._fail_test("fallback_systems")
                
        except Exception as e:
            print(f"❌ Fallback systems test failed: {e}")
            self.results["details"]["fallback_systems"] = f"❌ Error: {str(e)}"
            self._fail_test("fallback_systems")
    
    async def test_shap_explanations(self):
        """Test SHAP explanation generation"""
        print("\n🔍 Testing SHAP Explanations...")
        
        try:
            from services.portfolio_simulator import WEALTHWISE_AVAILABLE
            
            if not WEALTHWISE_AVAILABLE:
                print("⚠️ Skipping SHAP test - WealthWise not available")
                self.results["details"]["shap_explanations"] = "⚠️ Skipped - WealthWise not available"
                self._warning("shap_explanations")
                return
            
            from ai_models.stock_model.explainable_ai import SHAPExplainer
            
            shap_explainer = SHAPExplainer()
            
            if shap_explainer.is_available():
                # Test SHAP explanation generation
                test_explanation = shap_explainer.get_shap_explanation(
                    target_value=..., 
                    timeframe=..., 
                    risk_score=..., 
                    current_investment=..., 
                    monthly_contribution=..., 
                    market_volatility=...,   # ✅ updated
                    market_trend=...         # ✅ updated
                )
                
                if test_explanation and "human_readable_explanation" in test_explanation:
                    print("✅ SHAP explanations generated successfully")
                    self.results["details"]["shap_explanations"] = "✅ Working correctly"
                    self._pass_test("shap_explanations")
                else:
                    print("❌ SHAP explanation generation failed")
                    self.results["details"]["shap_explanations"] = "❌ Generation failed"
                    self._fail_test("shap_explanations")
            else:
                print("⚠️ SHAP model not available - needs training")
                self.results["details"]["shap_explanations"] = "⚠️ Model not trained"
                self._warning("shap_explanations")
                
        except Exception as e:
            print(f"❌ SHAP explanations test failed: {e}")
            self.results["details"]["shap_explanations"] = f"❌ Error: {str(e)}"
            self._fail_test("shap_explanations")
    
    def _pass_test(self, test_name: str):
        """Mark test as passed"""
        self.results["tests_passed"] += 1
    
    def _fail_test(self, test_name: str):
        """Mark test as failed"""
        self.results["tests_failed"] += 1
    
    def _warning(self, test_name: str):
        """Mark test as warning"""
        self.results["warnings"] += 1
    
    def generate_report(self):
        """Generate final verification report"""
        
        print("\n" + "=" * 60)
        print("📊 WEALTHWISE INTEGRATION VERIFICATION REPORT")
        print("=" * 60)
        
        total_tests = self.results["tests_passed"] + self.results["tests_failed"] + self.results["warnings"]
        
        print(f"✅ Tests Passed: {self.results['tests_passed']}")
        print(f"❌ Tests Failed: {self.results['tests_failed']}")
        print(f"⚠️ Warnings: {self.results['warnings']}")
        print(f"📊 Total Tests: {total_tests}")
        
        if self.results["tests_failed"] == 0:
            if self.results["warnings"] == 0:
                print("\n🎉 PERFECT! WealthWise integration is fully functional!")
                recommendation = "✅ Ready for production deployment"
            else:
                print("\n✅ GOOD! WealthWise integration is working with minor warnings")
                recommendation = "⚠️ Ready for production, monitor warnings"
        else:
            print("\n❌ ISSUES DETECTED! Please review failed tests")
            recommendation = "❌ Fix issues before production deployment"
        
        self.results["recommendation"] = recommendation
        
        print(f"\n🎯 Recommendation: {recommendation}")
        
        # Detailed breakdown
        print("\n📋 Detailed Results:")
        for test, result in self.results["details"].items():
            print(f"   {test}: {result}")
        
        print("\n" + "=" * 60)

# Additional utility functions for manual testing

async def test_single_simulation():
    """Run a single test simulation"""
    print("🧪 Running Single Test Simulation...")
    
    try:
        from services.portfolio_simulator import simulate_portfolio
        from database.db import SessionLocal
        
        test_input = {
            "user_id": "manual_test",
            "goal": "retirement planning",
            "target_value": 100000,
            "lump_sum": 5000,
            "monthly": 800,
            "timeframe": 15,
            "years_of_experience": 3,
            "income_bracket": "medium",
            "risk_score": 65,
            "risk_label": "Moderate Aggressive"
        }
        
        db = SessionLocal()
        try:
            result = await simulate_portfolio(test_input, db)
            
            print(f"✅ Simulation ID: {result.get('id')}")
            print(f"📈 Target Achieved: {result.get('target_achieved')}")
            print(f"🤖 WealthWise Enhanced: {result.get('wealthwise_enhanced', False)}")
            print(f"🔍 SHAP Available: {result.get('has_shap_explanations', False)}")
            print(f"📝 AI Summary Preview: {result.get('ai_summary', '')[:100]}...")
            
            return result
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Test simulation failed: {e}")
        return None

async def force_train_models():
    """Force train all WealthWise models"""
    print("🔄 Force Training All Models...")
    
    try:
        from ai_models.stock_model.explainable_ai import SHAPExplainer
        from ai_models.stock_model.utils import initialize_complete_system
        
        # Initialize system
        init_result = initialize_complete_system({
            'LOG_LEVEL': 'INFO',
            'LOG_TO_FILE': False,
            'ENABLE_PERFORMANCE_TRACKING': True
        })
        
        if not init_result['success']:
            print(f"❌ System initialization failed: {init_result.get('error')}")
            return False
        
        # Train SHAP model
        print("🧠 Training SHAP model...")
        shap_explainer = SHAPExplainer()
        success = shap_explainer.train_shap_model(num_samples=1000)
        
        if success:
            print("✅ SHAP model trained successfully")
            return True
        else:
            print("❌ SHAP model training failed")
            return False
            
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

# Main execution
async def main():
    """Main verification function"""
    
    print("🎯 WealthWise Integration Verification")
    print("Choose an option:")
    print("1. Full Integration Test")
    print("2. Single Test Simulation")
    print("3. Force Train Models")
    print("4. Quick Status Check")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            verifier = WealthWiseVerifier()
            await verifier.run_all_tests()
            
        elif choice == "2":
            await test_single_simulation()
            
        elif choice == "3":
            await force_train_models()
            
        elif choice == "4":
            # Quick status check
            try:
                from services.portfolio_simulator import WEALTHWISE_AVAILABLE
                print(f"WealthWise Available: {WEALTHWISE_AVAILABLE}")
                
                if WEALTHWISE_AVAILABLE:
                    from ai_models.stock_model.explainable_ai import SHAPExplainer
                    shap = SHAPExplainer()
                    print(f"SHAP Model Trained: {shap.is_available()}")
                
            except Exception as e:
                print(f"Status check failed: {e}")
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Verification cancelled")

if __name__ == "__main__":
    asyncio.run(main())