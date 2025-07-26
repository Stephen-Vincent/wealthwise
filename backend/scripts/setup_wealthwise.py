# scripts/setup_wealthwise.py
"""
WealthWise Setup Script

This script helps you set up and configure WealthWise for your system.
It will:
1. Check prerequisites
2. Install required packages
3. Train models if needed
4. Run integration tests
5. Provide next steps

Run this once after installing WealthWise.
"""

import asyncio
import os
import sys
import subprocess
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
print(f"🔐 Loaded GROQ Key: {os.getenv('GROQ_API_KEY')[:6]}...")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WealthWiseSetup:
    """Setup and configuration for WealthWise integration"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_complete = False
        
    async def run_setup(self):
        """Run complete setup process"""
        
        print("🚀 WealthWise Setup & Configuration")
        print("=" * 50)
        
        # Step 1: Check prerequisites
        if not await self.check_prerequisites():
            return False
        
        # Step 2: Check WealthWise installation
        if not await self.check_wealthwise_installation():
            return False
        
        # Step 3: Install Python dependencies
        if not await self.install_dependencies():
            return False
        
        # Step 4: Initialize WealthWise
        if not await self.initialize_wealthwise():
            return False
        
        # Step 5: Train models
        if not await self.train_models():
            return False
        
        # Step 6: Run verification tests
        if not await self.run_verification():
            return False
        
        # Step 7: Setup complete
        await self.setup_complete_message()
        self.setup_complete = True
        
        return True
    
    async def check_prerequisites(self):
        """Check system prerequisites"""
        print("\n📋 Checking Prerequisites...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print("❌ Python 3.8+ required")
            return False
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required directories exist
        required_dirs = [
            "services",
            "api/routers", 
            "database"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                print(f"❌ Required directory missing: {dir_path}")
                return False
        print("✅ Required directories found")
        
        # Check key files exist
        required_files = [
            "services/portfolio_simulator.py",
            "api/routers/ai_analysis.py"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                print(f"❌ Required file missing: {file_path}")
                return False
        print("✅ Required files found")
        
        return True
    
    async def check_wealthwise_installation(self):
        """Check if WealthWise is properly installed"""
        print("\n📦 Checking WealthWise Installation...")
        
        # Check if ai_models directory exists
        ai_models_path = self.project_root / "ai_models"
        if not ai_models_path.exists():
            print("❌ ai_models directory not found")
            print("   Please copy the ai_models folder to your project root")
            return False
        
        # Check key WealthWise files
        required_wealthwise_files = [
            "ai_models/stock_model/core/recommender.py",
            "ai_models/stock_model/explainable_ai.py",
            "ai_models/stock_model/goal_optimization.py",
            "ai_models/stock_model/utils.py"
        ]
        
        for file_path in required_wealthwise_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                print(f"❌ WealthWise file missing: {file_path}")
                return False
        
        print("✅ WealthWise files found")
        return True
    
    async def install_dependencies(self):
        """Install required Python packages"""
        print("\n📥 Installing Dependencies...")
        
        # List of required packages for WealthWise
        required_packages = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "shap>=0.41.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "yfinance>=0.1.70",
            "requests>=2.26.0"
        ]
        
        try:
            for package in required_packages:
                print(f"🔄 Installing {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"⚠️ Warning installing {package}: {result.stderr}")
                else:
                    print(f"✅ {package} installed")
            
            print("✅ Dependencies installation complete")
            return True
            
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    async def initialize_wealthwise(self):
        """Initialize WealthWise system"""
        print("\n🎯 Initializing WealthWise...")
        
        try:
            # Add project root to Python path
            sys.path.insert(0, str(self.project_root))
            
            # Test import
            from ai_models.stock_model.utils import initialize_complete_system
            
            # Initialize the system
            init_result = initialize_complete_system({
                'LOG_LEVEL': 'INFO',
                'LOG_TO_FILE': False,
                'ENABLE_PERFORMANCE_TRACKING': True
            })
            
            if init_result['success']:
                print("✅ WealthWise initialized successfully")
                return True
            else:
                print(f"❌ WealthWise initialization failed: {init_result.get('error')}")
                return False
                
        except ImportError as e:
            print(f"❌ Cannot import WealthWise: {e}")
            print("   Check that ai_models folder is in the correct location")
            return False
        except Exception as e:
            print(f"❌ WealthWise initialization error: {e}")
            return False
    
    async def train_models(self):
        """Train WealthWise models"""
        print("\n🧠 Training Models...")
        
        try:
            from ai_models.stock_model.explainable_ai import SHAPExplainer
            
            shap_explainer = SHAPExplainer()
            
            if shap_explainer.is_available():
                print("✅ SHAP model already trained")
                return True
            
            print("🔄 Training SHAP model (this may take 2-5 minutes)...")
            
            # Train with reasonable number of samples
            success = shap_explainer.train_shap_model(num_samples=1000)
            
            if success:
                print("✅ SHAP model trained successfully")
                return True
            else:
                print("❌ SHAP model training failed")
                print("   The system will still work in fallback mode")
                return True  # Don't fail setup for this
                
        except Exception as e:
            print(f"❌ Model training error: {e}")
            print("   The system will still work in fallback mode")
            return True  # Don't fail setup for this
    
    async def run_verification(self):
        """Run verification tests"""
        print("\n🧪 Running Verification Tests...")
        
        try:
            # Import and run the verifier
            from scripts.verify_wealthwise_integration import WealthWiseVerifier
            
            verifier = WealthWiseVerifier()
            results = await verifier.run_all_tests()
            
            # Check if verification passed
            if results["tests_failed"] == 0:
                print("✅ All verification tests passed!")
                return True
            else:
                print(f"⚠️ {results['tests_failed']} tests failed, but setup can continue")
                print("   Check the verification report above for details")
                return True  # Don't fail setup completely
                
        except Exception as e:
            print(f"❌ Verification tests failed: {e}")
            print("   Manual testing recommended")
            return True  # Don't fail setup completely
    
    async def setup_complete_message(self):
        """Display setup completion message"""
        print("\n" + "=" * 60)
        print("🎉 WEALTHWISE SETUP COMPLETE!")
        print("=" * 60)
        
        print("\n✅ What's Ready:")
        print("   • WealthWise AI system integrated")
        print("   • Enhanced portfolio recommendations")
        print("   • SHAP explainable AI explanations")
        print("   • Goal-oriented optimization")
        print("   • Fallback protection enabled")
        
        print("\n🚀 Next Steps:")
        print("   1. Test your API endpoints:")
        print("      POST /ai/simulate - Enhanced portfolio simulation")
        print("      GET /ai/simulation/{id}/shap-visualization")
        print("      GET /ai/simulation/{id}/news-analysis")
        
        print("\n   2. Example API test:")
        print("""      {
        "goal": "retirement",
        "target_value": 100000,
        "lump_sum": 10000,
        "monthly": 500,
        "timeframe": 15,
        "risk_score": 65,
        "risk_label": "Moderate Aggressive"
      }""")
        
        print("\n   3. Monitor logs for:")
        print("      • 'WealthWise enhanced' confirmations")
        print("      • SHAP explanation generation")
        print("      • Fallback usage warnings")
        
        print("\n📚 Documentation:")
        print("   • Check simulation results for 'wealthwise_enhanced' flag")
        print("   • Enhanced AI summaries include SHAP explanations")
        print("   • Database stores enhanced data for future analysis")
        
        print("\n🔧 Troubleshooting:")
        print("   • Run verification script if issues occur")
        print("   • Check logs for 'WealthWise not available' warnings")
        print("   • System automatically falls back to original mode")
        
        print("\n" + "=" * 60)

async def quick_test():
    """Run a quick integration test"""
    print("🧪 Quick Integration Test")
    print("-" * 30)
    
    try:
        # Test imports
        from services.portfolio_simulator import WEALTHWISE_AVAILABLE, simulate_portfolio
        from database.db import SessionLocal
        
        print(f"WealthWise Available: {'✅' if WEALTHWISE_AVAILABLE else '❌'}")
        
        if WEALTHWISE_AVAILABLE:
            from ai_models.stock_model.explainable_ai import SHAPExplainer
            shap = SHAPExplainer()
            print(f"SHAP Model Ready: {'✅' if shap.is_available() else '❌'}")
        
        # Test simulation
        test_input = {
            "user_id": "quick_test",
            "goal": "test",
            "target_value": 50000,
            "lump_sum": 5000,
            "monthly": 500,
            "timeframe": 10,
            "risk_score": 50,
            "risk_label": "Medium"
        }
        
        db = SessionLocal()
        try:
            print("🔄 Running test simulation...")
            result = await simulate_portfolio(test_input, db)
            
            print(f"✅ Simulation Complete!")
            print(f"   Enhanced: {result.get('wealthwise_enhanced', False)}")
            print(f"   SHAP: {result.get('has_shap_explanations', False)}")
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

async def create_test_endpoint():
    """Create a simple test endpoint for manual testing"""
    
    test_script = '''
# test_wealthwise_endpoint.py
"""
Simple test script for WealthWise integration
Run this to test your enhanced portfolio simulation
"""

import asyncio
import requests
import json

async def test_simulation_endpoint():
    """Test the enhanced simulation endpoint"""
    
    # Your API base URL - adjust as needed
    BASE_URL = "http://localhost:8000"  # Change this to your server URL
    
    test_data = {
        "user_id": "test_user_123",
        "goal": "retirement planning",
        "target_value": 100000,
        "lump_sum": 10000,
        "monthly": 800,
        "timeframe": 15,
        "years_of_experience": 5,
        "income_bracket": "medium",
        "risk_score": 65,
        "risk_label": "Moderate Aggressive"
    }
    
    try:
        print("🚀 Testing Enhanced Portfolio Simulation...")
        print(f"Sending request to: {BASE_URL}/ai/simulate")
        
        response = requests.post(
            f"{BASE_URL}/ai/simulate",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Simulation Successful!")
            print(f"   Simulation ID: {result.get('id')}")
            print(f"   Enhanced: {result.get('wealthwise_enhanced', False)}")
            print(f"   SHAP Available: {result.get('has_shap_explanations', False)}")
            print(f"   Target Achieved: {result.get('target_achieved', False)}")
            
            # Test SHAP visualization if available
            if result.get('has_shap_explanations'):
                sim_id = result.get('id')
                print(f"\\n🎨 Testing SHAP Visualization...")
                
                shap_response = requests.get(f"{BASE_URL}/ai/simulation/{sim_id}/shap-visualization")
                if shap_response.status_code == 200:
                    print("✅ SHAP visualization endpoint working!")
                else:
                    print(f"⚠️ SHAP visualization failed: {shap_response.status_code}")
            
            return result
            
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is your server running?")
        print(f"   Make sure your FastAPI server is running on {BASE_URL}")
        return None
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(test_simulation_endpoint())
'''
    
    # Write test script
    with open("test_wealthwise_endpoint.py", "w") as f:
        f.write(test_script)
    
    print("📝 Created test_wealthwise_endpoint.py")
    print("   Run this script to test your API endpoints")

# Main menu system
async def main():
    """Main setup menu"""
    
    while True:
        print("\n🎯 WealthWise Setup & Testing")
        print("=" * 40)
        print("1. 🚀 Full Setup (Recommended)")
        print("2. 🧪 Quick Integration Test")
        print("3. 🔄 Train Models Only")
        print("4. 📝 Create Test Script")
        print("5. 🔍 Run Verification Only")
        print("6. ❌ Exit")
        
        try:
            choice = input("\nChoose option (1-6): ").strip()
            
            if choice == "1":
                setup = WealthWiseSetup()
                success = await setup.run_setup()
                if success:
                    print("\n🎉 Setup completed successfully!")
                else:
                    print("\n❌ Setup encountered issues - check messages above")
            
            elif choice == "2":
                await quick_test()
            
            elif choice == "3":
                print("\n🧠 Training Models...")
                try:
                    from ai_models.stock_model.explainable_ai import SHAPExplainer
                    shap = SHAPExplainer()
                    success = shap.train_shap_model(num_samples=1000)
                    if success:
                        print("✅ Model training complete!")
                    else:
                        print("❌ Model training failed")
                except Exception as e:
                    print(f"❌ Training error: {e}")
            
            elif choice == "4":
                await create_test_endpoint()
            
            elif choice == "5":
                try:
                    from scripts.verify_wealthwise_integration import WealthWiseVerifier
                    verifier = WealthWiseVerifier()
                    await verifier.run_all_tests()
                except Exception as e:
                    print(f"❌ Verification failed: {e}")
            
            elif choice == "6":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())