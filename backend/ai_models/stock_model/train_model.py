# ai_models/stock_model/train_model.py

"""
Training Script for Enhanced Stock Recommender

This script trains the machine learning component of the stock recommender
to learn patterns from historical data and user preferences.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from enhanced_stock_recommender import EnhancedStockRecommender

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_enhanced_model():
    """
    Train the Enhanced Stock Recommender model.
    
    This creates a machine learning layer on top of the rule-based system
    to learn from patterns in user goals and market performance.
    """
    
    logger.info("🚀 Starting Enhanced Stock Recommender training")
    
    # Initialize the recommender
    recommender = EnhancedStockRecommender()
    
    # Step 1: Generate synthetic training data
    logger.info("📊 Generating synthetic training data...")
    training_data = recommender.generate_training_data(num_samples=2000)
    
    logger.info(f"✅ Generated {len(training_data)} training samples")
    logger.info("Sample data:")
    print(training_data.head())
    
    # Step 2: Train the model
    logger.info("🧠 Training machine learning model...")
    recommender.train_model(training_data)
    
    # Step 3: Test the trained model
    logger.info("🧪 Testing trained model with sample scenarios...")
    
    test_scenarios = [
        {
            "target_value": 50000,
            "timeframe": 10, 
            "risk_score": 40,
            "description": "Conservative long-term retirement goal"
        },
        {
            "target_value": 25000,
            "timeframe": 5,
            "risk_score": 70,
            "description": "Aggressive medium-term house deposit"
        },
        {
            "target_value": 100000,
            "timeframe": 15,
            "risk_score": 55,
            "description": "Moderate long-term wealth building"
        }
    ]
    
    for scenario in test_scenarios:
        logger.info(f"\n🎯 Testing: {scenario['description']}")
        logger.info(f"   Goal: £{scenario['target_value']:,} in {scenario['timeframe']} years")
        logger.info(f"   Risk Score: {scenario['risk_score']}")
        
        # Get recommendations
        recommendations = recommender.recommend_stocks(
            target_value=scenario['target_value'],
            timeframe=scenario['timeframe'],
            risk_score=scenario['risk_score']
        )
        
        logger.info(f"   Recommended stocks: {recommendations}")
    
    logger.info("\n✅ Training completed successfully!")
    logger.info("📁 Model files saved to ai_models/stock_model/models/")
    
    return recommender

def validate_existing_model():
    """
    Check if there's already a trained model and test it.
    """
    logger.info("🔍 Checking for existing trained model...")
    
    recommender = EnhancedStockRecommender()
    
    # Try to load existing model
    if recommender.load_model():
        logger.info("✅ Found existing trained model")
        
        # Test it with a sample recommendation
        test_stocks = recommender.recommend_stocks(
            target_value=50000,
            timeframe=10,
            risk_score=40
        )
        
        logger.info(f"🧪 Test recommendation: {test_stocks}")
        return True
    else:
        logger.info("❌ No existing trained model found")
        return False

def main():
    """
    Main training workflow:
    1. Check if model already exists
    2. If not, train a new one
    3. Validate the model works
    """
    
    print("=" * 60)
    print("🤖 Enhanced Stock Recommender Training Script")
    print("=" * 60)
    
    # Check if we already have a trained model
    if validate_existing_model():
        logger.info("✅ Model is ready to use!")
        
        response = input("\nDo you want to retrain the model? (y/N): ")
        if response.lower() != 'y':
            logger.info("👍 Using existing model")
            return
    
    # Train new model
    logger.info("🚀 Training new model...")
    trained_model = train_enhanced_model()
    
    # Final validation
    logger.info("\n🏁 Final validation...")
    if trained_model.model is not None:
        logger.info("✅ Model training successful!")
        logger.info("💡 Your enhanced stock recommender is ready to use")
        
        # Show model info
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        if os.path.exists(model_dir):
            model_files = os.listdir(model_dir)
            logger.info(f"📁 Model files created: {model_files}")
    else:
        logger.error("❌ Model training failed")

if __name__ == "__main__":
    main()