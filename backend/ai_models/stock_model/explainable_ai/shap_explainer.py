"""
SHAP Explainable AI Module

This module implements SHAP (SHapley Additive exPlanations) to make AI investment
decisions transparent and educational. It provides clear explanations for why
specific portfolios are recommended.

Key Features:
1. SHAP-powered explanations for portfolio recommendations
2. Machine learning model training for portfolio quality prediction
3. Human-readable explanations for each recommendation factor
4. Transparent decision-making process for educational purposes
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

# SHAP Integration for Explainable AI
try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP explainability module loaded successfully")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap matplotlib seaborn")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-Powered Explainable AI System for Investment Recommendations
    
    This class makes AI investment decisions transparent by using SHAP values
    to explain exactly why certain portfolios are recommended. It trains a
    machine learning model and uses SHAP to provide human-readable explanations.
    
    Key Components:
    1. Portfolio Quality Prediction Model (Random Forest)
    2. SHAP TreeExplainer for transparent explanations
    3. Human-readable explanation generation
    4. Feature importance analysis
    """
    
    def __init__(self):
        """Initialize the SHAP explainer system"""
        # Core ML models and scalers
        self.model = None
        self.scaler = StandardScaler()
        self.shap_explainer = None
        self.shap_values = None
        
        # Feature configuration for SHAP explanations
        self.feature_names = [
            "target_value_log",     # Log of investment target
            "timeframe",            # Investment horizon in years
            "risk_score",           # User's risk tolerance (0-100)
            "required_return",      # Annual return needed to reach goal
            "monthly_contribution", # Regular monthly investment amount
            "market_volatility",    # Current market volatility (VIX level)
            "market_trend_score"    # Market momentum indicator (0-5)
        ]
        
        # Training data cache
        self._training_cache = None
        self._model_trained = False
    
    def is_available(self) -> bool:
        """Check if SHAP is available for explainable AI"""
        return SHAP_AVAILABLE
    
    def prepare_features_for_shap(self, target_value: float, timeframe: int, 
                                 risk_score: float, current_investment: float = 0, 
                                 monthly_contribution: float = 0,
                                 market_volatility: float = 20.0,
                                 market_trend: float = 2.5) -> np.ndarray:
        """
        Prepare feature vector for SHAP explainable AI analysis
        
        Converts user inputs into standardized format for ML model input
        
        Args:
            target_value: User's financial goal
            timeframe: Years to reach goal
            risk_score: Risk tolerance 0-100
            current_investment: Starting amount
            monthly_contribution: Regular investments
            market_volatility: Current VIX level
            market_trend: Market momentum score
            
        Returns:
            numpy array with prepared features
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, returning dummy features")
            return np.array([[0, 0, 0, 0, 0, 0, 0]])
        
        try:
            # Calculate required return for goal achievement
            total_contributions = current_investment + (monthly_contribution * 12 * timeframe)
            if total_contributions > 0:
                required_multiplier = target_value / total_contributions
                required_return = (required_multiplier ** (1/timeframe)) - 1
                required_return = max(0.0, min(0.25, required_return))
            else:
                required_return = 0.10  # Default 10%
            
            # Create feature vector with proper scaling
            features = np.array([[
                np.log(max(1000, target_value)),        # Log transform for scaling
                float(timeframe),                       # Years to goal
                float(risk_score),                      # Risk tolerance
                float(required_return),                 # Required annual return
                float(monthly_contribution),            # Monthly investment
                float(market_volatility),               # Current VIX level
                float(market_trend)                     # Market trend strength
            ]])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return np.array([[0, 0, 0, 0, 0, 0, 0]])
    
    def train_shap_model(self, num_samples: int = 2000) -> bool:
        """
        Train machine learning model for SHAP explanations
        
        Creates a Random Forest model that predicts portfolio quality scores
        based on user inputs and market conditions. Designed specifically
        for SHAP explainability.
        
        Args:
            num_samples: Number of training examples to generate
            
        Returns:
            bool: True if training successful, False otherwise
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping model training")
            return False
        
        logger.info(f"🤖 Training SHAP explainability model with {num_samples} samples...")
        
        try:
            # Generate realistic training data
            X_train, y_train = self._generate_training_data(num_samples)
            
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100,       # 100 trees for good performance
                max_depth=15,           # Deep enough for complex patterns
                min_samples_split=5,    # Prevent overfitting
                random_state=42,        # Reproducible results
                n_jobs=-1              # Use all CPU cores
            )
            
            # Fit scaler and model
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_train_scaled, y_train)
            
            # Initialize SHAP explainer
            self.shap_explainer = shap.TreeExplainer(self.model)
            
            # Test the model
            test_features = self.prepare_features_for_shap(50000, 10, 60, 5000, 500)
            test_scaled = self.scaler.transform(test_features)
            test_prediction = self.model.predict(test_scaled)[0]
            
            self._model_trained = True
            
            logger.info(f"✅ SHAP model training complete!")
            logger.info(f"📊 Test prediction: {test_prediction:.1f}/100 portfolio quality score")
            
            return True
            
        except Exception as e:
            logger.error(f"SHAP model training failed: {e}")
            self.shap_explainer = None
            self._model_trained = False
            return False
    
    def get_shap_explanation(self, target_value: float, timeframe: int, 
                           risk_score: float, current_investment: float = 0,
                           monthly_contribution: float = 0,
                           market_volatility: float = 20.0,
                           market_trend: float = 2.5) -> Dict[str, Any]:
        """
        Generate SHAP-based explanation for portfolio recommendations
        
        This is the core function that makes AI decisions transparent and educational.
        Uses SHAP values to explain exactly why the AI recommends a particular portfolio.
        
        Args:
            User's financial parameters and market conditions
            
        Returns:
            Dict with comprehensive SHAP explanation:
            {
                "portfolio_quality_score": 78.5,
                "human_readable_explanation": {...},
                "feature_contributions": {...},
                "transparency_metrics": {...}
            }
        """
        if not SHAP_AVAILABLE:
            return {
                "error": "SHAP not available. Install with: pip install shap matplotlib seaborn",
                "portfolio_quality_score": 50,
                "explanation": "SHAP explainability not configured"
            }
        
        # Ensure model is trained
        if not self._model_trained or self.shap_explainer is None:
            logger.info("🔧 SHAP model not found, training now...")
            if not self.train_shap_model():
                return {"error": "Failed to initialize SHAP explainer"}
        
        try:
            logger.debug("🔍 Generating SHAP explanation for portfolio recommendation...")
            
            # Prepare features for analysis
            features = self.prepare_features_for_shap(
                target_value, timeframe, risk_score, current_investment, 
                monthly_contribution, market_volatility, market_trend
            )
            features_scaled = self.scaler.transform(features)
            
            # Get model prediction (portfolio quality score)
            portfolio_score = self.model.predict(features_scaled)[0]
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(features_scaled)
            
            # Create comprehensive explanation
            explanation = {
                "portfolio_quality_score": portfolio_score,
                "base_value": self.shap_explainer.expected_value,
                "feature_contributions": dict(zip(self.feature_names, shap_values[0])),
                "human_readable_explanation": self._create_human_explanation(
                    shap_values[0], features[0]
                ),
                "transparency_metrics": self._calculate_transparency_metrics(shap_values[0])
            }
            
            logger.info(f"🔍 SHAP explanation complete: {portfolio_score:.1f}/100 quality score")
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP explanation generation failed: {e}")
            return {"error": f"Could not generate explanation: {str(e)}"}
    
    def _generate_training_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic but realistic training data for SHAP model
        
        Creates training examples representing realistic user scenarios
        and calculates corresponding portfolio quality scores.
        
        Returns:
            Tuple of (X, y) where X is features and y is quality scores
        """
        X = []
        y = []
        
        logger.info("📝 Generating realistic training scenarios...")
        
        for i in range(num_samples):
            # Generate realistic user parameters
            target_value = np.random.lognormal(np.log(75000), 1.0)
            target_value = max(5000, min(1000000, target_value))
            
            # Common investment timeframes
            timeframe = np.random.choice([1, 2, 3, 5, 7, 10, 15, 20, 25, 30])
            
            # Risk scores with bias toward moderate
            risk_score = np.random.beta(2.5, 2.5) * 100
            
            # Starting investments and contributions
            current_investment = np.random.uniform(0, target_value * 0.3)
            monthly_contribution = np.random.uniform(0, target_value * 0.05)
            
            # Calculate derived features
            total_contributions = current_investment + (monthly_contribution * 12 * timeframe)
            if total_contributions > 0:
                required_multiplier = target_value / total_contributions
                required_return = (required_multiplier ** (1/timeframe)) - 1
                required_return = max(0.0, min(0.25, required_return))
            else:
                required_return = 0.10
            
            # Simulate market conditions
            market_volatility = max(5, min(50, np.random.normal(20, 8)))
            market_trend = np.random.uniform(0, 5)
            
            # Create feature vector
            features = [
                np.log(target_value),
                timeframe,
                risk_score,
                required_return,
                monthly_contribution,
                market_volatility,
                market_trend
            ]
            
            # Calculate portfolio quality score
            quality_score = self._calculate_synthetic_quality_score(
                target_value, timeframe, risk_score, required_return,
                monthly_contribution, market_volatility, market_trend
            )
            
            X.append(features)
            y.append(quality_score)
            
            if (i + 1) % 500 == 0:
                logger.debug(f"Generated {i + 1}/{num_samples} training examples...")
        
        return np.array(X), np.array(y)
    
    def _calculate_synthetic_quality_score(self, target_value: float, timeframe: int,
                                         risk_score: float, required_return: float,
                                         monthly_contribution: float, 
                                         market_volatility: float,
                                         market_trend: float) -> float:
        """
        Calculate synthetic portfolio quality score for training
        
        Simulates how "good" a portfolio recommendation would be for given parameters
        """
        base_score = 50.0  # Start with neutral score
        
        # Factor 1: Goal achievability
        if required_return <= 0.07:  # Very achievable
            base_score += 25
        elif required_return <= 0.12:  # Achievable
            base_score += 10
        elif required_return > 0.15:  # Very ambitious
            base_score -= 20
        
        # Factor 2: Time horizon advantage
        if timeframe >= 15:  # Long-term advantage
            base_score += 15
        elif timeframe >= 7:  # Medium-term
            base_score += 10
        elif timeframe <= 2:  # Short-term challenge
            base_score -= 15
        
        # Factor 3: Risk appropriateness
        if 30 <= risk_score <= 70:  # Balanced approach
            base_score += 10
        elif risk_score > 85:  # Very aggressive
            base_score -= 5
        elif risk_score < 15:  # Very conservative
            base_score -= 5
        
        # Factor 4: Market conditions
        if market_volatility > 35:  # High volatility
            base_score -= 10
        elif market_volatility < 15:  # Low volatility
            base_score += 8
        
        if market_trend > 3.5:  # Strong uptrend
            base_score += 12
        elif market_trend < 1.5:  # Weak trend
            base_score -= 10
        
        # Factor 5: Contribution consistency
        annual_contributions = monthly_contribution * 12
        contribution_ratio = annual_contributions / max(1000, target_value)
        
        if 0.05 <= contribution_ratio <= 0.20:  # Good contribution rate
            base_score += 10
        elif contribution_ratio > 0.25:  # Very high contributions
            base_score += 15
        elif contribution_ratio < 0.02:  # Low contributions
            base_score -= 10
        
        # Add realistic noise
        noise = np.random.normal(0, 5)
        final_score = base_score + noise
        
        # Bound between 0 and 100
        return max(0, min(100, final_score))
    
    def _create_human_explanation(self, shap_values: np.ndarray, 
                                 feature_values: np.ndarray) -> Dict[str, str]:
        """
        Convert SHAP values into human-readable explanations
        
        Translates mathematical SHAP values into plain English explanations
        that users can understand and learn from.
        """
        explanations = {}
        
        for i, (feature, shap_val, feature_val) in enumerate(
            zip(self.feature_names, shap_values, feature_values)
        ):
            # Determine impact magnitude
            if abs(shap_val) < 0.5:
                impact = "minimal impact"
            elif shap_val > 2.0:
                impact = "very strong positive impact"
            elif shap_val > 0.5:
                impact = "positive impact"
            elif shap_val < -2.0:
                impact = "very strong negative impact"
            else:
                impact = "negative impact"
            
            # Create specific explanations for each feature
            if feature == "target_value_log":
                actual_target = np.exp(feature_val)
                if shap_val > 0:
                    explanations[feature] = (
                        f"Your £{actual_target:,.0f} target has {impact} - "
                        f"ambitious goals drive growth-focused strategies"
                    )
                else:
                    explanations[feature] = (
                        f"Your £{actual_target:,.0f} target has {impact} - "
                        f"modest goals allow for conservative approaches"
                    )
            
            elif feature == "timeframe":
                if shap_val > 0:
                    explanations[feature] = (
                        f"Your {feature_val:.0f}-year timeframe has {impact} - "
                        f"longer periods enable compound growth and recovery"
                    )
                else:
                    explanations[feature] = (
                        f"Your {feature_val:.0f}-year timeframe has {impact} - "
                        f"shorter periods require careful risk management"
                    )
            
            elif feature == "risk_score":
                if shap_val > 0:
                    explanations[feature] = (
                        f"Your risk tolerance ({feature_val:.0f}/100) has {impact} - "
                        f"higher risk comfort enables growth investments"
                    )
                else:
                    explanations[feature] = (
                        f"Your risk tolerance ({feature_val:.0f}/100) has {impact} - "
                        f"conservative approach prioritizes stability"
                    )
            
            elif feature == "required_return":
                if shap_val > 0:
                    explanations[feature] = (
                        f"Required return ({feature_val:.1%} annually) has {impact} - "
                        f"ambitious targets drive growth allocation"
                    )
                else:
                    explanations[feature] = (
                        f"Required return ({feature_val:.1%} annually) has {impact} - "
                        f"high requirements may conflict with risk tolerance"
                    )
            
            elif feature == "monthly_contribution":
                if shap_val > 0:
                    explanations[feature] = (
                        f"Monthly contributions (£{feature_val:.0f}) have {impact} - "
                        f"regular investing smooths market volatility"
                    )
                else:
                    explanations[feature] = (
                        f"Monthly contributions (£{feature_val:.0f}) have {impact} - "
                        f"low investments increase market dependence"
                    )
            
            elif feature == "market_volatility":
                if shap_val > 0:
                    explanations[feature] = (
                        f"Market volatility ({feature_val:.1f} VIX) has {impact} - "
                        f"uncertainty creates opportunities for patient investors"
                    )
                else:
                    explanations[feature] = (
                        f"Market volatility ({feature_val:.1f} VIX) has {impact} - "
                        f"high uncertainty suggests defensive positioning"
                    )
            
            elif feature == "market_trend_score":
                if shap_val > 0:
                    explanations[feature] = (
                        f"Market trend ({feature_val:.1f}/5) has {impact} - "
                        f"strong trends support growth strategies"
                    )
                else:
                    explanations[feature] = (
                        f"Market trend ({feature_val:.1f}/5) has {impact} - "
                        f"weak trends suggest defensive positioning"
                    )
        
        return explanations
    
    def _calculate_transparency_metrics(self, shap_values: np.ndarray) -> Dict[str, Any]:
        """
        Calculate metrics about the explanation quality and transparency
        """
        return {
            "explanation_strength": float(np.sum(np.abs(shap_values))),
            "most_important_factor": self.feature_names[np.argmax(np.abs(shap_values))],
            "positive_factors": int(len([v for v in shap_values if v > 0.5])),
            "negative_factors": int(len([v for v in shap_values if v < -0.5])),
            "confidence": float(min(1.0, np.sum(np.abs(shap_values)) / 10.0))
        }
    
    def save_model(self, model_path: str) -> bool:
        """
        Save the trained SHAP model and scaler
        
        Args:
            model_path: Path to save the model files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.model is not None and self._model_trained:
                # Save model and scaler
                joblib.dump(self.model, f"{model_path}_model.joblib")
                joblib.dump(self.scaler, f"{model_path}_scaler.joblib")
                
                logger.info(f"✅ SHAP model saved to {model_path}")
                return True
            else:
                logger.warning("No trained model to save")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a pre-trained SHAP model and scaler
        
        Args:
            model_path: Path to load the model files from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load model and scaler
            self.model = joblib.load(f"{model_path}_model.joblib")
            self.scaler = joblib.load(f"{model_path}_scaler.joblib")
            
            # Initialize SHAP explainer if SHAP is available
            if SHAP_AVAILABLE:
                self.shap_explainer = shap.TreeExplainer(self.model)
                self._model_trained = True
                
                logger.info(f"✅ SHAP model loaded from {model_path}")
                return True
            else:
                logger.warning("SHAP not available, model loaded but explanations disabled")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False