"""
Model Management Utilities

This module provides utilities for managing, training, and maintaining
the machine learning models used in the WealthWise recommendation system.

Key Features:
1. Model training and validation
2. Model persistence and loading
3. Model performance monitoring
4. Automatic model updating
5. Model version management
"""

import os
import json
import joblib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import hashlib

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Centralized model management system for all ML components
    
    Handles training, saving, loading, and monitoring of all models
    used in the recommendation system.
    """
    
    def __init__(self, models_dir: str = "./ai_models/stock_model/models"):
        """
        Initialize model manager
        
        Args:
            models_dir: Directory to store model files
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Model registry - tracks all available models
        self.model_registry = {
            "shap_explainer": {
                "model_file": "shap_portfolio_quality_model.joblib",
                "scaler_file": "shap_portfolio_quality_scaler.joblib",
                "metadata_file": "shap_training_metadata.json",
                "description": "SHAP explainable AI model for portfolio quality prediction",
                "training_function": self._train_shap_model
            },
            "factor_weights": {
                "model_file": "factor_analysis_weights.json",
                "metadata_file": "factor_weights_metadata.json", 
                "description": "Optimized factor weights for different market regimes",
                "training_function": self._train_factor_weights
            },
            "correlation_cache": {
                "model_file": "correlation_cache.pickle",
                "metadata_file": "correlation_metadata.json",
                "description": "Cached correlation matrices for portfolio optimization",
                "training_function": self._build_correlation_cache
            }
        }
        
        self.metadata_file = os.path.join(models_dir, "model_registry.json")
        self.load_model_metadata()
    
    def get_model_path(self, model_name: str, file_type: str = "model_file") -> str:
        """Get full path to model file"""
        if model_name not in self.model_registry:
            raise ValueError(f"Unknown model: {model_name}")
        
        filename = self.model_registry[model_name][file_type]
        return os.path.join(self.models_dir, filename)
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if model is trained and available"""
        if model_name not in self.model_registry:
            return False
        
        model_path = self.get_model_path(model_name, "model_file")
        return os.path.exists(model_path)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        if model_name not in self.model_registry:
            return {"error": f"Unknown model: {model_name}"}
        
        model_info = self.model_registry[model_name].copy()
        model_info["available"] = self.is_model_available(model_name)
        
        # Load metadata if available
        metadata_path = self.get_model_path(model_name, "metadata_file")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                model_info["metadata"] = metadata
            except Exception as e:
                model_info["metadata_error"] = str(e)
        
        return model_info
    
    def train_model(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Train a specific model
        
        Args:
            model_name: Name of model to train
            **kwargs: Training parameters specific to the model
            
        Returns:
            Dict with training results
        """
        if model_name not in self.model_registry:
            return {"success": False, "error": f"Unknown model: {model_name}"}
        
        logger.info(f"🚀 Training model: {model_name}")
        
        try:
            # Get training function
            training_function = self.model_registry[model_name]["training_function"]
            
            # Train the model
            result = training_function(**kwargs)
            
            if result.get("success", False):
                # Update metadata
                self._update_model_metadata(model_name, result)
                logger.info(f"✅ Successfully trained {model_name}")
            else:
                logger.error(f"❌ Failed to train {model_name}: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Exception training {model_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def train_all_models(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train all models in the registry
        
        Args:
            force_retrain: Whether to retrain even if models exist
            
        Returns:
            Dict with results for each model
        """
        logger.info("🚀 Training all models...")
        
        results = {}
        
        for model_name in self.model_registry.keys():
            if not force_retrain and self.is_model_available(model_name):
                logger.info(f"⏭️ Skipping {model_name} (already trained)")
                results[model_name] = {"success": True, "skipped": True}
                continue
            
            results[model_name] = self.train_model(model_name)
        
        # Summary
        successful = sum(1 for r in results.values() if r.get("success", False))
        total = len(results)
        
        logger.info(f"✅ Model training complete: {successful}/{total} successful")
        
        return {
            "overall_success": successful == total,
            "successful_count": successful,
            "total_count": total,
            "individual_results": results
        }
    
    def load_model_metadata(self) -> Dict[str, Any]:
        """Load model metadata from disk"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model metadata: {e}")
        
        return {}
    
    def save_model_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save model metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")
    
    def _update_model_metadata(self, model_name: str, training_result: Dict[str, Any]) -> None:
        """Update metadata for a specific model"""
        metadata = self.load_model_metadata()
        
        if model_name not in metadata:
            metadata[model_name] = {}
        
        metadata[model_name].update({
            "last_trained": datetime.now().isoformat(),
            "training_result": training_result,
            "model_available": True
        })
        
        self.save_model_metadata(metadata)
    
    def get_model_status_summary(self) -> Dict[str, Any]:
        """Get summary of all model statuses"""
        summary = {
            "total_models": len(self.model_registry),
            "available_models": 0,
            "missing_models": [],
            "model_details": {}
        }
        
        for model_name in self.model_registry.keys():
            available = self.is_model_available(model_name)
            if available:
                summary["available_models"] += 1
            else:
                summary["missing_models"].append(model_name)
            
            summary["model_details"][model_name] = {
                "available": available,
                "description": self.model_registry[model_name]["description"]
            }
        
        summary["all_models_available"] = summary["available_models"] == summary["total_models"]
        
        return summary
    
    def cleanup_old_models(self, days_old: int = 30) -> Dict[str, Any]:
        """
        Clean up old model files
        
        Args:
            days_old: Remove files older than this many days
            
        Returns:
            Dict with cleanup results
        """
        logger.info(f"🧹 Cleaning up models older than {days_old} days...")
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        removed_files = []
        errors = []
        
        try:
            for filename in os.listdir(self.models_dir):
                filepath = os.path.join(self.models_dir, filename)
                
                if os.path.isfile(filepath):
                    file_mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_mod_time < cutoff_date:
                        # Don't remove current models
                        is_current_model = Any(
                            filename in model_config.values() 
                            for model_config in self.model_registry.values()
                        )
                        
                        if not is_current_model:
                            try:
                                os.remove(filepath)
                                removed_files.append(filename)
                                logger.debug(f"Removed old file: {filename}")
                            except Exception as e:
                                errors.append(f"Failed to remove {filename}: {e}")
        
        except Exception as e:
            errors.append(f"Error during cleanup: {e}")
        
        logger.info(f"✅ Cleanup complete: removed {len(removed_files)} files")
        
        return {
            "removed_files": removed_files,
            "errors": errors,
            "cleanup_successful": len(errors) == 0
        }
    
    # Model-specific training functions
    
    def _train_shap_model(self, num_samples: int = 2000) -> Dict[str, Any]:
        """Train SHAP explainer model"""
        try:
            from ai_models.stock_model.explainable_ai import SHAPExplainer
            
            explainer = SHAPExplainer()
            success = explainer.train_shap_model(num_samples=num_samples)
            
            if success:
                # Save the model
                model_path = self.get_model_path("shap_explainer", "model_file").replace(".joblib", "")
                explainer.save_model(model_path)
                
                return {
                    "success": True,
                    "num_samples": num_samples,
                    "model_type": "RandomForestRegressor",
                    "features": explainer.feature_names
                }
            else:
                return {"success": False, "error": "SHAP model training failed"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _train_factor_weights(self) -> Dict[str, Any]:
        """Train/optimize factor weights for different market regimes"""
        try:
            from ai_models.stock_model.analysis import FactorAnalyzer
            
            analyzer = FactorAnalyzer()
            
            # Define different market regimes and optimize weights for each
            regimes = ["bull", "bear", "sideways", "high_volatility", "low_volatility"]
            timeframes = [5, 10, 15, 20]
            
            optimized_weights = {}
            
            for regime in regimes:
                regime_weights = {}
                for timeframe in timeframes:
                    weights = analyzer.get_factor_weights_for_regime(regime, timeframe)
                    regime_weights[f"timeframe_{timeframe}"] = weights
                
                optimized_weights[regime] = regime_weights
            
            # Save the weights
            weights_path = self.get_model_path("factor_weights", "model_file")
            with open(weights_path, 'w') as f:
                json.dump(optimized_weights, f, indent=2)
            
            return {
                "success": True,
                "regimes_count": len(regimes),
                "timeframes_count": len(timeframes),
                "total_configurations": len(regimes) * len(timeframes)
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _build_correlation_cache(self) -> Dict[str, Any]:
        """Build correlation cache for common stock combinations"""
        try:
            from ai_models.stock_model.analysis import PortfolioOptimizer
            from ai_models.stock_model.core.config import ASSET_UNIVERSES
            
            optimizer = PortfolioOptimizer()
            
            # Build correlation matrices for common portfolio combinations
            correlation_cache = {}
            
            for risk_category, universe in ASSET_UNIVERSES.items():
                # Get all stocks from this risk category
                all_stocks = []
                for category, stocks in universe.items():
                    if isinstance(stocks, list):
                        all_stocks.extend(stocks)
                
                if len(all_stocks) > 3:
                    # Calculate correlation matrix
                    try:
                        dummy_weights = {stock: 1.0/len(all_stocks) for stock in all_stocks}
                        # This will populate the correlation cache
                        optimizer.optimize_for_diversification(all_stocks, dummy_weights)
                        
                        correlation_cache[risk_category] = {
                            "stocks": all_stocks,
                            "last_updated": datetime.now().isoformat(),
                            "status": "success"
                        }
                    except Exception as e:
                        correlation_cache[risk_category] = {
                            "stocks": all_stocks,
                            "last_updated": datetime.now().isoformat(),
                            "status": "failed",
                            "error": str(e)
                        }
            
            # Save the cache metadata
            cache_path = self.get_model_path("correlation_cache", "model_file")
            with open(cache_path, 'wb') as f:
                pickle.dump(correlation_cache, f)
            
            successful_categories = sum(1 for cache in correlation_cache.values() if cache["status"] == "success")
            
            return {
                "success": True,
                "categories_processed": len(correlation_cache),
                "successful_categories": successful_categories,
                "cache_entries": list(correlation_cache.keys())
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}


def initialize_models(models_dir: str = "./ai_models/stock_model/models", 
                     force_retrain: bool = False) -> Dict[str, Any]:
    """
    Initialize all models for the recommendation system
    
    Args:
        models_dir: Directory to store models
        force_retrain: Whether to retrain existing models
        
    Returns:
        Dict with initialization results
    """
    logger.info("🚀 Initializing model system...")
    
    try:
        # Create model manager
        manager = ModelManager(models_dir)
        
        # Check current status
        status = manager.get_model_status_summary()
        logger.info(f"📊 Current status: {status['available_models']}/{status['total_models']} models available")
        
        # Train models if needed
        if not status["all_models_available"] or force_retrain:
            training_results = manager.train_all_models(force_retrain=force_retrain)
            
            return {
                "success": training_results["overall_success"],
                "manager": manager,
                "training_results": training_results,
                "final_status": manager.get_model_status_summary()
            }
        else:
            logger.info("✅ All models already available")
            return {
                "success": True,
                "manager": manager,
                "training_results": {"message": "All models already available"},
                "final_status": status
            }
    
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_model_manager(models_dir: str = "./ai_models/stock_model/models") -> ModelManager:
    """
    Get a configured model manager instance
    
    Args:
        models_dir: Directory containing models
        
    Returns:
        ModelManager instance
    """
    return ModelManager(models_dir)


def check_model_requirements() -> Dict[str, Any]:
    """
    Check if all required dependencies for models are available
    
    Returns:
        Dict with requirement check results
    """
    requirements = {
        "required_packages": ["scikit-learn", "joblib", "numpy", "pandas"],
        "optional_packages": ["shap"],
        "system_requirements": ["write_access", "disk_space"]
    }
    
    results = {
        "all_requirements_met": True,
        "missing_requirements": [],
        "package_status": {},
        "system_status": {}
    }
    
    # Check required packages
    for package in requirements["required_packages"]:
        try:
            __import__(package)
            results["package_status"][package] = True
        except ImportError:
            results["package_status"][package] = False
            results["all_requirements_met"] = False
            results["missing_requirements"].append(f"pip install {package}")
    
    # Check optional packages
    for package in requirements["optional_packages"]:
        try:
            __import__(package)
            results["package_status"][package] = True
        except ImportError:
            results["package_status"][package] = False
            # Optional packages don't affect overall status
    
    # Check system requirements
    try:
        import tempfile
        import shutil
        
        # Check write access
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            results["system_status"]["write_access"] = True
        
        # Check disk space (basic check)
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        results["system_status"]["disk_space_gb"] = free_gb
        results["system_status"]["sufficient_disk_space"] = free_gb > 0.1  # Need at least 100MB
        
    except Exception as e:
        results["system_status"]["error"] = str(e)
        results["all_requirements_met"] = False
    
    return results