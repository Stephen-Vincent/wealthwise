"""
Serialization Utils Module

This module provides comprehensive JSON serialization utilities for handling
NumPy arrays, pandas objects, and complex nested data structures commonly
found in AI model outputs and portfolio simulation results.
"""

import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union, Optional
from datetime import datetime, date
import logging
import warnings

logger = logging.getLogger(__name__)

class SerializationManager:
    """
    Comprehensive serialization manager for portfolio simulation data.
    
    Handles:
    - NumPy arrays and scalars
    - Pandas DataFrames and Series
    - DateTime objects
    - Complex nested structures
    - SHAP explanation data
    - Portfolio optimization results
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the serialization manager.
        
        Args:
            strict_mode: If True, raises exceptions on serialization failures
        """
        self.strict_mode = strict_mode
        self.conversion_log = []
        logger.info(f"🔧 SerializationManager initialized (strict_mode: {strict_mode})")
    
    def serialize_for_json(self, data: Any, path: str = "root") -> Any:
        """
        Recursively convert any object to JSON-serializable format.
        
        Args:
            data: Object to serialize
            path: Current path in the object tree (for debugging)
            
        Returns:
            JSON-serializable object
        """
        
        try:
            # Handle None values
            if data is None:
                return None
            
            # Handle NumPy types
            if isinstance(data, np.ndarray):
                return self._serialize_numpy_array(data, path)
            elif isinstance(data, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(data)
            elif isinstance(data, (np.floating, np.float64, np.float32, np.float16)):
                return float(data)
            elif isinstance(data, np.bool_):
                return bool(data)
            elif isinstance(data, (np.str_, np.unicode_)):
                return str(data)
            elif isinstance(data, np.complex128):
                return {"real": float(data.real), "imag": float(data.imag), "_type": "complex"}
            
            # Handle pandas types
            elif isinstance(data, pd.Series):
                return self._serialize_pandas_series(data, path)
            elif isinstance(data, pd.DataFrame):
                return self._serialize_pandas_dataframe(data, path)
            elif isinstance(data, pd.Timestamp):
                return data.isoformat()
            elif isinstance(data, pd.Timedelta):
                return str(data)
            
            # Handle datetime objects
            elif isinstance(data, (datetime, date)):
                return data.isoformat()
            
            # Handle complex numbers
            elif isinstance(data, complex):
                return {"real": data.real, "imag": data.imag, "_type": "complex"}
            
            # Handle dictionaries
            elif isinstance(data, dict):
                return self._serialize_dict(data, path)
            
            # Handle lists and tuples
            elif isinstance(data, (list, tuple)):
                return self._serialize_sequence(data, path)
            
            # Handle sets
            elif isinstance(data, set):
                return list(data)
            
            # Handle custom objects with __dict__
            elif hasattr(data, '__dict__'):
                return self._serialize_custom_object(data, path)
            
            # Handle objects with to_dict method
            elif hasattr(data, 'to_dict') and callable(getattr(data, 'to_dict')):
                return self.serialize_for_json(data.to_dict(), f"{path}.to_dict()")
            
            # Handle objects with tolist method (like some NumPy objects)
            elif hasattr(data, 'tolist') and callable(getattr(data, 'tolist')):
                return self.serialize_for_json(data.tolist(), f"{path}.tolist()")
            
            # Handle basic JSON-serializable types
            elif isinstance(data, (str, int, float, bool)):
                return data
            
            # Handle bytes
            elif isinstance(data, bytes):
                return data.decode('utf-8', errors='ignore')
            
            # Fallback for unknown types
            else:
                return self._handle_unknown_type(data, path)
                
        except Exception as e:
            return self._handle_serialization_error(data, path, e)
    
    def _serialize_numpy_array(self, arr: np.ndarray, path: str) -> Any:
        """Serialize NumPy arrays with size limits and type preservation."""
        
        try:
            # Check array size to prevent memory issues
            if arr.size > 100000:  # 100k elements limit
                logger.warning(f"⚠️ Large array at {path}: {arr.shape} ({arr.size} elements)")
                if self.strict_mode:
                    raise ValueError(f"Array too large for serialization: {arr.shape}")
                return {
                    "_type": "large_array",
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "size": int(arr.size),
                    "sample": arr.flatten()[:100].tolist()  # First 100 elements as sample
                }
            
            # Handle different array types
            if arr.dtype.kind in ['U', 'S']:  # String arrays
                return arr.astype(str).tolist()
            elif arr.dtype.kind == 'O':  # Object arrays
                return [self.serialize_for_json(item, f"{path}[{i}]") for i, item in enumerate(arr)]
            elif arr.dtype.kind in ['c']:  # Complex arrays
                return [{"real": float(x.real), "imag": float(x.imag)} for x in arr]
            else:
                # Numeric arrays
                result = arr.tolist()
                self.conversion_log.append(f"Converted NumPy array at {path}: {arr.shape} {arr.dtype}")
                return result
                
        except Exception as e:
            logger.warning(f"⚠️ NumPy array serialization failed at {path}: {e}")
            return {"_error": f"NumPy array serialization failed: {str(e)}", "_shape": str(getattr(arr, 'shape', 'unknown'))}
    
    def _serialize_pandas_series(self, series: pd.Series, path: str) -> Any:
        """Serialize pandas Series with index preservation."""
        
        try:
            if len(series) > 10000:  # Large series limit
                logger.warning(f"⚠️ Large Series at {path}: {len(series)} elements")
                if self.strict_mode:
                    raise ValueError(f"Series too large for serialization: {len(series)}")
                return {
                    "_type": "large_series",
                    "length": len(series),
                    "dtype": str(series.dtype),
                    "sample": series.head(100).to_dict()
                }
            
            # Regular series serialization
            result = {
                "values": self.serialize_for_json(series.values, f"{path}.values"),
                "index": self.serialize_for_json(series.index.tolist(), f"{path}.index"),
                "name": series.name,
                "_type": "pandas_series"
            }
            
            self.conversion_log.append(f"Converted pandas Series at {path}: {len(series)} elements")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Pandas Series serialization failed at {path}: {e}")
            return {"_error": f"Pandas Series serialization failed: {str(e)}"}
    
    def _serialize_pandas_dataframe(self, df: pd.DataFrame, path: str) -> Any:
        """Serialize pandas DataFrame with multiple format options."""
        
        try:
            if len(df) * len(df.columns) > 50000:  # Large DataFrame limit
                logger.warning(f"⚠️ Large DataFrame at {path}: {df.shape}")
                if self.strict_mode:
                    raise ValueError(f"DataFrame too large for serialization: {df.shape}")
                return {
                    "_type": "large_dataframe",
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "sample": df.head(50).to_dict('records')
                }
            
            # Choose serialization method based on DataFrame characteristics
            if df.index.name or not df.index.equals(pd.RangeIndex(len(df))):
                # Preserve custom index
                result = {
                    "data": df.to_dict('records'),
                    "index": self.serialize_for_json(df.index.tolist(), f"{path}.index"),
                    "columns": list(df.columns),
                    "_type": "pandas_dataframe_with_index"
                }
            else:
                # Simple records format for default index
                result = df.to_dict('records')
            
            self.conversion_log.append(f"Converted pandas DataFrame at {path}: {df.shape}")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Pandas DataFrame serialization failed at {path}: {e}")
            return {"_error": f"Pandas DataFrame serialization failed: {str(e)}"}
    
    def _serialize_dict(self, data: dict, path: str) -> dict:
        """Serialize dictionary with key conversion."""
        
        try:
            result = {}
            for key, value in data.items():
                # Convert non-string keys to strings
                str_key = str(key)
                result[str_key] = self.serialize_for_json(value, f"{path}.{str_key}")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Dictionary serialization failed at {path}: {e}")
            return {"_error": f"Dictionary serialization failed: {str(e)}"}
    
    def _serialize_sequence(self, data: Union[list, tuple], path: str) -> list:
        """Serialize list or tuple with element-wise conversion."""
        
        try:
            if len(data) > 10000:  # Large sequence limit
                logger.warning(f"⚠️ Large sequence at {path}: {len(data)} elements")
                if self.strict_mode:
                    raise ValueError(f"Sequence too large for serialization: {len(data)}")
                return [self.serialize_for_json(item, f"{path}[{i}]") for i, item in enumerate(data[:1000])]  # First 1000 elements
            
            return [self.serialize_for_json(item, f"{path}[{i}]") for i, item in enumerate(data)]
            
        except Exception as e:
            logger.warning(f"⚠️ Sequence serialization failed at {path}: {e}")
            return [f"_error: Sequence serialization failed: {str(e)}"]
    
    def _serialize_custom_object(self, obj: Any, path: str) -> Any:
        """Serialize custom objects by inspecting their __dict__."""
        
        try:
            # Avoid infinite recursion with common problematic objects
            obj_type = type(obj).__name__
            if obj_type in ['module', 'function', 'method', 'type', 'class']:
                return f"<{obj_type}: {str(obj)}>"
            
            # Try to serialize the object's __dict__
            obj_dict = {}
            for attr_name, attr_value in obj.__dict__.items():
                if not attr_name.startswith('_'):  # Skip private attributes
                    obj_dict[attr_name] = self.serialize_for_json(attr_value, f"{path}.{attr_name}")
            
            obj_dict['_type'] = obj_type
            self.conversion_log.append(f"Converted custom object {obj_type} at {path}")
            return obj_dict
            
        except Exception as e:
            logger.warning(f"⚠️ Custom object serialization failed at {path}: {e}")
            return {"_error": f"Custom object serialization failed: {str(e)}", "_type": type(obj).__name__}
    
    def _handle_unknown_type(self, data: Any, path: str) -> Any:
        """Handle unknown types with fallback strategies."""
        
        obj_type = type(data).__name__
        
        try:
            # Try converting to string
            str_repr = str(data)
            if len(str_repr) < 1000:  # Reasonable string length
                self.conversion_log.append(f"Converted unknown type {obj_type} to string at {path}")
                return f"<{obj_type}: {str_repr}>"
            else:
                return f"<{obj_type}: [large object]>"
                
        except Exception as e:
            logger.warning(f"⚠️ Unknown type {obj_type} at {path} couldn't be converted: {e}")
            if self.strict_mode:
                raise TypeError(f"Cannot serialize type {obj_type} at {path}")
            return f"<non-serializable: {obj_type}>"
    
    def _handle_serialization_error(self, data: Any, path: str, error: Exception) -> Any:
        """Handle serialization errors with graceful fallbacks."""
        
        obj_type = type(data).__name__
        error_msg = str(error)
        
        logger.error(f"❌ Serialization error at {path} for type {obj_type}: {error_msg}")
        
        if self.strict_mode:
            raise error
        
        # Return error information
        return {
            "_serialization_error": True,
            "_error_message": error_msg,
            "_type": obj_type,
            "_path": path
        }
    
    def clean_shap_explanation(self, shap_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Specifically clean SHAP explanation data for JSON serialization.
        
        SHAP explanations often contain NumPy arrays and complex nested structures
        that need special handling.
        """
        
        if not shap_data:
            return {}
        
        try:
            logger.info("🧹 Cleaning SHAP explanation data")
            cleaned_shap = {}
            
            # Handle common SHAP fields with specific logic
            field_handlers = {
                'shap_values': self._clean_shap_values,
                'feature_importance': self._clean_feature_importance,
                'expected_value': self._clean_expected_value,
                'feature_names': self._clean_feature_names,
                'human_readable_explanation': self._clean_human_explanations,
                'portfolio_quality_score': self._clean_numeric_score,
                'confidence_score': self._clean_numeric_score,
                'feature_contributions': self._clean_feature_contributions,
                'interaction_values': self._clean_interaction_values
            }
            
            for key, value in shap_data.items():
                if key in field_handlers:
                    cleaned_shap[key] = field_handlers[key](value, f"shap.{key}")
                else:
                    # Generic cleaning for other fields
                    cleaned_shap[key] = self.serialize_for_json(value, f"shap.{key}")
            
            logger.info("✅ SHAP explanation data cleaned successfully")
            return cleaned_shap
            
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning SHAP explanation: {e}")
            return {"error": f"SHAP data cleaning failed: {str(e)}", "original_keys": list(shap_data.keys())}
    
    def _clean_shap_values(self, shap_values: Any, path: str) -> Any:
        """Clean SHAP values (typically NumPy arrays)."""
        
        if shap_values is None:
            return None
        
        if isinstance(shap_values, np.ndarray):
            # Handle different SHAP value formats
            if shap_values.ndim == 1:
                return shap_values.tolist()
            elif shap_values.ndim == 2:
                return [row.tolist() for row in shap_values]
            else:
                logger.warning(f"⚠️ High-dimensional SHAP values at {path}: {shap_values.shape}")
                return {"shape": list(shap_values.shape), "sample": shap_values.flatten()[:100].tolist()}
        
        return self.serialize_for_json(shap_values, path)
    
    def _clean_feature_importance(self, importance: Any, path: str) -> Any:
        """Clean feature importance scores."""
        
        if importance is None:
            return None
        
        if isinstance(importance, dict):
            return {str(k): float(v) if isinstance(v, (int, float, np.number)) else self.serialize_for_json(v, f"{path}.{k}") 
                   for k, v in importance.items()}
        
        return self.serialize_for_json(importance, path)
    
    def _clean_expected_value(self, expected_value: Any, path: str) -> Optional[float]:
        """Clean expected value (baseline)."""
        
        if expected_value is None:
            return None
        
        try:
            return float(expected_value)
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Cannot convert expected value to float at {path}: {expected_value}")
            return None
    
    def _clean_feature_names(self, feature_names: Any, path: str) -> List[str]:
        """Clean feature names."""
        
        if feature_names is None:
            return []
        
        try:
            return [str(name) for name in feature_names]
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning feature names at {path}: {e}")
            return []
    
    def _clean_human_explanations(self, explanations: Any, path: str) -> Dict[str, str]:
        """Clean human-readable explanations."""
        
        if not explanations:
            return {}
        
        try:
            if isinstance(explanations, dict):
                return {str(k): str(v) for k, v in explanations.items() if v is not None}
            else:
                return {"explanation": str(explanations)}
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning human explanations at {path}: {e}")
            return {}
    
    def _clean_numeric_score(self, score: Any, path: str) -> Optional[float]:
        """Clean numeric scores."""
        
        if score is None:
            return None
        
        try:
            return float(score)
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Cannot convert score to float at {path}: {score}")
            return None
    
    def _clean_feature_contributions(self, contributions: Any, path: str) -> Any:
        """Clean feature contribution data."""
        
        if contributions is None:
            return None
        
        if isinstance(contributions, dict):
            cleaned = {}
            for feature, contrib in contributions.items():
                if isinstance(contrib, (list, np.ndarray)):
                    cleaned[str(feature)] = self.serialize_for_json(contrib, f"{path}.{feature}")
                else:
                    cleaned[str(feature)] = float(contrib) if isinstance(contrib, (int, float, np.number)) else str(contrib)
            return cleaned
        
        return self.serialize_for_json(contributions, path)
    
    def _clean_interaction_values(self, interactions: Any, path: str) -> Any:
        """Clean SHAP interaction values."""
        
        if interactions is None:
            return None
        
        if isinstance(interactions, np.ndarray):
            if interactions.size > 10000:  # Large interaction matrix
                return {
                    "shape": list(interactions.shape),
                    "sample": interactions.flatten()[:100].tolist(),
                    "_type": "large_interaction_matrix"
                }
            return interactions.tolist()
        
        return self.serialize_for_json(interactions, path)
    
    def validate_json_serialization(self, data: Any, description: str = "data") -> bool:
        """
        Test if data can be JSON serialized and provide detailed feedback.
        
        Args:
            data: Data to test
            description: Description for logging
            
        Returns:
            True if serializable, False otherwise
        """
        
        try:
            json.dumps(data)
            logger.debug(f"✅ {description} is JSON serializable")
            return True
            
        except Exception as e:
            logger.error(f"❌ {description} is NOT JSON serializable: {e}")
            
            # Provide detailed error analysis
            self._analyze_serialization_failure(data, description, e)
            return False
    
    def _analyze_serialization_failure(self, data: Any, description: str, error: Exception):
        """Analyze why JSON serialization failed."""
        
        error_type = type(error).__name__
        error_msg = str(error)
        
        logger.error(f"🔍 Serialization failure analysis for {description}:")
        logger.error(f"   Error type: {error_type}")
        logger.error(f"   Error message: {error_msg}")
        logger.error(f"   Data type: {type(data).__name__}")
        
        # Try to identify problematic elements
        if isinstance(data, dict):
            self._analyze_dict_serialization_issues(data)
        elif isinstance(data, (list, tuple)):
            self._analyze_sequence_serialization_issues(data)
        else:
            logger.error(f"   Data representation: {str(data)[:200]}...")
    
    def _analyze_dict_serialization_issues(self, data: dict):
        """Analyze serialization issues in dictionaries."""
        
        problematic_keys = []
        
        for key, value in data.items():
            try:
                json.dumps({key: value})
            except Exception as e:
                problematic_keys.append((key, type(value).__name__, str(e)[:100]))
        
        if problematic_keys:
            logger.error("   Problematic dictionary keys:")
            for key, value_type, error in problematic_keys[:5]:  # Show first 5
                logger.error(f"     '{key}' ({value_type}): {error}")
    
    def _analyze_sequence_serialization_issues(self, data: Union[list, tuple]):
        """Analyze serialization issues in sequences."""
        
        problematic_indices = []
        
        for i, item in enumerate(data[:100]):  # Check first 100 items
            try:
                json.dumps(item)
            except Exception as e:
                problematic_indices.append((i, type(item).__name__, str(e)[:100]))
        
        if problematic_indices:
            logger.error("   Problematic sequence indices:")
            for idx, item_type, error in problematic_indices[:5]:  # Show first 5
                logger.error(f"     [{idx}] ({item_type}): {error}")
    
    def get_conversion_summary(self) -> Dict[str, Any]:
        """
        Get summary of all conversions performed.
        
        Returns:
            Summary of serialization operations
        """
        
        return {
            "total_conversions": len(self.conversion_log),
            "conversion_log": self.conversion_log[-50:],  # Last 50 conversions
            "strict_mode": self.strict_mode
        }
    
    def reset_conversion_log(self):
        """Reset the conversion log."""
        self.conversion_log = []
        logger.info("🧹 Conversion log reset")
    
    def create_serialization_report(self, data: Any, description: str = "data") -> Dict[str, Any]:
        """
        Create a comprehensive serialization report for complex data.
        
        Args:
            data: Data to analyze
            description: Description of the data
            
        Returns:
            Detailed serialization report
        """
        
        report = {
            "description": description,
            "original_type": type(data).__name__,
            "timestamp": datetime.now().isoformat(),
            "serialization_successful": False,
            "cleaned_data": None,
            "issues_found": [],
            "recommendations": []
        }
        
        try:
            # Test original data
            if self.validate_json_serialization(data, description):
                report["serialization_successful"] = True
                report["cleaned_data"] = data
            else:
                # Try cleaning the data
                logger.info(f"🧹 Attempting to clean {description}")
                cleaned_data = self.serialize_for_json(data)
                
                if self.validate_json_serialization(cleaned_data, f"cleaned_{description}"):
                    report["serialization_successful"] = True
                    report["cleaned_data"] = cleaned_data
                    report["issues_found"].append("Original data required cleaning")
                    report["recommendations"].append("Use cleaned data for JSON serialization")
                else:
                    report["issues_found"].append("Data could not be cleaned for JSON serialization")
                    report["recommendations"].append("Consider using alternative serialization method")
            
            # Add conversion summary
            report["conversion_summary"] = self.get_conversion_summary()
            
        except Exception as e:
            report["issues_found"].append(f"Serialization report generation failed: {str(e)}")
            logger.error(f"❌ Error creating serialization report: {e}")
        
        return report
    
    def safe_serialize_with_fallback(self, data: Any, fallback_value: Any = None) -> Any:
        """
        Safely serialize data with a fallback value if serialization fails.
        
        Args:
            data: Data to serialize
            fallback_value: Value to return if serialization fails
            
        Returns:
            Serialized data or fallback value
        """
        
        try:
            cleaned_data = self.serialize_for_json(data)
            
            # Test that the cleaned data is actually serializable
            json.dumps(cleaned_data)
            
            return cleaned_data
            
        except Exception as e:
            logger.warning(f"⚠️ Safe serialization failed, using fallback: {e}")
            
            if fallback_value is None:
                fallback_value = {
                    "_serialization_failed": True,
                    "_error": str(e),
                    "_type": type(data).__name__,
                    "_timestamp": datetime.now().isoformat()
                }
            
            return fallback_value