"""
Utilities Module for WealthWise Enhanced Stock Recommender

This module provides essential utility functions and configurations that support
the entire recommendation system. It includes data validation, logging setup,
and other common functionality used across all components.

Core Components:
1. DataValidator - Comprehensive data validation and stock filtering
2. Logging Configuration - Structured logging with performance monitoring
3. Utility Functions - Common helper functions for data processing

Key Features:
- Stock ticker validation and filtering
- Financial parameter validation with bounds checking
- Market data quality assessment
- Structured logging with performance tracking
- Error handling and fallback mechanisms
- Caching for improved performance
- Audit trail for compliance

Example Usage:
    from ai_models.stock_model.utils import DataValidator, setup_logging
    from ai_models.stock_model.utils import validate_tickers, get_logger
    
    # Set up logging
    setup_logging(log_level="INFO", log_to_file=True)
    logger = get_logger(__name__)
    
    # Validate stock tickers
    tickers = ["AAPL", "GOOGL", "INVALID123"]
    valid_tickers = validate_tickers(tickers)
    logger.info(f"Valid tickers: {valid_tickers}")
    
    # Validate financial parameters
    from ai_models.stock_model.utils import validate_financial_inputs
    
    validation = validate_financial_inputs(
        target_value=50000,
        current_investment=5000,
        timeframe=10,
        monthly_contribution=300,
        risk_score=65
    )
    
    if validation["is_valid"]:
        logger.info("All parameters valid")
        params = validation["cleaned_params"]
    else:
        logger.error(f"Validation errors: {validation['errors']}")

Data Validation Features:
- Ticker symbol pattern matching and API validation
- Financial parameter bounds checking and type conversion
- Market data quality assessment and cleaning
- Cross-validation between related parameters
- Comprehensive error reporting with specific guidance

Logging Features:
- Multiple output destinations (console, files, performance logs)
- Structured formatting with timestamps and context
- Performance monitoring with automatic timing
- Log rotation and retention management
- Audit logging for compliance and debugging
- Configurable log levels for different components

Quality Assurance:
- Input sanitization and bounds checking
- Graceful error handling with informative messages
- Fallback mechanisms when validation fails
- Caching to reduce redundant API calls
- Performance monitoring and optimization
"""

from .data_validation import (
    DataValidator,
    validate_tickers,
    validate_financial_inputs,
    clean_ticker_list
)

from .logging_config import (
    setup_logging,
    get_logger,
    log_performance,
    log_function_call,
    create_trading_session_logger,
    log_recommendation_audit,
    log_data_download,
    log_calculation_result,
    log_model_performance,
    StructuredFormatter,
    PerformanceFilter
)

from .model_manager import (
    ModelManager,
    initialize_models,
    get_model_manager,
    check_model_requirements
)

from .data_manager import (
    DataManager,
    initialize_data_system,
    get_data_manager
)

__all__ = [
    # Data Validation
    'DataValidator',
    'validate_tickers',
    'validate_financial_inputs', 
    'clean_ticker_list',
    
    # Logging
    'setup_logging',
    'get_logger',
    'log_performance',
    'log_function_call',
    'create_trading_session_logger',
    'log_recommendation_audit',
    'log_data_download',
    'log_calculation_result',
    'log_model_performance',
    'StructuredFormatter',
    'PerformanceFilter',
    
    # Model Management
    'ModelManager',
    'initialize_models',
    'get_model_manager', 
    'check_model_requirements',
    
    # Data Management
    'DataManager',
    'initialize_data_system',
    'get_data_manager'
]

# Version and metadata
__version__ = '1.0.0'
__author__ = 'WealthWise Team'
__description__ = 'Utility functions and configurations for the WealthWise recommendation system'

# Default configuration constants
DEFAULT_CONFIG = {
    'LOG_LEVEL': 'INFO',
    'LOG_TO_FILE': True,
    'LOG_TO_CONSOLE': True,
    'LOG_DIRECTORY': './logs',
    'ENABLE_PERFORMANCE_TRACKING': True,
    'CACHE_TIMEOUT': 3600,  # 1 hour
    'MAX_INVALID_TICKER_RATIO': 0.3,
    'MIN_PRICE_POINTS': 50,
    'MIN_VOLUME_THRESHOLD': 1000,
    'MAX_PRICE_CHANGE': 0.50
}

def initialize_complete_system(config: dict = None) -> dict:
    """
    Initialize the complete WealthWise system including logging, validation, 
    models, and data management
    
    This is the main system initialization function that sets up everything
    needed for the recommendation system to work properly.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Dict with complete initialization results
    """
    try:
        # Merge default config with user overrides
        system_config = DEFAULT_CONFIG.copy()
        if config:
            system_config.update(config)
        
        print("🚀 Initializing WealthWise Enhanced Stock Recommender System...")
        
        # 1. Initialize basic utilities (logging, validation)
        basic_init = _initialize_system(system_config)
        if not basic_init['success']:
            return {'success': False, 'error': 'Basic system initialization failed'}
        
        logger = get_logger('complete_system_init')
        logger.info("✅ Basic utilities initialized")
        
        # 2. Initialize data management system
        logger.info("📊 Initializing data management...")
        data_init = initialize_data_system()
        if not data_init['success']:
            logger.warning(f"⚠️ Data system initialization had issues: {data_init.get('error')}")
        else:
            logger.info("✅ Data management system ready")
        
        # 3. Initialize model management system
        logger.info("🤖 Initializing model management...")
        model_init = initialize_models(force_retrain=False)
        if not model_init['success']:
            logger.warning(f"⚠️ Model system initialization had issues: {model_init.get('error')}")
        else:
            logger.info("✅ Model management system ready")
        
        # 4. System health check
        logger.info("🏥 Performing system health check...")
        
        # Check model requirements
        model_requirements = check_model_requirements()
        
        # Get system status
        system_status = get_system_status()
        
        # Overall system health
        system_healthy = (
            basic_init['success'] and
            data_init['success'] and 
            model_init['success'] and
            model_requirements['all_requirements_met'] and
            system_status['system_healthy']
        )
        
        if system_healthy:
            logger.info("🎉 WealthWise system fully initialized and healthy!")
        else:
            logger.warning("⚠️ System initialized but some components may have issues")
        
        return {
            'success': True,
            'system_healthy': system_healthy,
            'basic_init': basic_init,
            'data_init': data_init,
            'model_init': model_init,
            'model_requirements': model_requirements,
            'system_status': system_status,
            'config': system_config,
            'recommendations': _get_system_recommendations(
                basic_init, data_init, model_init, model_requirements
            )
        }
        
    except Exception as e:
        print(f"❌ Complete system initialization failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'system_healthy': False
        }

def _get_system_recommendations(basic_init: dict, data_init: dict, 
                              model_init: dict, model_requirements: dict) -> list:
    """Get recommendations for improving system setup"""
    recommendations = []
    
    if not basic_init.get('success'):
        recommendations.append("Fix basic system initialization issues")
    
    if not data_init.get('success'):
        recommendations.append("Check data directory permissions and disk space")
    
    if not model_init.get('success'):
        recommendations.append("Install required ML packages: scikit-learn, joblib")
    
    if not model_requirements.get('all_requirements_met'):
        missing = model_requirements.get('missing_requirements', [])
        recommendations.extend(missing)
    
    # Check for optional packages
    if not model_requirements.get('package_status', {}).get('shap', False):
        recommendations.append("Install SHAP for explainable AI: pip install shap")
    
    if not recommendations:
        recommendations.append("System is fully optimized! All components working properly.")
    
    return recommendations

def _initialize_system(config: dict = None) -> dict:
    """
    Initialize the entire utility system with logging and validation

    This convenience function sets up logging, validation, and other
    system components with sensible defaults or custom configuration.

    Args:
        config: Optional configuration overrides

    Returns:
        Dict with initialization results and system status
    """
    try:
        # Merge default config with user overrides
        system_config = DEFAULT_CONFIG.copy()
        if config:
            system_config.update(config)

        # Set up logging first
        logging_result = setup_logging(
            log_level=system_config['LOG_LEVEL'],
            log_to_file=system_config['LOG_TO_FILE'],
            log_to_console=system_config['LOG_TO_CONSOLE'],
            log_directory=system_config['LOG_DIRECTORY'],
            enable_performance_tracking=system_config['ENABLE_PERFORMANCE_TRACKING']
        )

        # Get system logger
        logger = get_logger('system_initialization')

        if logging_result['success']:
            logger.info("🚀 WealthWise utility system initialized successfully")
            logger.info(f"📊 Configuration: {system_config}")
        else:
            logger.warning("⚠️ Logging setup had issues, using fallback configuration")

        # Initialize data validator with configuration
        validator = DataValidator()
        logger.info("✅ Data validation system ready")

        return {
            'success': True,
            'logging': logging_result,
            'config': system_config,
            'validator': validator,
            'system_ready': True
        }

    except Exception as e:
        # Fallback initialization
        print(f"System initialization failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'system_ready': False
        }
    """
    Initialize the entire utility system with logging and validation
    
    This convenience function sets up logging, validation, and other
    system components with sensible defaults or custom configuration.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Dict with initialization results and system status
    """
    try:
        # Merge default config with user overrides
        system_config = DEFAULT_CONFIG.copy()
        if config:
            system_config.update(config)
        
        # Set up logging first
        logging_result = setup_logging(
            log_level=system_config['LOG_LEVEL'],
            log_to_file=system_config['LOG_TO_FILE'],
            log_to_console=system_config['LOG_TO_CONSOLE'],
            log_directory=system_config['LOG_DIRECTORY'],
            enable_performance_tracking=system_config['ENABLE_PERFORMANCE_TRACKING']
        )
        
        # Get system logger
        logger = get_logger('system_initialization')
        
        if logging_result['success']:
            logger.info("🚀 WealthWise utility system initialized successfully")
            logger.info(f"📊 Configuration: {system_config}")
        else:
            logger.warning("⚠️ Logging setup had issues, using fallback configuration")
        
        # Initialize data validator with configuration
        validator = DataValidator()
        logger.info("✅ Data validation system ready")
        
        return {
            'success': True,
            'logging': logging_result,
            'config': system_config,
            'validator': validator,
            'system_ready': True
        }
        
    except Exception as e:
        # Fallback initialization
        print(f"System initialization failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'system_ready': False
        }

def get_system_status() -> dict:
    """
    Get current system status and health check
    
    Returns:
        Dict with system health information
    """
    try:
        import logging
        
        # Check logging system
        root_logger = logging.getLogger()
        handlers_count = len(root_logger.handlers)
        
        # Check if we can create a validator
        try:
            validator = DataValidator()
            validation_ready = True
        except:
            validation_ready = False
        
        # Memory usage (basic check)
        import sys
        memory_usage = sys.getsizeof(DataValidator()) if validation_ready else 0
        
        status = {
            'timestamp': str(pd.Timestamp.now()),
            'logging_handlers': handlers_count,
            'validation_system': validation_ready,
            'memory_usage_bytes': memory_usage,
            'system_healthy': handlers_count > 0 and validation_ready
        }
        
        return status
        
    except Exception as e:
        return {
            'timestamp': str(pd.Timestamp.now()),
            'error': str(e),
            'system_healthy': False
        }

def validate_system_requirements() -> dict:
    """
    Validate that all required dependencies and configurations are available
    
    Returns:
        Dict with requirement validation results
    """
    requirements = {
        'required_packages': ['yfinance', 'pandas', 'numpy', 'scikit-learn'],
        'optional_packages': ['shap', 'plotly', 'matplotlib', 'seaborn'],
        'system_features': []
    }
    
    results = {
        'required_packages': {},
        'optional_packages': {},
        'system_features': {},
        'all_required_available': True,
        'recommendations': []
    }
    
    # Check required packages
    for package in requirements['required_packages']:
        try:
            __import__(package)
            results['required_packages'][package] = True
        except ImportError:
            results['required_packages'][package] = False
            results['all_required_available'] = False
            results['recommendations'].append(f"Install required package: pip install {package}")
    
    # Check optional packages
    for package in requirements['optional_packages']:
        try:
            __import__(package)
            results['optional_packages'][package] = True
        except ImportError:
            results['optional_packages'][package] = False
            results['recommendations'].append(f"Consider installing optional package: pip install {package}")
    
    # System features
    import os
    results['system_features']['file_write_access'] = os.access('.', os.W_OK)
    results['system_features']['logs_directory_writable'] = True
    
    try:
        os.makedirs('./logs', exist_ok=True)
        results['system_features']['logs_directory_writable'] = True
    except:
        results['system_features']['logs_directory_writable'] = False
        results['recommendations'].append("Ensure write access to logs directory")
    
    return results

def create_sample_validation_data() -> dict:
    """
    Create sample data for testing validation functions
    
    Returns:
        Dict with sample data for testing
    """
    return {
        'valid_tickers': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'VTI', 'BND'],
        'invalid_tickers': ['INVALID123', '', 'TOOLONG123456', '123ABC'],
        'financial_params': {
            'valid': {
                'target_value': 50000,
                'current_investment': 5000,
                'timeframe': 10,
                'monthly_contribution': 300,
                'risk_score': 65
            },
            'invalid': {
                'target_value': -1000,  # Negative
                'current_investment': 'not_a_number',  # Wrong type
                'timeframe': 0,  # Zero timeframe
                'monthly_contribution': -500,  # Negative contribution
                'risk_score': 150  # Out of range
            }
        }
    }

# Import pandas for timestamp functionality
import pandas as pd