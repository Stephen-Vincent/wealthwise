"""
Logging Configuration Module

This module provides comprehensive logging configuration for the WealthWise
Enhanced Stock Recommender system. It sets up structured logging with
appropriate levels, formatting, and output destinations.

Key Features:
1. Multiple log levels and handlers
2. Structured log formatting with timestamps
3. File and console output options
4. Performance monitoring and metrics
5. Error tracking and debugging support
6. Configurable log rotation and retention
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import json


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that creates structured, readable log messages
    with consistent formatting across the entire system.
    """
    
    def __init__(self, include_extra_fields: bool = True):
        """
        Initialize the structured formatter
        
        Args:
            include_extra_fields: Whether to include extra fields in log records
        """
        self.include_extra_fields = include_extra_fields
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with structured information
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message string
        """
        # Basic log components
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        level = record.levelname
        logger_name = record.name
        message = record.getMessage()
        
        # Create base log entry
        log_parts = [
            f"[{timestamp}]",
            f"[{level:8}]",
            f"[{logger_name}]",
            message
        ]
        
        # Add exception information if present
        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            log_parts.append(f"\n{exception_text}")
        
        # Add extra fields if requested and present
        if self.include_extra_fields and hasattr(record, '__dict__'):
            extra_fields = {}
            
            # Standard fields to exclude
            standard_fields = {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                'filename', 'module', 'lineno', 'funcName', 'created', 
                'msecs', 'relativeCreated', 'thread', 'threadName', 
                'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info'
            }
            
            # Collect extra fields
            for key, value in record.__dict__.items():
                if key not in standard_fields and not key.startswith('_'):
                    extra_fields[key] = value
            
            # Add extra fields to log if any exist
            if extra_fields:
                extra_str = " | ".join([f"{k}={v}" for k, v in extra_fields.items()])
                log_parts.append(f" | {extra_str}")
        
        return " ".join(log_parts)


class PerformanceFilter(logging.Filter):
    """
    Custom filter to track performance metrics and add timing information
    to log records for performance monitoring.
    """
    
    def __init__(self):
        """Initialize the performance filter"""
        super().__init__()
        self.start_times = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records and add performance timing information
        
        Args:
            record: Log record to filter
            
        Returns:
            True to allow the record through (we don't actually filter anything)
        """
        # Add timestamp for performance tracking
        record.timestamp = datetime.now()
        
        # Track function execution times if message contains timing markers
        message = record.getMessage()
        
        if "🚀" in message or "Starting" in message:
            # Mark start of operation
            operation_key = f"{record.name}_{record.funcName}"
            self.start_times[operation_key] = record.timestamp
            record.operation_start = True
        
        elif "✅" in message or "Complete" in message or "Success" in message:
            # Mark end of operation and calculate duration
            operation_key = f"{record.name}_{record.funcName}"
            if operation_key in self.start_times:
                start_time = self.start_times.pop(operation_key)
                duration = (record.timestamp - start_time).total_seconds()
                record.operation_duration = duration
                record.operation_end = True
                
                # Add duration to the message
                if duration > 0.1:  # Only show duration for operations > 100ms
                    record.msg = f"{record.msg} (took {duration:.2f}s)"
        
        return True


def setup_logging(log_level: str = "INFO", 
                 log_to_file: bool = True,
                 log_to_console: bool = True,
                 log_directory: str = "./logs",
                 enable_performance_tracking: bool = True) -> Dict[str, Any]:
    """
    Set up comprehensive logging configuration for the entire application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to files
        log_to_console: Whether to log to console
        log_directory: Directory for log files
        enable_performance_tracking: Whether to enable performance monitoring
        
    Returns:
        Dict with logging configuration details
    """
    try:
        # Convert string level to logging level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create log directory if it doesn't exist
        if log_to_file:
            os.makedirs(log_directory, exist_ok=True)
        
        # Get root logger and configure
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        handlers_created = []
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            
            # Use colored output for console if available
            console_formatter = StructuredFormatter(include_extra_fields=False)
            console_handler.setFormatter(console_formatter)
            
            # Add performance filter if enabled
            if enable_performance_tracking:
                console_handler.addFilter(PerformanceFilter())
            
            root_logger.addHandler(console_handler)
            handlers_created.append("console")
        
        # File handlers
        if log_to_file:
            # Main application log (rotating file)
            main_log_file = os.path.join(log_directory, "wealthwise_recommender.log")
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file, 
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(numeric_level)
            
            file_formatter = StructuredFormatter(include_extra_fields=True)
            file_handler.setFormatter(file_formatter)
            
            if enable_performance_tracking:
                file_handler.addFilter(PerformanceFilter())
            
            root_logger.addHandler(file_handler)
            handlers_created.append("main_file")
            
            # Error-only log file
            error_log_file = os.path.join(log_directory, "errors.log")
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=5*1024*1024,   # 5MB
                backupCount=3
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            
            root_logger.addHandler(error_handler)
            handlers_created.append("error_file")
            
            # Performance log (if enabled)
            if enable_performance_tracking:
                perf_log_file = os.path.join(log_directory, "performance.log")
                perf_handler = logging.handlers.RotatingFileHandler(
                    perf_log_file,
                    maxBytes=5*1024*1024,
                    backupCount=2
                )
                perf_handler.setLevel(logging.INFO)
                
                # Custom filter for performance logs
                class PerfOnlyFilter(logging.Filter):
                    def filter(self, record):
                        return hasattr(record, 'operation_duration') or "performance" in record.getMessage().lower()
                
                perf_handler.addFilter(PerfOnlyFilter())
                perf_handler.setFormatter(file_formatter)
                
                root_logger.addHandler(perf_handler)
                handlers_created.append("performance_file")
        
        # Configure specific logger levels for different modules
        module_log_levels = {
            "yfinance": logging.WARNING,           # Reduce yfinance noise
            "urllib3": logging.WARNING,            # Reduce HTTP request logs
            "matplotlib": logging.WARNING,         # Reduce matplotlib logs
            "PIL": logging.WARNING,                # Reduce image library logs
            "requests": logging.WARNING,           # Reduce requests logs
        }
        
        for module_name, level in module_log_levels.items():
            logging.getLogger(module_name).setLevel(level)
        
        # Create specific loggers for different components
        component_loggers = [
            "ai_models.stock_model.core",
            "ai_models.stock_model.analysis", 
            "ai_models.stock_model.explainable_ai",
            "ai_models.stock_model.goal_optimization",
            "ai_models.stock_model.utils"
        ]
        
        for logger_name in component_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(numeric_level)
        
        # Log successful setup
        setup_logger = logging.getLogger("logging_setup")
        setup_logger.info(f"✅ Logging configured successfully")
        setup_logger.info(f"📊 Log level: {log_level}")
        setup_logger.info(f"📁 Handlers: {', '.join(handlers_created)}")
        if log_to_file:
            setup_logger.info(f"📂 Log directory: {log_directory}")
        
        return {
            "success": True,
            "log_level": log_level,
            "handlers": handlers_created,
            "log_directory": log_directory if log_to_file else None,
            "performance_tracking": enable_performance_tracking
        }
        
    except Exception as e:
        # Fallback logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger("logging_setup")
        logger.error(f"Failed to set up advanced logging: {e}")
        logger.info("Using basic logging configuration as fallback")
        
        return {
            "success": False,
            "error": str(e),
            "fallback": True
        }


def get_logger(name: str, 
               extra_context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a configured logger with optional extra context
    
    Args:
        name: Logger name (usually __name__)
        extra_context: Additional context to include in all log messages
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Add extra context if provided
    if extra_context:
        # Create a custom adapter that adds extra fields
        class ContextAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                # Add extra context to every log message
                if 'extra' not in kwargs:
                    kwargs['extra'] = {}
                kwargs['extra'].update(self.extra)
                return msg, kwargs
        
        logger = ContextAdapter(logger, extra_context)
    
    return logger


def log_performance(operation_name: str):
    """
    Decorator to automatically log performance of functions
    
    Args:
        operation_name: Name of the operation for logging
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            start_time = datetime.now()
            logger.info(f"🚀 Starting {operation_name}")
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                logger.info(f"✅ {operation_name} completed successfully", 
                           extra={"operation_duration": duration})
                
                return result
                
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                logger.error(f"❌ {operation_name} failed: {str(e)}", 
                           extra={"operation_duration": duration, "error": str(e)})
                raise
        
        return wrapper
    return decorator


def log_function_call(include_args: bool = False, 
                     include_result: bool = False):
    """
    Decorator to log function calls with optional arguments and results
    
    Args:
        include_args: Whether to include function arguments in logs
        include_result: Whether to include function result in logs
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            # Log function entry
            log_msg = f"Calling {func.__name__}"
            extra_info = {}
            
            if include_args:
                # Be careful with sensitive information
                safe_args = []
                for arg in args:
                    if isinstance(arg, (str, int, float, bool)):
                        safe_args.append(str(arg))
                    else:
                        safe_args.append(f"<{type(arg).__name__}>")
                
                safe_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, (str, int, float, bool)):
                        safe_kwargs[k] = v
                    else:
                        safe_kwargs[k] = f"<{type(v).__name__}>"
                
                extra_info["args"] = safe_args
                extra_info["kwargs"] = safe_kwargs
            
            logger.debug(log_msg, extra=extra_info)
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    if isinstance(result, (str, int, float, bool)):
                        logger.debug(f"{func.__name__} returned: {result}")
                    else:
                        logger.debug(f"{func.__name__} returned: <{type(result).__name__}>")
                
                return result
                
            except Exception as e:
                logger.error(f"{func.__name__} raised exception: {str(e)}")
                raise
        
        return wrapper
    return decorator


def create_trading_session_logger(session_id: str) -> logging.Logger:
    """
    Create a logger for a specific trading/analysis session
    
    Args:
        session_id: Unique identifier for the session
        
    Returns:
        Logger configured for the session
    """
    logger_name = f"trading_session.{session_id}"
    logger = logging.getLogger(logger_name)
    
    # Add session context to all messages
    extra_context = {
        "session_id": session_id,
        "session_start": datetime.now().isoformat()
    }
    
    return get_logger(logger_name, extra_context)


def log_recommendation_audit(user_id: str, recommendation_data: Dict[str, Any]):
    """
    Log recommendation for audit trail and compliance
    
    Args:
        user_id: User identifier
        recommendation_data: Complete recommendation data for audit
    """
    audit_logger = logging.getLogger("recommendation_audit")
    
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "recommendation_type": recommendation_data.get("type", "unknown"),
        "stocks_recommended": recommendation_data.get("stocks", []),
        "risk_score": recommendation_data.get("risk_score"),
        "target_value": recommendation_data.get("target_value"),
        "feasibility_score": recommendation_data.get("feasibility_score"),
        "methodology": recommendation_data.get("methodology", "unknown")
    }
    
    audit_logger.info("Investment recommendation generated", 
                     extra={"audit_data": audit_entry})


def log_data_download(ticker: str, period: str, success: bool, 
                     data_points: Optional[int] = None):
    """Log data download operations"""
    logger = logging.getLogger("data_download")
    
    if success:
        logger.info(f"📊 Downloaded {ticker} data for {period}", 
                   extra={"ticker": ticker, "period": period, "data_points": data_points})
    else:
        logger.warning(f"❌ Failed to download {ticker} data for {period}",
                      extra={"ticker": ticker, "period": period})


def log_calculation_result(calculation_type: str, inputs: Dict[str, Any], 
                          result: Any, duration: Optional[float] = None):
    """Log calculation results for debugging and verification"""
    logger = logging.getLogger("calculations")
    
    extra_info = {
        "calculation_type": calculation_type,
        "inputs": inputs,
        "result_type": type(result).__name__
    }
    
    if duration:
        extra_info["duration"] = duration
    
    logger.info(f"🧮 {calculation_type} calculation completed", extra=extra_info)


def log_model_performance(model_name: str, metrics: Dict[str, float]):
    """Log ML model performance metrics"""
    logger = logging.getLogger("model_performance")
    
    logger.info(f"🤖 {model_name} performance metrics", 
               extra={"model": model_name, "metrics": metrics})


def create_json_logger(log_file: str) -> logging.Logger:
    """
    Create a logger that outputs structured JSON logs
    
    Useful for integration with log analysis tools and monitoring systems.
    
    Args:
        log_file: Path to JSON log file
        
    Returns:
        Logger configured for JSON output
    """
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'lineno', 
                              'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process']:
                    log_entry[key] = value
            
            return json.dumps(log_entry)
    
    # Create handler and logger
    handler = logging.FileHandler(log_file)
    handler.setFormatter(JSONFormatter())
    
    logger = logging.getLogger(f"json_logger_{log_file}")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger


def setup_debug_logging(debug_modules: List[str] = None) -> None:
    """
    Set up detailed debug logging for specific modules
    
    Args:
        debug_modules: List of module names to enable debug logging for
    """
    if debug_modules is None:
        debug_modules = [
            "ai_models.stock_model.analysis.factor_analysis",
            "ai_models.stock_model.explainable_ai.shap_explainer",
            "ai_models.stock_model.goal_optimization.goal_calculator"
        ]
    
    for module_name in debug_modules:
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.DEBUG)
        
        # Add debug handler if not already present
        if not Any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            debug_handler = logging.StreamHandler(sys.stdout)
            debug_handler.setLevel(logging.DEBUG)
            debug_formatter = StructuredFormatter(include_extra_fields=True)
            debug_handler.setFormatter(debug_formatter)
            logger.addHandler(debug_handler)
    
    debug_logger = logging.getLogger("debug_setup")
    debug_logger.info(f"🔍 Debug logging enabled for {len(debug_modules)} modules")


def log_system_metrics():
    """Log system performance and resource usage metrics"""
    try:
        import psutil
        import gc
        
        # Memory usage
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        metrics = {
            "system_memory_percent": memory.percent,
            "process_memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent(),
            "gc_collections": sum(gc.get_count()),
            "timestamp": datetime.now().isoformat()
        }
        
        metrics_logger = logging.getLogger("system_metrics")
        metrics_logger.info("📊 System metrics", extra={"metrics": metrics})
        
    except ImportError:
        # psutil not available, log basic info
        import sys
        
        basic_metrics = {
            "python_version": sys.version,
            "platform": sys.platform,
            "timestamp": datetime.now().isoformat()
        }
        
        metrics_logger = logging.getLogger("system_metrics")
        metrics_logger.info("📊 Basic system info", extra={"metrics": basic_metrics})


def configure_third_party_logging():
    """Configure logging levels for third-party libraries to reduce noise"""
    third_party_configs = {
        # Financial data libraries
        "yfinance": logging.WARNING,
        "pandas_datareader": logging.WARNING,
        
        # HTTP and networking
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "httpx": logging.WARNING,
        
        # Data science libraries
        "matplotlib": logging.WARNING,
        "seaborn": logging.WARNING,
        "plotly": logging.WARNING,
        "PIL": logging.WARNING,
        
        # Machine learning libraries
        "sklearn": logging.WARNING,
        "tensorflow": logging.ERROR,
        "torch": logging.WARNING,
        
        # Other common libraries
        "asyncio": logging.WARNING,
        "concurrent": logging.WARNING
    }
    
    configured_count = 0
    for library, level in third_party_configs.items():
        try:
            logger = logging.getLogger(library)
            logger.setLevel(level)
            configured_count += 1
        except:
            continue
    
    config_logger = logging.getLogger("third_party_config")
    config_logger.debug(f"🔧 Configured logging for {configured_count} third-party libraries")


# Initialize third-party logging configuration on import
configure_third_party_logging()