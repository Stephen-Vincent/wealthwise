"""
Custom exception classes for the Portfolio Simulator Service.

This module defines specific exception types to provide better error handling
and more informative error messages throughout the application.
"""

from typing import Optional, Dict, Any


class PortfolioSimulatorError(Exception):
    """Base exception class for all portfolio simulator errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class ValidationError(PortfolioSimulatorError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, **kwargs):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Name of the field that failed validation
            value: The invalid value that caused the error
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value
        if field:
            self.details.update({"field": field, "value": str(value)})


class DataProviderError(PortfolioSimulatorError):
    """Raised when data provider operations fail."""
    
    def __init__(self, message: str, provider: Optional[str] = None, 
                 symbols: Optional[list] = None, **kwargs):
        """
        Initialize data provider error.
        
        Args:
            message: Error message
            provider: Name of the data provider (e.g., "yfinance")
            symbols: List of symbols that failed to download
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.provider = provider
        self.symbols = symbols or []
        if provider:
            self.details.update({"provider": provider})
        if symbols:
            self.details.update({"symbols": symbols})


class InsufficientDataError(DataProviderError):
    """Raised when insufficient market data is available for simulation."""
    
    def __init__(self, message: str, required_days: Optional[int] = None, 
                 available_days: Optional[int] = None, **kwargs):
        """
        Initialize insufficient data error.
        
        Args:
            message: Error message
            required_days: Number of days required for simulation
            available_days: Number of days actually available
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.required_days = required_days
        self.available_days = available_days
        if required_days is not None and available_days is not None:
            self.details.update({
                "required_days": required_days,
                "available_days": available_days
            })


class AIServiceError(PortfolioSimulatorError):
    """Raised when AI service operations fail."""
    
    def __init__(self, message: str, service: Optional[str] = None, 
                 model: Optional[str] = None, **kwargs):
        """
        Initialize AI service error.
        
        Args:
            message: Error message
            service: Name of the AI service (e.g., "groq", "wealthwise")
            model: Model name that failed
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.service = service
        self.model = model
        if service:
            self.details.update({"service": service})
        if model:
            self.details.update({"model": model})


class SHAPExplanationError(AIServiceError):
    """Raised when SHAP explanation generation fails."""
    pass


class VisualizationError(PortfolioSimulatorError):
    """Raised when visualization generation fails."""
    
    def __init__(self, message: str, chart_type: Optional[str] = None, 
                 file_path: Optional[str] = None, **kwargs):
        """
        Initialize visualization error.
        
        Args:
            message: Error message
            chart_type: Type of chart that failed to generate
            file_path: File path where chart was supposed to be saved
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.chart_type = chart_type
        self.file_path = file_path
        if chart_type:
            self.details.update({"chart_type": chart_type})
        if file_path:
            self.details.update({"file_path": file_path})


class SimulationError(PortfolioSimulatorError):
    """Raised when portfolio simulation fails."""
    
    def __init__(self, message: str, simulation_type: Optional[str] = None, 
                 timeframe: Optional[int] = None, **kwargs):
        """
        Initialize simulation error.
        
        Args:
            message: Error message
            simulation_type: Type of simulation that failed
            timeframe: Investment timeframe in years
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.simulation_type = simulation_type
        self.timeframe = timeframe
        if simulation_type:
            self.details.update({"simulation_type": simulation_type})
        if timeframe:
            self.details.update({"timeframe": timeframe})


class OptimizationError(PortfolioSimulatorError):
    """Raised when portfolio optimization fails."""
    
    def __init__(self, message: str, method: Optional[str] = None, 
                 constraints: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize optimization error.
        
        Args:
            message: Error message
            method: Optimization method that failed
            constraints: Optimization constraints that were applied
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.method = method
        self.constraints = constraints or {}
        if method:
            self.details.update({"method": method})
        if constraints:
            self.details.update({"constraints": constraints})


class DatabaseError(PortfolioSimulatorError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 table: Optional[str] = None, **kwargs):
        """
        Initialize database error.
        
        Args:
            message: Error message
            operation: Database operation that failed (e.g., "insert", "update")
            table: Database table involved in the operation
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.operation = operation
        self.table = table
        if operation:
            self.details.update({"operation": operation})
        if table:
            self.details.update({"table": table})


class SecurityError(PortfolioSimulatorError):
    """Raised when security violations are detected."""
    
    def __init__(self, message: str, violation_type: Optional[str] = None, 
                 attempted_value: Optional[str] = None, **kwargs):
        """
        Initialize security error.
        
        Args:
            message: Error message
            violation_type: Type of security violation
            attempted_value: The value that triggered the security violation
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message, **kwargs)
        self.violation_type = violation_type
        self.attempted_value = attempted_value
        if violation_type:
            self.details.update({"violation_type": violation_type})
        # Don't include attempted_value in details for security reasons