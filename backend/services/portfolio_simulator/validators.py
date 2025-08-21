"""
Input validation and sanitization for the Portfolio Simulator Service.

This module provides comprehensive validation for all user inputs to ensure
data integrity, security, and prevent common attack vectors.
"""

import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal, InvalidOperation

from .config import get_config
from .exceptions import ValidationError, SecurityError


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self):
        """Initialize validator with configuration."""
        self.config = get_config()
        
        # Regex patterns for validation
        self.safe_string_pattern = re.compile(r'^[a-zA-Z0-9\s\-_.,()]+$')
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        self.alphanumeric_pattern = re.compile(r'^[a-zA-Z0-9]+$')
        
        # Path traversal detection
        self.path_traversal_patterns = [
            '../', '..\\', '..%2f', '..%5c', 
            '..%252f', '..%255c', '%2e%2e%2f', '%2e%2e%5c'
        ]
    
    def validate_simulation_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete simulation input data.
        
        Args:
            data: Raw input data from user
            
        Returns:
            Dict containing validated and sanitized data
            
        Raises:
            ValidationError: If any validation fails
        """
        validated = {}
        
        # Required fields validation
        validated['goal'] = self.validate_goal(data.get('goal'))
        validated['target_value'] = self.validate_target_value(data.get('target_value'))
        validated['timeframe'] = self.validate_timeframe(data.get('timeframe'))
        validated['risk_score'] = self.validate_risk_score(data.get('risk_score'))
        validated['risk_label'] = self.validate_risk_label(data.get('risk_label'))
        
        # Optional fields with defaults
        validated['lump_sum'] = self.validate_investment_amount(
            data.get('lump_sum', 0), allow_zero=True
        )
        validated['monthly'] = self.validate_investment_amount(
            data.get('monthly', 0), allow_zero=True
        )
        validated['years_of_experience'] = self.validate_experience_years(
            data.get('years_of_experience', 0)
        )
        validated['income_bracket'] = self.validate_income_bracket(
            data.get('income_bracket', 'medium')
        )
        validated['user_id'] = self.validate_user_id(data.get('user_id'))
        
        # Business logic validation
        self._validate_investment_logic(validated)
        
        return validated
    
    def validate_goal(self, goal: Any) -> str:
        """
        Validate investment goal.
        
        Args:
            goal: User's investment goal
            
        Returns:
            Validated goal string
            
        Raises:
            ValidationError: If goal is invalid
        """
        if not goal:
            raise ValidationError("Investment goal is required", field="goal")
        
        goal_str = str(goal).strip()
        
        if len(goal_str) < 3:
            raise ValidationError(
                "Goal must be at least 3 characters long", 
                field="goal", value=goal_str
            )
        
        if len(goal_str) > 100:
            raise ValidationError(
                "Goal must be less than 100 characters", 
                field="goal", value=goal_str
            )
        
        # Sanitize goal string
        sanitized_goal = self.sanitize_string(goal_str)
        
        if not sanitized_goal:
            raise ValidationError(
                "Goal contains invalid characters", 
                field="goal", value=goal_str
            )
        
        return sanitized_goal
    
    def validate_target_value(self, target_value: Any) -> float:
        """
        Validate target investment value.
        
        Args:
            target_value: Target value to achieve
            
        Returns:
            Validated target value as float
            
        Raises:
            ValidationError: If target value is invalid
        """
        if target_value is None:
            raise ValidationError(
                "Target value is required", field="target_value"
            )
        
        try:
            value = float(target_value)
        except (ValueError, TypeError):
            raise ValidationError(
                "Target value must be a valid number", 
                field="target_value", value=target_value
            )
        
        if math.isnan(value) or math.isinf(value):
            raise ValidationError(
                "Target value must be a finite number", 
                field="target_value", value=target_value
            )
        
        min_value = self.config.simulation.min_investment_amount
        max_value = self.config.simulation.max_investment_amount
        
        if value < min_value:
            raise ValidationError(
                f"Target value must be at least £{min_value:,.2f}", 
                field="target_value", value=value
            )
        
        if value > max_value:
            raise ValidationError(
                f"Target value cannot exceed £{max_value:,.2f}", 
                field="target_value", value=value
            )
        
        return round(value, 2)
    
    def validate_timeframe(self, timeframe: Any) -> int:
        """
        Validate investment timeframe in years.
        
        Args:
            timeframe: Investment timeframe
            
        Returns:
            Validated timeframe as integer
            
        Raises:
            ValidationError: If timeframe is invalid
        """
        if timeframe is None:
            raise ValidationError("Timeframe is required", field="timeframe")
        
        try:
            years = int(timeframe)
        except (ValueError, TypeError):
            raise ValidationError(
                "Timeframe must be a valid integer", 
                field="timeframe", value=timeframe
            )
        
        min_years = self.config.simulation.min_timeframe_years
        max_years = self.config.simulation.max_timeframe_years
        
        if years < min_years:
            raise ValidationError(
                f"Timeframe must be at least {min_years} year(s)", 
                field="timeframe", value=years
            )
        
        if years > max_years:
            raise ValidationError(
                f"Timeframe cannot exceed {max_years} years", 
                field="timeframe", value=years
            )
        
        return years
    
    def validate_risk_score(self, risk_score: Any) -> int:
        """
        Validate risk score (0-100).
        
        Args:
            risk_score: User's risk tolerance score
            
        Returns:
            Validated risk score as integer
            
        Raises:
            ValidationError: If risk score is invalid
        """
        if risk_score is None:
            raise ValidationError("Risk score is required", field="risk_score")
        
        try:
            score = int(risk_score)
        except (ValueError, TypeError):
            raise ValidationError(
                "Risk score must be a valid integer", 
                field="risk_score", value=risk_score
            )
        
        if score < 0 or score > 100:
            raise ValidationError(
                "Risk score must be between 0 and 100", 
                field="risk_score", value=score
            )
        
        return score
    
    def validate_risk_label(self, risk_label: Any) -> str:
        """
        Validate risk label.
        
        Args:
            risk_label: Human-readable risk label
            
        Returns:
            Validated risk label
            
        Raises:
            ValidationError: If risk label is invalid
        """
        valid_labels = ['Low', 'Medium', 'High', 'Conservative', 'Moderate', 'Aggressive']
        
        if not risk_label:
            raise ValidationError("Risk label is required", field="risk_label")
        
        label_str = str(risk_label).strip().title()
        
        if label_str not in valid_labels:
            raise ValidationError(
                f"Risk label must be one of: {', '.join(valid_labels)}", 
                field="risk_label", value=risk_label
            )
        
        return label_str
    
    def validate_investment_amount(self, amount: Any, allow_zero: bool = False) -> float:
        """
        Validate investment amount (lump sum or monthly).
        
        Args:
            amount: Investment amount to validate
            allow_zero: Whether zero values are allowed
            
        Returns:
            Validated amount as float
            
        Raises:
            ValidationError: If amount is invalid
        """
        if amount is None:
            amount = 0
        
        try:
            value = float(amount)
        except (ValueError, TypeError):
            raise ValidationError(
                "Investment amount must be a valid number", 
                field="investment_amount", value=amount
            )
        
        if math.isnan(value) or math.isinf(value):
            raise ValidationError(
                "Investment amount must be a finite number", 
                field="investment_amount", value=amount
            )
        
        if value < 0:
            raise ValidationError(
                "Investment amount cannot be negative", 
                field="investment_amount", value=value
            )
        
        if not allow_zero and value == 0:
            raise ValidationError(
                "Investment amount must be greater than zero", 
                field="investment_amount", value=value
            )
        
        if value > self.config.simulation.max_investment_amount:
            raise ValidationError(
                f"Investment amount cannot exceed £{self.config.simulation.max_investment_amount:,.2f}", 
                field="investment_amount", value=value
            )
        
        return round(value, 2)
    
    def validate_experience_years(self, years: Any) -> int:
        """
        Validate years of investment experience.
        
        Args:
            years: Years of experience
            
        Returns:
            Validated years as integer
            
        Raises:
            ValidationError: If years is invalid
        """
        if years is None:
            return 0
        
        try:
            exp_years = int(years)
        except (ValueError, TypeError):
            raise ValidationError(
                "Experience years must be a valid integer", 
                field="years_of_experience", value=years
            )
        
        if exp_years < 0:
            raise ValidationError(
                "Experience years cannot be negative", 
                field="years_of_experience", value=exp_years
            )
        
        if exp_years > 70:  # Reasonable upper limit
            raise ValidationError(
                "Experience years cannot exceed 70", 
                field="years_of_experience", value=exp_years
            )
        
        return exp_years
    
    def validate_income_bracket(self, bracket: Any) -> str:
        """
        Validate income bracket.
        
        Args:
            bracket: Income bracket string
            
        Returns:
            Validated income bracket
            
        Raises:
            ValidationError: If bracket is invalid
        """
        valid_brackets = ['low', 'medium', 'high', 'very_high']
        
        if not bracket:
            return 'medium'  # Default
        
        bracket_str = str(bracket).lower().strip()
        
        if bracket_str not in valid_brackets:
            raise ValidationError(
                f"Income bracket must be one of: {', '.join(valid_brackets)}", 
                field="income_bracket", value=bracket
            )
        
        return bracket_str
    
    def validate_user_id(self, user_id: Any) -> Optional[int]:
        """
        Validate user ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            Validated user ID or None
            
        Raises:
            ValidationError: If user ID is invalid
        """
        if user_id is None:
            return None
        
        try:
            uid = int(user_id)
        except (ValueError, TypeError):
            raise ValidationError(
                "User ID must be a valid integer", 
                field="user_id", value=user_id
            )
        
        if uid <= 0:
            raise ValidationError(
                "User ID must be positive", 
                field="user_id", value=uid
            )
        
        return uid
    
    def validate_ticker_symbols(self, symbols: List[str]) -> List[str]:
        """
        Validate stock ticker symbols.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            List of validated ticker symbols
            
        Raises:
            ValidationError: If any symbol is invalid
        """
        if not symbols:
            raise ValidationError("At least one ticker symbol is required")
        
        if len(symbols) > 20:  # Reasonable limit
            raise ValidationError("Cannot have more than 20 ticker symbols")
        
        validated_symbols = []
        
        for symbol in symbols:
            if not isinstance(symbol, str):
                raise ValidationError(
                    f"Ticker symbol must be a string: {symbol}", 
                    field="ticker_symbols", value=symbol
                )
            
            symbol = symbol.strip().upper()
            
            if not symbol:
                continue  # Skip empty symbols
            
            if not self.alphanumeric_pattern.match(symbol.replace('.', '')):
                raise ValidationError(
                    f"Invalid ticker symbol format: {symbol}", 
                    field="ticker_symbols", value=symbol
                )
            
            if len(symbol) > 10:  # Most symbols are 1-5 characters
                raise ValidationError(
                    f"Ticker symbol too long: {symbol}", 
                    field="ticker_symbols", value=symbol
                )
            
            validated_symbols.append(symbol)
        
        return list(set(validated_symbols))  # Remove duplicates
    
    def validate_weights(self, weights: List[float]) -> List[float]:
        """
        Validate portfolio weights.
        
        Args:
            weights: List of portfolio weights
            
        Returns:
            Validated and normalized weights
            
        Raises:
            ValidationError: If weights are invalid
        """
        if not weights:
            raise ValidationError("Weights are required")
        
        validated_weights = []
        
        for i, weight in enumerate(weights):
            try:
                w = float(weight)
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Weight at position {i} must be a number: {weight}", 
                    field="weights", value=weight
                )
            
            if math.isnan(w) or math.isinf(w):
                raise ValidationError(
                    f"Weight at position {i} must be finite: {weight}", 
                    field="weights", value=weight
                )
            
            if w < 0:
                raise ValidationError(
                    f"Weight at position {i} cannot be negative: {weight}", 
                    field="weights", value=weight
                )
            
            validated_weights.append(w)
        
        # Normalize weights to sum to 1
        total_weight = sum(validated_weights)
        
        if total_weight == 0:
            raise ValidationError("All weights cannot be zero")
        
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            # Normalize weights
            validated_weights = [w / total_weight for w in validated_weights]
        
        return validated_weights
    
    def sanitize_string(self, text: str, max_length: int = 200) -> str:
        """
        Sanitize string input to prevent injection attacks.
        
        Args:
            text: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If dangerous patterns are detected
        """
        if not text:
            return ""
        
        text = str(text).strip()
        
        # Check for path traversal attempts
        text_lower = text.lower()
        for pattern in self.path_traversal_patterns:
            if pattern in text_lower:
                raise SecurityError(
                    "Path traversal attempt detected", 
                    violation_type="path_traversal"
                )
        
        # Check for SQL injection patterns
        sql_patterns = ['union', 'select', 'insert', 'update', 'delete', 'drop', '--', ';']
        for pattern in sql_patterns:
            if pattern in text_lower:
                raise SecurityError(
                    "Potential SQL injection detected", 
                    violation_type="sql_injection"
                )
        
        # Remove potentially dangerous characters
        if not self.safe_string_pattern.match(text):
            # Keep only safe characters
            safe_chars = re.findall(r'[a-zA-Z0-9\s\-_.,()]', text)
            text = ''.join(safe_chars)
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and other attacks.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Safe filename
            
        Raises:
            SecurityError: If filename contains dangerous patterns
        """
        if not filename:
            raise ValidationError("Filename cannot be empty")
        
        filename = str(filename).strip()
        
        # Check for path traversal
        if any(pattern in filename.lower() for pattern in self.path_traversal_patterns):
            raise SecurityError(
                "Path traversal in filename", 
                violation_type="path_traversal"
            )
        
        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_')
        
        # Keep only safe characters for filenames
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Limit length
        max_length = self.config.security.max_filename_length
        if len(safe_filename) > max_length:
            name, ext = safe_filename.rsplit('.', 1) if '.' in safe_filename else (safe_filename, '')
            name = name[:max_length - len(ext) - 1]
            safe_filename = f"{name}.{ext}" if ext else name
        
        # Ensure filename is not empty after sanitization
        if not safe_filename or safe_filename in ['.', '..']:
            safe_filename = f"sanitized_{hash(filename) % 10000}.txt"
        
        return safe_filename
    
    def _validate_investment_logic(self, data: Dict[str, Any]) -> None:
        """
        Validate business logic for investment parameters.
        
        Args:
            data: Validated input data
            
        Raises:
            ValidationError: If business logic validation fails
        """
        lump_sum = data['lump_sum']
        monthly = data['monthly']
        target_value = data['target_value']
        timeframe = data['timeframe']
        
        # At least one investment type must be non-zero
        if lump_sum == 0 and monthly == 0:
            raise ValidationError(
                "Either lump sum or monthly investment must be greater than zero"
            )
        
        # Basic feasibility check
        total_contributions = lump_sum + (monthly * 12 * timeframe)
        
        if total_contributions > target_value:
            # This is actually good - they're contributing more than their target
            pass
        elif total_contributions < target_value * 0.1:  # Less than 10% of target
            raise ValidationError(
                "Total contributions are too low compared to target value. "
                "Consider increasing investments or extending timeframe."
            )
        
        # Check if monthly investment is reasonable for timeframe
        if monthly > 0 and timeframe < 1:
            raise ValidationError(
                "Monthly investments require a timeframe of at least 1 year"
            )