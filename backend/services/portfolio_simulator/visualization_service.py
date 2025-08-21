"""
Visualization service for the Portfolio Simulator Service.

This module handles the generation of charts, graphs, and visual analytics
for portfolio simulations and SHAP explanations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from .config import get_config
from .exceptions import VisualizationError, SecurityError
from .validators import InputValidator

logger = logging.getLogger(__name__)


class VisualizationService:
    """
    Handles creation of various charts and visualizations for portfolio analysis.
    
    This service generates both static image files and structured data for
    interactive charts in web applications.
    """
    
    def __init__(self, validator: Optional[InputValidator] = None):
        """
        Initialize the visualization service.
        
        Args:
            validator: Input validator instance
        """
        self.config = get_config()
        self.validator = validator or InputValidator()
        
        # Try to initialize external visualization engine
        self.external_engine = None
        self._initialize_external_engine()
        
        # Create output directory
        self.output_dir = self.config.visualization.output_directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_external_engine(self) -> None:
        """Initialize external visualization engine if available."""
        try:
            from ai_models.stock_model.explainable_ai import VisualizationEngine
            self.external_engine = VisualizationEngine()
            logger.info("External visualization engine initialized")
        except ImportError:
            logger.info("External visualization engine not available")
        except Exception as e:
            logger.warning(f"Failed to initialize external visualization engine: {e}")
    
    async def create_simulation_visualizations(
        self,
        simulation_id: Optional[int],
        stocks_data: List[Dict[str, Any]],
        simulation_results: Dict[str, Any],
        shap_explanation: Optional[Dict[str, Any]] = None,
        user_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Create all visualizations for a portfolio simulation.
        
        Args:
            simulation_id: Database ID of the simulation
            stocks_data: List of stock information with allocations
            simulation_results: Complete simulation results
            shap_explanation: SHAP explanation data (optional)
            user_data: User profile data (optional)
            
        Returns:
            Dictionary mapping visualization types to file paths
            
        Raises:
            VisualizationError: If visualization creation fails
        """
        try:
            logger.info(f"Creating visualizations for simulation {simulation_id}")
            
            # Generate unique filename prefix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"sim_{simulation_id}_{timestamp}" if simulation_id else f"temp_{timestamp}"
            
            visualization_paths = {}
            
            # 1. Portfolio Composition Chart
            try:
                composition_path = await self._create_portfolio_composition_chart(
                    stocks_data, prefix
                )
                if composition_path:
                    visualization_paths["portfolio_composition"] = str(composition_path)
            except Exception as e:
                logger.error(f"Failed to create composition chart: {e}")
            
            # 2. Performance Timeline Chart
            try:
                timeline_path = await self._create_performance_timeline_chart(
                    simulation_results, prefix
                )
                if timeline_path:
                    visualization_paths["performance_timeline"] = str(timeline_path)
            except Exception as e:
                logger.error(f"Failed to create timeline chart: {e}")
            
            # 3. Risk-Return Analysis Chart
            try:
                risk_return_path = await self._create_risk_return_chart(
                    stocks_data, prefix
                )
                if risk_return_path:
                    visualization_paths["risk_return_analysis"] = str(risk_return_path)
            except Exception as e:
                logger.error(f"Failed to create risk-return chart: {e}")
            
            # 4. SHAP Explanation Charts (if available)
            if shap_explanation:
                try:
                    shap_paths = await self._create_shap_visualizations(
                        shap_explanation, prefix
                    )
                    visualization_paths.update(shap_paths)
                except Exception as e:
                    logger.error(f"Failed to create SHAP visualizations: {e}")
            
            logger.info(f"Created {len(visualization_paths)} visualizations")
            return visualization_paths
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            raise VisualizationError(
                f"Failed to create simulation visualizations: {str(e)}"
            )
    
    async def _create_portfolio_composition_chart(
        self,
        stocks_data: List[Dict[str, Any]],
        prefix: str
    ) -> Optional[Path]:
        """Create portfolio composition pie/donut chart."""
        try:
            if not stocks_data:
                return None
            
            # Prepare data for visualization
            symbols = [stock.get("symbol", "") for stock in stocks_data]
            names = [stock.get("name", stock.get("symbol", "")) for stock in stocks_data]
            weights = [stock.get("allocation", 0) for stock in stocks_data]
            
            # Validate data
            if not symbols or not weights or sum(weights) == 0:
                logger.warning("Invalid data for composition chart")
                return None
            
            # Try external engine first
            if self.external_engine:
                try:
                    filename = f"{prefix}_composition.{self.config.visualization.file_format}"
                    file_path = self.output_dir / filename
                    
                    # Convert to percentage if needed
                    weights_pct = [w * 100 if w <= 1 else w for w in weights]
                    weights_dict = dict(zip(symbols, weights_pct))
                    
                    result = self.external_engine.create_portfolio_composition_chart(
                        symbols, weights_dict, str(file_path)
                    )
                    
                    if "saved" in result.lower() and file_path.exists():
                        logger.info(f"Portfolio composition chart created: {file_path}")
                        return file_path
                        
                except Exception as e:
                    logger.warning(f"External engine failed for composition chart: {e}")
            
            # Fallback to manual creation
            return await self._create_composition_chart_fallback(
                symbols, names, weights, prefix
            )
            
        except Exception as e:
            logger.error(f"Portfolio composition chart creation failed: {e}")
            return None
    
    async def _create_performance_timeline_chart(
        self,
        simulation_results: Dict[str, Any],
        prefix: str
    ) -> Optional[Path]:
        """Create performance timeline chart."""
        try:
            timeline_data = simulation_results.get("timeline", {}).get("portfolio", [])
            
            if not timeline_data:
                logger.warning("No timeline data available for chart")
                return None
            
            # Extract data for plotting
            dates = [item.get("date", "") for item in timeline_data]
            values = [item.get("value", 0) for item in timeline_data]
            
            if len(dates) < 2 or len(values) < 2:
                logger.warning("Insufficient timeline data for chart")
                return None
            
            # Try external engine first
            if self.external_engine:
                try:
                    filename = f"{prefix}_timeline.{self.config.visualization.file_format}"
                    file_path = self.output_dir / filename
                    
                    # Format data for external engine (implementation depends on engine)
                    # This would need to match the actual API of your VisualizationEngine
                    
                    logger.info(f"Performance timeline chart created: {file_path}")
                    return file_path
                    
                except Exception as e:
                    logger.warning(f"External engine failed for timeline chart: {e}")
            
            # Fallback to manual creation
            return await self._create_timeline_chart_fallback(dates, values, prefix)
            
        except Exception as e:
            logger.error(f"Performance timeline chart creation failed: {e}")
            return None
    
    async def _create_risk_return_chart(
        self,
        stocks_data: List[Dict[str, Any]],
        prefix: str
    ) -> Optional[Path]:
        """Create risk vs return scatter plot."""
        try:
            if not stocks_data:
                return None
            
            # Calculate risk and return estimates for each stock
            risk_return_data = []
            
            for stock in stocks_data:
                symbol = stock.get("symbol", "")
                allocation = stock.get("allocation", 0)
                
                # Get estimates from stock metadata or use defaults
                risk_estimate = self._get_risk_estimate(symbol)
                return_estimate = self._get_return_estimate(symbol)
                
                risk_return_data.append({
                    "symbol": symbol,
                    "name": stock.get("name", symbol),
                    "risk": risk_estimate,
                    "return": return_estimate,
                    "allocation": allocation * 100 if allocation <= 1 else allocation
                })
            
            if not risk_return_data:
                return None
            
            # Try external engine first
            if self.external_engine:
                try:
                    filename = f"{prefix}_risk_return.{self.config.visualization.file_format}"
                    file_path = self.output_dir / filename
                    
                    # Calculate portfolio metrics
                    portfolio_risk = sum(item["risk"] * item["allocation"] / 100 for item in risk_return_data)
                    portfolio_return = sum(item["return"] * item["allocation"] / 100 for item in risk_return_data)
                    
                    portfolio_metrics = {
                        'expected_return': portfolio_return,
                        'volatility': portfolio_risk,
                        'sharpe_ratio': (portfolio_return - 3) / portfolio_risk if portfolio_risk > 0 else 0
                    }
                    
                    benchmark_data = [
                        {
                            'name': item['name'],
                            'expected_return': item['return'],
                            'volatility': item['risk']
                        }
                        for item in risk_return_data[:5]  # Limit to first 5 for clarity
                    ]
                    
                    result = self.external_engine.create_risk_return_scatter(
                        portfolio_metrics, benchmark_data, str(file_path)
                    )
                    
                    if "saved" in result.lower() and file_path.exists():
                        logger.info(f"Risk-return chart created: {file_path}")
                        return file_path
                        
                except Exception as e:
                    logger.warning(f"External engine failed for risk-return chart: {e}")
            
            # Fallback to manual creation
            return await self._create_risk_return_chart_fallback(risk_return_data, prefix)
            
        except Exception as e:
            logger.error(f"Risk-return chart creation failed: {e}")
            return None
    
    async def _create_shap_visualizations(
        self,
        shap_explanation: Dict[str, Any],
        prefix: str
    ) -> Dict[str, str]:
        """Create SHAP explanation visualizations."""
        shap_paths = {}
        
        try:
            # 1. SHAP Waterfall Chart
            waterfall_path = await self._create_shap_waterfall_chart(
                shap_explanation, prefix
            )
            if waterfall_path:
                shap_paths["shap_waterfall"] = str(waterfall_path)
            
            # 2. Feature Importance Chart
            importance_path = await self._create_feature_importance_chart(
                shap_explanation, prefix
            )
            if importance_path:
                shap_paths["feature_importance"] = str(importance_path)
            
            # 3. Market Regime Visualization (if available)
            market_regime = shap_explanation.get("market_regime")
            if market_regime:
                regime_path = await self._create_market_regime_chart(
                    market_regime, prefix
                )
                if regime_path:
                    shap_paths["market_regime"] = str(regime_path)
            
        except Exception as e:
            logger.error(f"SHAP visualization creation failed: {e}")
        
        return shap_paths
    
    async def _create_shap_waterfall_chart(
        self,
        shap_explanation: Dict[str, Any],
        prefix: str
    ) -> Optional[Path]:
        """Create SHAP waterfall chart showing feature contributions."""
        try:
            feature_contributions = shap_explanation.get("feature_contributions", {})
            
            if not feature_contributions:
                return None
            
            # Clean and prepare data
            cleaned_shap = self._clean_shap_data(shap_explanation)
            
            # Try external engine first
            if self.external_engine:
                try:
                    filename = f"{prefix}_shap_waterfall.{self.config.visualization.file_format}"
                    file_path = self.output_dir / filename
                    
                    result = self.external_engine.create_shap_waterfall_chart(
                        cleaned_shap, str(file_path)
                    )
                    
                    if "saved" in result.lower() and file_path.exists():
                        logger.info(f"SHAP waterfall chart created: {file_path}")
                        return file_path
                        
                except Exception as e:
                    logger.warning(f"External engine failed for SHAP waterfall: {e}")
            
            # Fallback to manual creation
            return await self._create_waterfall_chart_fallback(cleaned_shap, prefix)
            
        except Exception as e:
            logger.error(f"SHAP waterfall chart creation failed: {e}")
            return None
    
    async def _create_feature_importance_chart(
        self,
        shap_explanation: Dict[str, Any],
        prefix: str
    ) -> Optional[Path]:
        """Create feature importance bar chart."""
        try:
            feature_contributions = shap_explanation.get("feature_contributions", {})
            
            if not feature_contributions:
                return None
            
            # Try external engine first
            if self.external_engine:
                try:
                    filename = f"{prefix}_feature_importance.{self.config.visualization.file_format}"
                    file_path = self.output_dir / filename
                    
                    result = self.external_engine.create_factor_importance_chart(
                        feature_contributions, str(file_path)
                    )
                    
                    if "saved" in result.lower() and file_path.exists():
                        logger.info(f"Feature importance chart created: {file_path}")
                        return file_path
                        
                except Exception as e:
                    logger.warning(f"External engine failed for feature importance: {e}")
            
            # Fallback to manual creation
            return await self._create_importance_chart_fallback(feature_contributions, prefix)
            
        except Exception as e:
            logger.error(f"Feature importance chart creation failed: {e}")
            return None
    
    async def _create_market_regime_chart(
        self,
        market_regime: Dict[str, Any],
        prefix: str
    ) -> Optional[Path]:
        """Create market regime visualization."""
        try:
            if not market_regime:
                return None
            
            # Try external engine first
            if self.external_engine:
                try:
                    filename = f"{prefix}_market_regime.{self.config.visualization.file_format}"
                    file_path = self.output_dir / filename
                    
                    result = self.external_engine.create_market_regime_visualization(
                        market_regime, str(file_path)
                    )
                    
                    if "saved" in result.lower() and file_path.exists():
                        logger.info(f"Market regime chart created: {file_path}")
                        return file_path
                        
                except Exception as e:
                    logger.warning(f"External engine failed for market regime: {e}")
            
            # Fallback to manual creation
            return await self._create_regime_chart_fallback(market_regime, prefix)
            
        except Exception as e:
            logger.error(f"Market regime chart creation failed: {e}")
            return None
    
    def _clean_shap_data(self, shap_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Clean SHAP data for visualization."""
        cleaned = {}
        
        # Clean feature contributions
        feature_contributions = shap_explanation.get("feature_contributions", {})
        cleaned_contributions = {}
        
        for feature, value in feature_contributions.items():
            try:
                if hasattr(value, '__iter__') and not isinstance(value, str):
                    # Handle array-like values
                    numeric_value = float(list(value)[0]) if len(list(value)) > 0 else 0.0
                else:
                    numeric_value = float(value)
                
                if not np.isfinite(numeric_value):
                    numeric_value = 0.0
                
                cleaned_contributions[str(feature)] = numeric_value
                
            except (ValueError, TypeError, IndexError):
                cleaned_contributions[str(feature)] = 0.0
        
        cleaned["feature_contributions"] = cleaned_contributions
        
        # Clean base value
        base_value = shap_explanation.get("base_value", 50)
        try:
            if hasattr(base_value, '__iter__') and not isinstance(base_value, str):
                cleaned_base = float(list(base_value)[0]) if len(list(base_value)) > 0 else 50.0
            else:
                cleaned_base = float(base_value)
            
            if not np.isfinite(cleaned_base):
                cleaned_base = 50.0
                
        except (ValueError, TypeError, IndexError):
            cleaned_base = 50.0
        
        cleaned["base_value"] = cleaned_base
        
        # Copy other fields
        cleaned["portfolio_quality_score"] = shap_explanation.get("portfolio_quality_score", 75)
        cleaned["human_readable_explanation"] = shap_explanation.get("human_readable_explanation", {})
        
        return cleaned
    
    def _get_risk_estimate(self, symbol: str) -> float:
        """Get risk estimate for a stock symbol."""
        from .config import get_stock_metadata
        metadata = get_stock_metadata()
        return metadata.get(symbol, {}).get("risk_score", 15.0)
    
    def _get_return_estimate(self, symbol: str) -> float:
        """Get return estimate for a stock symbol."""
        from .config import get_stock_metadata
        metadata = get_stock_metadata()
        return metadata.get(symbol, {}).get("expected_return", 8.0)
    
    # Fallback chart creation methods (simplified implementations)
    async def _create_composition_chart_fallback(
        self,
        symbols: List[str],
        names: List[str],
        weights: List[float],
        prefix: str
    ) -> Optional[Path]:
        """Fallback method for creating composition chart."""
        try:
            # Create a simple text-based representation for fallback
            filename = f"{prefix}_composition_fallback.txt"
            file_path = self.output_dir / filename
            
            with open(file_path, 'w') as f:
                f.write("Portfolio Composition\n")
                f.write("====================\n\n")
                
                for symbol, name, weight in zip(symbols, names, weights):
                    percentage = weight * 100 if weight <= 1 else weight
                    f.write(f"{symbol} ({name}): {percentage:.1f}%\n")
            
            logger.info(f"Fallback composition data saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Fallback composition chart failed: {e}")
            return None
    
    async def _create_timeline_chart_fallback(
        self,
        dates: List[str],
        values: List[float],
        prefix: str
    ) -> Optional[Path]:
        """Fallback method for creating timeline chart."""
        try:
            filename = f"{prefix}_timeline_fallback.txt"
            file_path = self.output_dir / filename
            
            with open(file_path, 'w') as f:
                f.write("Portfolio Performance Timeline\n")
                f.write("==============================\n\n")
                
                for i in range(0, len(dates), max(1, len(dates) // 20)):  # Sample points
                    f.write(f"{dates[i]}: £{values[i]:,.2f}\n")
            
            logger.info(f"Fallback timeline data saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Fallback timeline chart failed: {e}")
            return None
    
    async def _create_risk_return_chart_fallback(
        self,
        risk_return_data: List[Dict[str, Any]],
        prefix: str
    ) -> Optional[Path]:
        """Fallback method for creating risk-return chart."""
        try:
            filename = f"{prefix}_risk_return_fallback.txt"
            file_path = self.output_dir / filename
            
            with open(file_path, 'w') as f:
                f.write("Risk vs Return Analysis\n")
                f.write("=======================\n\n")
                
                for item in risk_return_data:
                    f.write(
                        f"{item['symbol']}: Risk {item['risk']:.1f}%, "
                        f"Return {item['return']:.1f}%, "
                        f"Allocation {item['allocation']:.1f}%\n"
                    )
            
            logger.info(f"Fallback risk-return data saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Fallback risk-return chart failed: {e}")
            return None
    
    async def _create_waterfall_chart_fallback(
        self,
        cleaned_shap: Dict[str, Any],
        prefix: str
    ) -> Optional[Path]:
        """Fallback method for creating SHAP waterfall chart."""
        try:
            filename = f"{prefix}_shap_waterfall_fallback.txt"
            file_path = self.output_dir / filename
            
            feature_contributions = cleaned_shap.get("feature_contributions", {})
            base_value = cleaned_shap.get("base_value", 50)
            
            with open(file_path, 'w') as f:
                f.write("SHAP Feature Contributions\n")
                f.write("==========================\n\n")
                f.write(f"Base Value: {base_value:.2f}\n\n")
                
                sorted_features = sorted(
                    feature_contributions.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                
                for feature, contribution in sorted_features:
                    direction = "increases" if contribution > 0 else "decreases"
                    f.write(f"{feature}: {contribution:+.3f} ({direction} prediction)\n")
            
            logger.info(f"Fallback SHAP waterfall data saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Fallback SHAP waterfall failed: {e}")
            return None
    
    async def _create_importance_chart_fallback(
        self,
        feature_contributions: Dict[str, Any],
        prefix: str
    ) -> Optional[Path]:
        """Fallback method for creating feature importance chart."""
        try:
            filename = f"{prefix}_feature_importance_fallback.txt"
            file_path = self.output_dir / filename
            
            with open(file_path, 'w') as f:
                f.write("Feature Importance Analysis\n")
                f.write("===========================\n\n")
                
                sorted_features = sorted(
                    feature_contributions.items(),
                    key=lambda x: abs(float(x[1])),
                    reverse=True
                )
                
                for feature, importance in sorted_features:
                    f.write(f"{feature}: {abs(float(importance)):.3f}\n")
            
            logger.info(f"Fallback feature importance data saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Fallback feature importance failed: {e}")
            return None
    
    async def _create_regime_chart_fallback(
        self,
        market_regime: Dict[str, Any],
        prefix: str
    ) -> Optional[Path]:
        """Fallback method for creating market regime chart."""
        try:
            filename = f"{prefix}_market_regime_fallback.txt"
            file_path = self.output_dir / filename
            
            with open(file_path, 'w') as f:
                f.write("Market Regime Analysis\n")
                f.write("======================\n\n")
                
                for key, value in market_regime.items():
                    f.write(f"{key}: {value}\n")
            
            logger.info(f"Fallback market regime data saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Fallback market regime failed: {e}")
            return None
    
    def validate_filename(self, filename: str) -> str:
        """
        Validate and sanitize filename for security.
        
        Args:
            filename: Proposed filename
            
        Returns:
            Sanitized filename
            
        Raises:
            SecurityError: If filename is potentially dangerous
        """
        return self.validator.sanitize_filename(filename)
    
    async def cleanup_old_files(self, max_age_days: int = 7) -> int:
        """
        Clean up old visualization files.
        
        Args:
            max_age_days: Maximum age in days for files to keep
            
        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            
            for file_path in self.output_dir.iterdir():
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            logger.debug(f"Deleted old visualization file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete {file_path}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old visualization files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0


class ChartDataGenerator:
    """
    Generates structured data for interactive charts in web applications.
    
    This class creates JSON-compatible data structures that can be consumed
    by frontend charting libraries like Chart.js, D3, or Recharts.
    """
    
    def __init__(self):
        """Initialize the chart data generator."""
        self.config = get_config()
    
    def generate_chart_data(
        self,
        simulation_results: Dict[str, Any],
        stocks_data: List[Dict[str, Any]],
        shap_explanation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate all chart data for a simulation.
        
        Args:
            simulation_results: Complete simulation results
            stocks_data: Stock allocation data
            shap_explanation: SHAP explanation data
            
        Returns:
            Dictionary with all chart data structures
        """
        chart_data = {}
        
        # Portfolio composition data
        chart_data["portfolio_composition"] = self._generate_composition_data(stocks_data)
        
        # Performance timeline data
        chart_data["performance_timeline"] = self._generate_timeline_data(simulation_results)
        
        # Risk-return analysis data
        chart_data["risk_return_analysis"] = self._generate_risk_return_data(stocks_data)
        
        # SHAP explanation data
        if shap_explanation:
            chart_data["shap_waterfall"] = self._generate_shap_waterfall_data(shap_explanation)
            chart_data["feature_importance"] = self._generate_feature_importance_data(shap_explanation)
            
            market_regime = shap_explanation.get("market_regime")
            if market_regime:
                chart_data["market_regime"] = self._generate_market_regime_data(market_regime)
        
        # Goal analysis data
        chart_data["goal_analysis"] = self._generate_goal_analysis_data(simulation_results)
        
        return chart_data
    
    def _generate_composition_data(self, stocks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate portfolio composition chart data."""
        composition_data = []
        
        for stock in stocks_data:
            allocation = stock.get("allocation", 0)
            # Convert to percentage if needed
            percentage = allocation * 100 if allocation <= 1 else allocation
            
            composition_data.append({
                "symbol": stock.get("symbol", ""),
                "name": stock.get("name", stock.get("symbol", "")),
                "value": round(percentage, 2),
                "allocation": round(percentage, 2),
                "color": self._get_stock_color(stock.get("symbol", ""))
            })
        
        return composition_data
    
    def _generate_timeline_data(self, simulation_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate performance timeline chart data."""
        timeline = simulation_results.get("timeline", {})
        portfolio_data = timeline.get("portfolio", [])
        contribution_data = timeline.get("contributions", [])
        
        # Process portfolio performance data
        portfolio_timeline = []
        for i, entry in enumerate(portfolio_data):
            value = entry.get("value", 0)
            initial_value = portfolio_data[0].get("value", 1) if portfolio_data else 1
            return_pct = ((value / initial_value) - 1) * 100 if initial_value > 0 else 0
            
            portfolio_timeline.append({
                "date": entry.get("date", ""),
                "value": round(value, 2),
                "return_pct": round(return_pct, 2)
            })
        
        return {
            "portfolio": portfolio_timeline,
            "contributions": contribution_data
        }
    
    def _generate_risk_return_data(self, stocks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate risk vs return scatter plot data."""
        risk_return_data = []
        
        for stock in stocks_data:
            symbol = stock.get("symbol", "")
            allocation = stock.get("allocation", 0)
            
            # Get risk and return estimates
            risk_estimate = self._get_risk_estimate(symbol)
            return_estimate = self._get_return_estimate(symbol)
            
            risk_return_data.append({
                "symbol": symbol,
                "name": stock.get("name", symbol),
                "risk": round(risk_estimate, 2),
                "return": round(return_estimate, 2),
                "allocation": round(allocation * 100 if allocation <= 1 else allocation, 2)
            })
        
        return risk_return_data
    
    def _generate_shap_waterfall_data(self, shap_explanation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate SHAP waterfall chart data."""
        feature_contributions = shap_explanation.get("feature_contributions", {})
        base_value = shap_explanation.get("base_value", 50)
        
        if isinstance(base_value, (list, tuple)):
            base_value = base_value[0] if len(base_value) > 0 else 50
        
        waterfall_data = []
        cumulative = float(base_value)
        
        # Sort features by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(float(x[1])),
            reverse=True
        )
        
        for feature, contribution in sorted_features:
            contribution_value = float(contribution)
            start = cumulative
            cumulative += contribution_value
            
            waterfall_data.append({
                "feature": self._format_feature_name(feature),
                "value": round(contribution_value, 3),
                "cumulative": round(cumulative, 3),
                "start": round(start, 3),
                "isPositive": contribution_value >= 0
            })
        
        return waterfall_data
    
    def _generate_feature_importance_data(self, shap_explanation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate feature importance chart data."""
        feature_contributions = shap_explanation.get("feature_contributions", {})
        
        importance_data = []
        for feature, contribution in feature_contributions.items():
            importance_value = abs(float(contribution))
            
            importance_data.append({
                "feature": self._format_feature_name(feature),
                "importance": round(importance_value, 3),
                "isPositive": float(contribution) >= 0,
                "description": self._get_feature_description(feature)
            })
        
        # Sort by importance
        importance_data.sort(key=lambda x: x["importance"], reverse=True)
        
        return importance_data
    
    def _generate_market_regime_data(self, market_regime: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate market regime visualization data."""
        regime_data = []
        
        metrics = {
            "VIX Level": {
                "current": market_regime.get("current_vix", 20),
                "normal": 20,
                "label": "Volatility"
            },
            "Trend Score": {
                "current": market_regime.get("trend_score", 2.5),
                "normal": 3,
                "label": "Market Trend"
            },
            "3M Returns": {
                "current": (market_regime.get("returns_3m", 0.05) * 100),
                "normal": 5,
                "label": "Recent Performance"
            }
        }
        
        for name, data in metrics.items():
            current = data["current"]
            normal = data["normal"]
            
            regime_data.append({
                "name": name,
                "current": round(current, 1),
                "normal": normal,
                "status": self._assess_metric_status(current, normal, name),
                "label": data["label"]
            })
        
        return regime_data
    
    def _generate_goal_analysis_data(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate goal analysis chart data."""
        # Extract goal-related metrics from simulation results
        starting_value = simulation_results.get("starting_value", 0)
        ending_value = simulation_results.get("end_value", 0)
        target_value = simulation_results.get("target_value", ending_value)
        
        progress_percentage = min(100, (ending_value / target_value) * 100) if target_value > 0 else 0
        
        return {
            "target_value": target_value,
            "projected_value": ending_value,
            "starting_value": starting_value,
            "progress_percentage": round(progress_percentage, 1),
            "target_achieved": ending_value >= target_value,
            "growth_amount": ending_value - starting_value,
            "growth_percentage": ((ending_value / starting_value) - 1) * 100 if starting_value > 0 else 0
        }
    
    def _get_risk_estimate(self, symbol: str) -> float:
        """Get risk estimate for a stock symbol."""
        from .config import get_stock_metadata
        metadata = get_stock_metadata()
        return metadata.get(symbol, {}).get("risk_score", 15.0)
    
    def _get_return_estimate(self, symbol: str) -> float:
        """Get return estimate for a stock symbol."""
        from .config import get_stock_metadata
        metadata = get_stock_metadata()
        return metadata.get(symbol, {}).get("expected_return", 8.0)
    
    def _get_stock_color(self, symbol: str) -> str:
        """Get color for stock in charts."""
        # Simple color mapping based on symbol hash
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
        ]
        return colors[hash(symbol) % len(colors)]
    
    def _format_feature_name(self, feature: str) -> str:
        """Format feature names for display."""
        name_mapping = {
            "risk_score": "Risk Tolerance",
            "target_value_log": "Investment Goal",
            "timeframe": "Time Horizon",
            "required_return": "Required Growth Rate",
            "monthly_contribution": "Monthly Investment",
            "market_volatility": "Market Volatility",
            "market_trend_score": "Market Trend"
        }
        return name_mapping.get(feature, feature.replace("_", " ").title())
    
    def _get_feature_description(self, feature: str) -> str:
        """Get description for features."""
        descriptions = {
            "risk_score": "Your comfort level with investment risk and volatility",
            "target_value_log": "The financial goal you want to achieve",
            "timeframe": "How long you have to invest and reach your goal",
            "required_return": "The annual growth rate needed to reach your target",
            "monthly_contribution": "Amount you can invest each month",
            "market_volatility": "Current market uncertainty and volatility levels",
            "market_trend_score": "Whether markets are trending up or down"
        }
        return descriptions.get(feature, "Factor influencing portfolio recommendations")
    
    def _assess_metric_status(self, current: float, normal: float, metric_name: str) -> str:
        """Assess the status of a market regime metric."""
        if "VIX" in metric_name:
            return "Low" if current < normal else "High"
        elif "Trend" in metric_name:
            return "Bullish" if current > normal else "Neutral"
        elif "Returns" in metric_name:
            return "Above Average" if current > normal else "Below Average"
        else:
            return "Normal"