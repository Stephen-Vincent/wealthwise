"""
Visualization Module for Explainable AI

This module creates beautiful visualizations that explain AI decision-making
processes. It generates professional charts showing why specific portfolios
are recommended, making AI decisions transparent and educational.

Key Features:
1. SHAP waterfall charts showing factor contributions
2. Portfolio composition visualizations
3. Risk-return scatter plots
4. Factor importance charts
5. Market regime visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings

# Optional dependencies for enhanced visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Professional Visualization Engine for Explainable AI
    
    Creates publication-quality charts and visualizations that explain
    AI investment decisions in an intuitive, educational manner.
    """
    
    def __init__(self):
        """Initialize the visualization engine with styling"""
        # Set up matplotlib styling
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color schemes for different chart types
        self.colors = {
            'positive': '#2E8B57',      # Sea Green
            'negative': '#DC143C',      # Crimson
            'neutral': '#4682B4',       # Steel Blue
            'background': '#F5F5F5',    # White Smoke
            'accent': '#FF6347'         # Tomato
        }
        
        # Font configurations
        self.font_config = {
            'title_size': 14,
            'label_size': 12,
            'tick_size': 10,
            'legend_size': 10
        }
    
    def create_shap_waterfall_chart(self, shap_explanation: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> str:
        """
        Create beautiful SHAP waterfall chart showing AI decision factors
        
        This visualization shows how each factor contributes to the final
        portfolio recommendation, making AI decisions completely transparent.
        
        Args:
            shap_explanation: Dictionary from SHAPExplainer.get_shap_explanation()
            save_path: Optional path to save the chart
            
        Returns:
            String indicating success/failure
        """
        if not SHAP_AVAILABLE:
            return "SHAP not available for waterfall charts"
        
        try:
            logger.info("📊 Creating SHAP waterfall visualization...")
            
            # Extract data from SHAP explanation
            feature_contributions = shap_explanation.get("feature_contributions", {})
            base_value = shap_explanation.get("base_value", 50)
            portfolio_score = shap_explanation.get("portfolio_quality_score", 50)
            
            if not feature_contributions:
                return "No feature contributions found in SHAP explanation"
            
            # Prepare data for plotting
            features = []
            values = []
            colors = []
            
            for feature, value in feature_contributions.items():
                # Make feature names more readable
                readable_name = self._make_feature_readable(feature)
                
                features.append(readable_name)
                values.append(value)
                colors.append(self.colors['positive'] if value > 0 else self.colors['negative'])
            
            # Create the waterfall chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create horizontal bar chart
            bars = ax.barh(features, values, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=0.5)
            
            # Customize the plot
            ax.set_xlabel('Impact on Portfolio Quality Score', 
                         fontsize=self.font_config['label_size'], fontweight='bold')
            ax.set_title('Why This Portfolio Was Recommended\nAI Explanation Using SHAP Values', 
                        fontsize=self.font_config['title_size'], fontweight='bold', pad=20)
            
            # Add vertical line at x=0
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax.text(width + (0.1 if width >= 0 else -0.1), 
                       bar.get_y() + bar.get_height()/2,
                       f'{value:.1f}', ha='left' if width >= 0 else 'right', 
                       va='center', fontweight='bold', 
                       fontsize=self.font_config['tick_size'])
            
            # Add explanation text
            explanation_text = (
                f'Portfolio Quality Score: {portfolio_score:.1f}/100 (Base: {base_value:.1f})\n'
                f'Green bars improve the recommendation, red bars reduce it.\n'
                f'This shows exactly why the AI chose this portfolio for your goals.'
            )
            
            plt.figtext(0.02, 0.02, explanation_text, fontsize=10, 
                       style='italic', wrap=True)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                logger.info(f"📊 SHAP waterfall chart saved to {save_path}")
                return f"SHAP waterfall chart saved to {save_path}"
            else:
                plt.show()
                return "SHAP waterfall chart displayed successfully"
                
        except Exception as e:
            logger.error(f"SHAP waterfall chart creation failed: {e}")
            return f"Could not create waterfall chart: {str(e)}"
    
    def create_portfolio_composition_chart(self, stocks: List[str], 
                                          weights: Dict[str, float],
                                          save_path: Optional[str] = None) -> str:
        """
        Create professional portfolio composition visualization
        
        Shows the recommended portfolio allocation with clear labels
        and professional styling.
        
        Args:
            stocks: List of stock tickers
            weights: Dictionary of stock weights
            save_path: Optional save path
            
        Returns:
            Status message
        """
        try:
            logger.info(f"📊 Creating portfolio composition chart for {len(stocks)} holdings...")
            
            # Prepare data
            valid_stocks = [stock for stock in stocks if stock in weights and weights[stock] > 0]
            valid_weights = [weights[stock] for stock in valid_stocks]
            
            if not valid_stocks:
                return "No valid portfolio data to visualize"
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create pie chart with custom colors
            wedges, texts, autotexts = ax.pie(
                valid_weights, 
                labels=valid_stocks,
                autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',
                startangle=90,
                colors=sns.color_palette("husl", len(valid_stocks))
            )
            
            # Enhance text styling
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            for text in texts:
                text.set_fontsize(11)
                text.set_fontweight('bold')
            
            # Add title and styling
            ax.set_title('Recommended Portfolio Allocation', 
                        fontsize=self.font_config['title_size'], 
                        fontweight='bold', pad=20)
            
            # Add legend for small allocations
            small_allocations = [(stock, weight) for stock, weight in zip(valid_stocks, valid_weights) if weight < 5]
            if small_allocations:
                legend_text = "Small allocations: " + ", ".join([f"{s}: {w:.1f}%" for s, w in small_allocations])
                plt.figtext(0.02, 0.02, legend_text, fontsize=9, style='italic')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                logger.info(f"📊 Portfolio composition chart saved to {save_path}")
                return f"Portfolio composition chart saved to {save_path}"
            else:
                plt.show()
                return "Portfolio composition chart displayed successfully"
                
        except Exception as e:
            logger.error(f"Portfolio composition chart creation failed: {e}")
            return f"Could not create composition chart: {str(e)}"
    
    def create_risk_return_scatter(self, portfolio_metrics: Dict[str, float],
                                  benchmark_data: Optional[List[Dict]] = None,
                                  save_path: Optional[str] = None) -> str:
        """
        Create risk-return scatter plot showing portfolio positioning
        
        Visualizes where the recommended portfolio sits in risk-return space
        compared to benchmarks or alternative portfolios.
        
        Args:
            portfolio_metrics: Dict with expected_return, volatility, sharpe_ratio
            benchmark_data: Optional list of benchmark portfolios
            save_path: Optional save path
            
        Returns:
            Status message
        """
        try:
            logger.info("📊 Creating risk-return scatter plot...")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Extract portfolio metrics
            portfolio_return = portfolio_metrics.get('expected_return', 0.08)
            portfolio_vol = portfolio_metrics.get('volatility', 0.15)
            portfolio_sharpe = portfolio_metrics.get('sharpe_ratio', 0.5)
            
            # Plot the recommended portfolio
            ax.scatter(portfolio_vol * 100, portfolio_return * 100, 
                      s=200, c=self.colors['accent'], marker='*', 
                      edgecolors='black', linewidth=2, 
                      label=f'Recommended Portfolio\n(Sharpe: {portfolio_sharpe:.2f})',
                      zorder=5)
            
            # Plot benchmarks if provided
            if benchmark_data:
                for i, benchmark in enumerate(benchmark_data):
                    ax.scatter(benchmark.get('volatility', 0.15) * 100,
                              benchmark.get('expected_return', 0.08) * 100,
                              s=100, alpha=0.7, 
                              label=benchmark.get('name', f'Benchmark {i+1}'))
            
            # Add efficient frontier if we have multiple points
            if benchmark_data and len(benchmark_data) > 3:
                vols = [b.get('volatility', 0.15) * 100 for b in benchmark_data]
                rets = [b.get('expected_return', 0.08) * 100 for b in benchmark_data]
                
                # Sort by volatility for smooth line
                sorted_pairs = sorted(zip(vols, rets))
                sorted_vols, sorted_rets = zip(*sorted_pairs)
                
                ax.plot(sorted_vols, sorted_rets, '--', alpha=0.5, 
                       color=self.colors['neutral'], label='Efficient Frontier')
            
            # Customize the plot
            ax.set_xlabel('Risk (Annual Volatility %)', 
                         fontsize=self.font_config['label_size'], fontweight='bold')
            ax.set_ylabel('Expected Return (%)', 
                         fontsize=self.font_config['label_size'], fontweight='bold')
            ax.set_title('Portfolio Risk-Return Profile', 
                        fontsize=self.font_config['title_size'], fontweight='bold')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(fontsize=self.font_config['legend_size'], loc='upper left')
            
            # Add Sharpe ratio lines
            if portfolio_sharpe > 0:
                x_line = np.linspace(0, max(25, portfolio_vol * 100 * 1.5), 100)
                y_line = x_line * portfolio_sharpe * 0.01 + 2  # Assuming 2% risk-free rate
                ax.plot(x_line, y_line, ':', alpha=0.5, color='gray', 
                       label=f'Sharpe {portfolio_sharpe:.2f} line')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                logger.info(f"📊 Risk-return scatter plot saved to {save_path}")
                return f"Risk-return scatter plot saved to {save_path}"
            else:
                plt.show()
                return "Risk-return scatter plot displayed successfully"
                
        except Exception as e:
            logger.error(f"Risk-return scatter plot creation failed: {e}")
            return f"Could not create risk-return plot: {str(e)}"
    
    def create_factor_importance_chart(self, factor_scores: Dict[str, float],
                                     save_path: Optional[str] = None) -> str:
        """
        Create factor importance visualization
        
        Shows how different factors contribute to stock selection decisions.
        
        Args:
            factor_scores: Dictionary of factor names and their importance scores
            save_path: Optional save path
            
        Returns:
            Status message
        """
        try:
            logger.info("📊 Creating factor importance chart...")
            
            # Prepare data
            factors = list(factor_scores.keys())
            scores = list(factor_scores.values())
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color bars based on positive/negative values
            colors = [self.colors['positive'] if score > 0 else self.colors['negative'] 
                     for score in scores]
            
            bars = ax.barh(factors, scores, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax.text(width + (0.01 if width >= 0 else -0.01),
                       bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', ha='left' if width >= 0 else 'right',
                       va='center', fontweight='bold')
            
            # Customize plot
            ax.set_xlabel('Factor Score', fontsize=self.font_config['label_size'], 
                         fontweight='bold')
            ax.set_title('Multi-Factor Analysis Results', 
                        fontsize=self.font_config['title_size'], fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax.grid(True, axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                logger.info(f"📊 Factor importance chart saved to {save_path}")
                return f"Factor importance chart saved to {save_path}"
            else:
                plt.show()
                return "Factor importance chart displayed successfully"
                
        except Exception as e:
            logger.error(f"Factor importance chart creation failed: {e}")
            return f"Could not create factor importance chart: {str(e)}"
    
    def create_market_regime_visualization(self, market_data: Dict[str, Any],
                                         save_path: Optional[str] = None) -> str:
        """
        Create market regime visualization
        
        Shows current market conditions and how they influence recommendations.
        
        Args:
            market_data: Dictionary from MarketRegimeDetector
            save_path: Optional save path
            
        Returns:
            Status message
        """
        try:
            logger.info("📊 Creating market regime visualization...")
            
            # Create subplot layout
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Regime Classification
            regime = market_data.get('regime', 'neutral')
            confidence = market_data.get('confidence', 0.5)
            
            ax1.pie([confidence, 1-confidence], 
                   labels=[f'{regime.title()} Market', 'Uncertainty'],
                   colors=[self.colors['positive'], self.colors['neutral']],
                   autopct='%1.1f%%', startangle=90)
            ax1.set_title('Market Regime Classification')
            
            # 2. Trend Score
            trend_score = market_data.get('trend_score', 2.5)
            ax2.bar(['Trend Strength'], [trend_score], color=self.colors['accent'])
            ax2.set_ylim(0, 5)
            ax2.set_ylabel('Score (0-5)')
            ax2.set_title('Market Trend Strength')
            ax2.grid(True, alpha=0.3)
            
            # 3. Volatility Level
            current_vix = market_data.get('current_vix', 20)
            avg_vix = market_data.get('avg_vix', 20)
            
            ax3.bar(['Current VIX', 'Average VIX'], [current_vix, avg_vix],
                   color=[self.colors['negative'], self.colors['neutral']])
            ax3.set_ylabel('VIX Level')
            ax3.set_title('Market Volatility (Fear Index)')
            ax3.grid(True, alpha=0.3)
            
            # 4. Recent Returns
            returns_1m = market_data.get('returns_1m', 0) * 100
            returns_3m = market_data.get('returns_3m', 0) * 100
            
            colors_returns = [self.colors['positive'] if r > 0 else self.colors['negative'] 
                             for r in [returns_1m, returns_3m]]
            
            ax4.bar(['1 Month', '3 Month'], [returns_1m, returns_3m], color=colors_returns)
            ax4.set_ylabel('Return (%)')
            ax4.set_title('Recent Market Performance')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.grid(True, alpha=0.3)
            
            # Add overall title
            fig.suptitle('Current Market Conditions Analysis', 
                        fontsize=self.font_config['title_size'], fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                logger.info(f"📊 Market regime visualization saved to {save_path}")
                return f"Market regime visualization saved to {save_path}"
            else:
                plt.show()
                return "Market regime visualization displayed successfully"
                
        except Exception as e:
            logger.error(f"Market regime visualization creation failed: {e}")
            return f"Could not create market regime visualization: {str(e)}"
    
    def create_interactive_dashboard(self, comprehensive_data: Dict[str, Any],
                                   save_path: Optional[str] = None) -> str:
        """
        Create interactive dashboard using Plotly (if available)
        
        Combines multiple visualizations into an interactive dashboard
        that users can explore to understand AI recommendations.
        
        Args:
            comprehensive_data: All data for creating dashboard
            save_path: Optional save path for HTML file
            
        Returns:
            Status message
        """
        if not PLOTLY_AVAILABLE:
            return "Plotly not available for interactive dashboard. Install with: pip install plotly"
        
        try:
            logger.info("📊 Creating interactive dashboard...")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Portfolio Allocation', 'Factor Contributions', 
                              'Risk-Return Profile', 'Market Conditions'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # 1. Portfolio Allocation (Pie Chart)
            portfolio_data = comprehensive_data.get('portfolio', {})
            stocks = portfolio_data.get('stocks', [])
            weights = portfolio_data.get('weights', {})
            
            if stocks and weights:
                fig.add_trace(
                    go.Pie(labels=stocks, 
                          values=[weights.get(stock, 0) for stock in stocks],
                          name="Portfolio"),
                    row=1, col=1
                )
            
            # 2. SHAP Factor Contributions (Bar Chart)
            shap_data = comprehensive_data.get('shap_explanation', {})
            feature_contributions = shap_data.get('feature_contributions', {})
            
            if feature_contributions:
                factors = [self._make_feature_readable(f) for f in feature_contributions.keys()]
                contributions = list(feature_contributions.values())
                colors_shap = ['green' if c > 0 else 'red' for c in contributions]
                
                fig.add_trace(
                    go.Bar(x=factors, y=contributions, 
                          marker_color=colors_shap,
                          name="SHAP Values"),
                    row=1, col=2
                )
            
            # 3. Risk-Return Profile (Scatter)
            metrics = comprehensive_data.get('portfolio_metrics', {})
            expected_return = metrics.get('expected_return', 0.08) * 100
            volatility = metrics.get('volatility', 0.15) * 100
            
            fig.add_trace(
                go.Scatter(x=[volatility], y=[expected_return],
                          mode='markers', marker=dict(size=15, color='red'),
                          name="Your Portfolio"),
                row=2, col=1
            )
            
            # 4. Market Conditions (Bar Chart)
            market_data = comprehensive_data.get('market_regime', {})
            trend_score = market_data.get('trend_score', 2.5)
            current_vix = market_data.get('current_vix', 20)
            
            fig.add_trace(
                go.Bar(x=['Trend Score', 'VIX Level'], 
                      y=[trend_score, current_vix],
                      marker_color=['blue', 'orange'],
                      name="Market Data"),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="AI Investment Recommendation Dashboard",
                title_x=0.5,
                height=800,
                showlegend=False
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Risk (Volatility %)", row=2, col=1)
            fig.update_yaxes(title_text="Return (%)", row=2, col=1)
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"📊 Interactive dashboard saved to {save_path}")
                return f"Interactive dashboard saved to {save_path}"
            else:
                fig.show()
                return "Interactive dashboard displayed successfully"
                
        except Exception as e:
            logger.error(f"Interactive dashboard creation failed: {e}")
            return f"Could not create interactive dashboard: {str(e)}"
    
    def _make_feature_readable(self, feature_name: str) -> str:
        """Convert technical feature names to human-readable format"""
        readable_names = {
            "target_value_log": "Investment Goal",
            "timeframe": "Time Horizon",
            "risk_score": "Risk Tolerance", 
            "required_return": "Required Return",
            "monthly_contribution": "Monthly Savings",
            "market_volatility": "Market Fear (VIX)",
            "market_trend_score": "Market Momentum"
        }
        
        return readable_names.get(feature_name, feature_name.replace('_', ' ').title())
    
    def create_comprehensive_report(self, all_data: Dict[str, Any], 
                                  output_dir: str = "./visualizations/") -> Dict[str, str]:
        """
        Create a comprehensive set of visualizations for a complete report
        
        Generates all relevant charts and saves them to files for inclusion
        in reports or presentations.
        
        Args:
            all_data: Complete dataset from recommendation system
            output_dir: Directory to save all visualizations
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        import os
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"📊 Creating comprehensive visualization report in {output_dir}")
            
            results = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. SHAP Waterfall Chart
            shap_data = all_data.get('shap_explanation')
            if shap_data:
                shap_path = os.path.join(output_dir, f"shap_explanation_{timestamp}.png")
                result = self.create_shap_waterfall_chart(shap_data, shap_path)
                if "saved" in result:
                    results['shap_waterfall'] = shap_path
            
            # 2. Portfolio Composition
            portfolio_data = all_data.get('portfolio', {})
            if portfolio_data:
                composition_path = os.path.join(output_dir, f"portfolio_composition_{timestamp}.png")
                result = self.create_portfolio_composition_chart(
                    portfolio_data.get('stocks', []),
                    portfolio_data.get('weights', {}),
                    composition_path
                )
                if "saved" in result:
                    results['portfolio_composition'] = composition_path
            
            # 3. Risk-Return Analysis
            metrics = all_data.get('portfolio_metrics')
            if metrics:
                risk_return_path = os.path.join(output_dir, f"risk_return_{timestamp}.png")
                result = self.create_risk_return_scatter(metrics, None, risk_return_path)
                if "saved" in result:
                    results['risk_return'] = risk_return_path
            
            # 4. Market Regime Analysis
            market_data = all_data.get('market_regime')
            if market_data:
                market_path = os.path.join(output_dir, f"market_regime_{timestamp}.png")
                result = self.create_market_regime_visualization(market_data, market_path)
                if "saved" in result:
                    results['market_regime'] = market_path
            
            # 5. Interactive Dashboard
            if PLOTLY_AVAILABLE:
                dashboard_path = os.path.join(output_dir, f"interactive_dashboard_{timestamp}.html")
                result = self.create_interactive_dashboard(all_data, dashboard_path)
                if "saved" in result:
                    results['interactive_dashboard'] = dashboard_path
            
            logger.info(f"✅ Comprehensive report created with {len(results)} visualizations")
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive report creation failed: {e}")
            return {}