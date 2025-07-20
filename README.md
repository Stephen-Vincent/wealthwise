# 🚀 WealthWise - AI-Powered Investment Planning API

**WealthWise** is an intelligent personal finance and investment planning platform built with FastAPI. It leverages advanced AI techniques including machine learning, market regime detection, and correlation analysis to provide personalized investment recommendations and portfolio optimization.

## 🌟 What WealthWise Does

WealthWise helps users make informed investment decisions through:

### 🎯 **Core Features**

- **AI-Enhanced Risk Assessment**: Intelligent risk profiling based on user goals and preferences
- **Goal-Oriented Portfolio Creation**: Personalized investment strategies tailored to specific financial objectives
- **Market-Adaptive Recommendations**: Dynamic portfolio adjustments based on real-time market conditions
- **Advanced Portfolio Simulation**: Monte Carlo simulations showing potential investment outcomes
- **Multi-Factor Stock Analysis**: AI-powered selection using momentum, quality, volatility, and value factors

### 🤖 **AI Technologies Used**

- **Machine Learning**: Random Forest models for pattern recognition and prediction
- **Market Regime Detection**: Time series analysis to identify bull/bear/sideways markets
- **Factor Analysis**: Multi-dimensional stock evaluation (momentum, quality, value, volatility)
- **Correlation Optimization**: Modern Portfolio Theory for enhanced diversification
- **Predictive Analytics**: Goal achievement probability assessment

### 💼 **Investment Capabilities**

- **Risk-Based Asset Allocation**: From ultra-conservative to ultra-aggressive strategies
- **ETF-Focused Portfolios**: Professional-grade diversification through Exchange-Traded Funds
- **Dynamic Risk Adjustment**: AI automatically adjusts portfolios based on market conditions
- **Goal Feasibility Analysis**: Realistic assessment of target achievement probability

---

## 🛠️ Technical Stack

- **Backend**: FastAPI (Python)
- **AI/ML**: scikit-learn, pandas, numpy
- **Market Data**: yfinance (Yahoo Finance API)
- **Database**: SQLAlchemy with SQLite
- **Authentication**: Secure user management
- **Deployment**: Production-ready with comprehensive logging

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- **Git** (for cloning the repository)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/wealthwise.git
   cd wealthwise
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables** (Optional)

   ```bash
   # Create .env file if needed for API keys or configuration
   echo "DATABASE_URL=sqlite:///./wealthwise.db" > .env
   ```

5. **Initialize the AI Model**
   ```bash
   # Train the AI stock recommendation model
   python ai_models/stock_model/train_model.py
   ```

### Running the Application

1. **Start the Development Server**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the Application**
   - **API Documentation**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/
   - **Alternative Docs**: http://localhost:8000/redoc

### Production Deployment

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 📚 API Documentation

### 🔐 Authentication Endpoints

#### `POST /auth/signup`

Create a new user account.

```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "secure_password"
}
```

#### `POST /auth/login`

Authenticate existing user.

```json
{
  "email": "john@example.com",
  "password": "secure_password"
}
```

### 🎯 Onboarding & Risk Assessment

#### `POST /onboarding`

Submit user financial profile and receive AI-calculated risk score.

```json
{
  "experience": 3,
  "goal": "Save for retirement",
  "lumpSum": 10000,
  "monthly": 500,
  "timeframe": "10+ years",
  "consent": true,
  "user_id": 1,
  "income_bracket": "High (£70k+)",
  "target_achieved": false,
  "name": "John Doe"
}
```

**Response:**

```json
{
  "id": 5,
  "risk": "Moderate Aggressive",
  "risk_score": 65
}
```

### 🤖 AI-Powered Recommendations

#### `POST /recommend-stocks`

Get AI-enhanced portfolio recommendations based on risk profile.

```json
{
  "risk_score": 65,
  "timeframe": 10,
  "target_value": 100000,
  "current_investment": 10000,
  "monthly_contribution": 500
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "ticker": "VTI",
      "name": "Vanguard Total Stock Market ETF",
      "allocation": 0.25,
      "risk_level": "Medium"
    },
    {
      "ticker": "QQQ",
      "name": "Invesco QQQ Trust",
      "allocation": 0.2,
      "risk_level": "High"
    }
  ],
  "portfolio_metrics": {
    "expected_return": 0.114,
    "volatility": 0.18,
    "sharpe_ratio": 0.63
  },
  "market_regime": "strong_bull",
  "goal_feasibility": 85
}
```

### 📈 Portfolio Simulation

#### `POST /simulate-portfolio`

Generate Monte Carlo simulation for investment outcomes.

```json
{
  "id": 5 // onboarding submission ID
}
```

**Response:**

```json
{
  "risk": "Moderate Aggressive",
  "target_value": 100000,
  "simulation_id": 12,
  "portfolio": {
    "allocation": {
      "VTI": 0.25,
      "QQQ": 0.2,
      "VEA": 0.25,
      "VWO": 0.15,
      "BND": 0.15
    },
    "expected_return": 0.114,
    "volatility": 0.18
  },
  "timeline": [
    {
      "year": 1,
      "median_value": 15500,
      "lower_bound": 12000,
      "upper_bound": 19000
    },
    {
      "year": 5,
      "median_value": 45000,
      "lower_bound": 32000,
      "upper_bound": 62000
    },
    {
      "year": 10,
      "median_value": 105000,
      "lower_bound": 68000,
      "upper_bound": 165000
    }
  ],
  "success_probability": 85
}
```

### 📊 User Data Management

#### `GET /users/{user_id}/simulations`

Retrieve all simulations for a specific user.

#### `GET /simulations/{simulation_id}`

Get detailed simulation results.

#### `DELETE /simulations/{simulation_id}`

Remove a simulation.

#### `POST /simulations`

Save simulation results to database.

### 🔧 Utility Endpoints

#### `GET /stock-name-map`

Get mapping of stock tickers to company names.

#### `GET /`

Health check endpoint.

#### `DELETE /clear-database`

⚠️ **Development only** - Clear all data.

---

## 🧠 AI Model Architecture

### Stock Recommendation Engine

The AI system uses multiple sophisticated techniques:

1. **Market Regime Detection**

   - Analyzes VIX volatility, price trends, and momentum
   - Classifies markets as: strong_bull, bull, bear, high_volatility, low_volatility, sideways
   - Confidence scoring for regime predictions

2. **Multi-Factor Analysis**

   - **Momentum**: 6-month and 12-month price momentum
   - **Quality**: Return consistency and volume stability
   - **Volatility**: Risk-adjusted performance metrics
   - **Value**: P/E and P/B ratio analysis
   - **Size**: Market capitalization effects
   - **Technical**: RSI and moving average signals

3. **Correlation Optimization**

   - Modern Portfolio Theory implementation
   - Correlation matrix analysis for diversification
   - Dynamic weight optimization based on asset correlations

4. **Goal-Oriented Optimization**
   - Required return calculations for target achievement
   - Feasibility assessment with confidence intervals
   - Dynamic allocation adjustment based on goal requirements

### Risk Categories

- **Ultra Conservative** (0-15): Capital preservation focus
- **Conservative** (15-30): Income and modest growth
- **Moderate** (30-50): Balanced growth and income
- **Moderate Aggressive** (50-70): Growth focus with stability
- **Aggressive** (70-85): High growth seeking
- **Ultra Aggressive** (85-100): Maximum growth potential

---

## 🧪 Testing & Development

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest

# Run with coverage
pytest --cov=app tests/
```

### Debug AI Models

```bash
# Test AI recommendation system
python ai_models/stock_model/debug_recommender.py

# Train models with custom data
python ai_models/stock_model/train_model.py
```

### API Testing

Use the interactive documentation at `/docs` or test with curl:

```bash
# Health check
curl http://localhost:8000/

# Get stock recommendations
curl -X POST "http://localhost:8000/recommend-stocks" \
     -H "Content-Type: application/json" \
     -d '{"risk_score": 60, "timeframe": 5}'
```

---

## 📁 Project Structure

```
wealthwise/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── models/                 # Database models
│   ├── routes/                 # API endpoints
│   └── database.py             # Database configuration
├── ai_models/
│   └── stock_model/
│       ├── enhanced_stock_recommender.py  # AI recommendation engine
│       ├── train_model.py      # Model training script
│       └── debug_recommender.py # Testing utilities
├── requirements.txt            # Python dependencies
├── README.md                  # This file
└── .env.example               # Environment variables template
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Database
DATABASE_URL=sqlite:///./wealthwise.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=True

# External APIs (if needed)
ALPHA_VANTAGE_API_KEY=your_key_here
```

### AI Model Configuration

The AI models can be configured in `ai_models/stock_model/enhanced_stock_recommender.py`:

- **Rebalancing thresholds**: Adjust portfolio drift tolerance
- **Risk categories**: Modify risk score mappings
- **Asset universes**: Update ETF selections
- **Factor weights**: Customize stock selection criteria

---

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Cloud Deployment

The application is ready for deployment on:

- **Heroku**: `Procfile` included
- **AWS Lambda**: With Mangum adapter
- **Google Cloud Run**: Container-ready
- **DigitalOcean App Platform**: Direct deployment

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Yahoo Finance** for market data API
- **scikit-learn** for machine learning capabilities
- **FastAPI** for the excellent web framework
- **Modern Portfolio Theory** research and implementations

---

**Built with ❤️ for smarter investing**
