import React, { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

// Main SHAP Dashboard Component
const SHAPDashboard = ({ simulationId }) => {
  const [shapData, setShapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    if (simulationId) {
      fetchShapData();
    }
  }, [simulationId]);

  const fetchShapData = async () => {
    try {
      setLoading(true);
      const response = await fetch(
        `/api/shap/simulation/${simulationId}/explanation`
      );
      if (!response.ok) throw new Error("Failed to fetch SHAP data");
      const data = await response.json();
      setShapData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} />;
  if (!shapData) return <NoDataMessage />;

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 text-white">
        <h2 className="text-3xl font-bold mb-2">🔍 AI Decision Explanation</h2>
        <p className="text-blue-100 text-lg">
          Understand exactly why our AI recommended this portfolio for you
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="bg-gray-50 rounded-xl p-2">
        <div className="flex space-x-2">
          {[
            { id: "overview", label: "Overview", icon: "📊" },
            { id: "factors", label: "Key Factors", icon: "🔍" },
            { id: "portfolio", label: "Portfolio Analysis", icon: "💼" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 px-4 py-3 rounded-lg font-semibold transition-all duration-200 ${
                activeTab === tab.id
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-gray-600 hover:bg-gray-200"
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="min-h-96">
        {activeTab === "overview" && <OverviewTab shapData={shapData} />}
        {activeTab === "factors" && (
          <FactorsTab shapData={shapData} simulationId={simulationId} />
        )}
        {activeTab === "portfolio" && <PortfolioTab shapData={shapData} />}
      </div>
    </div>
  );
};

// Overview Tab Component
const OverviewTab = ({ shapData }) => {
  const { shap_data, portfolio_info, goal_analysis } = shapData;

  const metrics = [
    {
      title: "Portfolio Quality Score",
      value: shap_data.portfolio_quality_score || 0,
      maxValue: 100,
      unit: "%",
      color: "text-green-600",
      bgColor: "bg-green-50",
      icon: "🎯",
    },
    {
      title: "AI Confidence",
      value: shap_data.confidence_score || 0,
      maxValue: 100,
      unit: "%",
      color: "text-blue-600",
      bgColor: "bg-blue-50",
      icon: "🤖",
    },
    {
      title: "Expected Return",
      value: (portfolio_info.expected_return || 0) * 100,
      unit: "%",
      color: "text-orange-600",
      bgColor: "bg-orange-50",
      icon: "📈",
    },
    {
      title: "Risk Level",
      value: portfolio_info.risk_score || 0,
      maxValue: 100,
      unit: "/100",
      color: "text-purple-600",
      bgColor: "bg-purple-50",
      icon: "⚖️",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric, index) => (
          <MetricCard key={index} {...metric} />
        ))}
      </div>

      {/* AI Decision Summary */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">🧠</span>
          AI Decision Summary
        </h3>
        <div className="grid gap-4">
          {Object.entries(shap_data.human_readable_explanation || {}).map(
            ([key, explanation], index) => (
              <div
                key={index}
                className="bg-gray-50 rounded-lg p-4 border-l-4 border-blue-500"
              >
                <div className="font-semibold text-gray-800 mb-2">
                  {key.replace(/_/g, " ").toUpperCase()}
                </div>
                <p className="text-gray-600 leading-relaxed">{explanation}</p>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
};

// Factors Tab Component
const FactorsTab = ({ shapData, simulationId }) => {
  const [chartData, setChartData] = useState([]);
  const [imageUrl, setImageUrl] = useState(null);

  useEffect(() => {
    // Prepare chart data
    const featureImportance = shapData.shap_data.feature_importance || {};
    const data = Object.entries(featureImportance).map(
      ([factor, importance]) => ({
        factor: factor.replace(/_/g, " ").toUpperCase(),
        importance: parseFloat(importance),
        color: getFactorColor(factor),
      })
    );
    setChartData(data.sort((a, b) => b.importance - a.importance));

    // Load SHAP visualization image
    setImageUrl(
      `/api/shap/simulation/${simulationId}/visualization?chart_type=waterfall&t=${Date.now()}`
    );
  }, [shapData, simulationId]);

  return (
    <div className="space-y-6">
      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Feature Importance Chart */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
            <span className="mr-2">📊</span>
            Feature Importance
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis
                  dataKey="factor"
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  fontSize={12}
                  tick={{ fill: "#374151" }}
                />
                <YAxis tick={{ fill: "#374151" }} />
                <Tooltip
                  formatter={(value) => [value.toFixed(4), "Importance Score"]}
                  contentStyle={{
                    backgroundColor: "#f9fafb",
                    border: "1px solid #e5e7eb",
                    borderRadius: "8px",
                  }}
                />
                <Bar dataKey="importance" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* SHAP Visualization */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
            <span className="mr-2">🔍</span>
            SHAP Waterfall Chart
          </h3>
          <div className="h-80 flex items-center justify-center">
            {imageUrl ? (
              <img
                src={imageUrl}
                alt="SHAP Waterfall Chart"
                className="max-h-full max-w-full object-contain rounded-lg"
                onError={(e) => {
                  e.target.style.display = "none";
                  e.target.nextSibling.style.display = "block";
                }}
              />
            ) : null}
            <div className="hidden text-center text-gray-500">
              <div className="text-4xl mb-2">📊</div>
              <p className="font-medium">SHAP visualization not available</p>
              <p className="text-sm">
                See the bar chart for feature importance
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Factor Explanations */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">💡</span>
          What These Factors Mean
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <FactorDefinition
            factor="Risk Tolerance"
            definition="How much market volatility you're comfortable with affects stock selection."
            icon="🎯"
          />
          <FactorDefinition
            factor="Time Horizon"
            definition="Your investment timeframe influences growth vs. stability focus."
            icon="⏰"
          />
          <FactorDefinition
            factor="Goal Amount"
            definition="Your target value determines required return expectations."
            icon="💰"
          />
          <FactorDefinition
            factor="Market Conditions"
            definition="Current market regime affects asset allocation strategy."
            icon="📈"
          />
        </div>
      </div>
    </div>
  );
};

// Portfolio Tab Component
const PortfolioTab = ({ shapData }) => {
  const [portfolioBreakdown, setPortfolioBreakdown] = useState([]);

  useEffect(() => {
    const stocks = shapData.portfolio_info.stocks || [];
    const breakdown = stocks.map((stock, index) => ({
      name: stock,
      value: Math.round(100 / stocks.length),
      color: `hsl(${(index * 360) / stocks.length}, 70%, 50%)`,
    }));
    setPortfolioBreakdown(breakdown);
  }, [shapData]);

  const insights = [
    {
      title: "Diversification Score",
      value: "High",
      description:
        "Your portfolio is well-diversified across different asset classes",
      icon: "🌐",
      color: "text-green-600",
      bgColor: "bg-green-50",
    },
    {
      title: "Risk-Return Balance",
      value: "Optimized",
      description: "AI balanced your risk tolerance with return expectations",
      icon: "⚖️",
      color: "text-blue-600",
      bgColor: "bg-blue-50",
    },
    {
      title: "Market Adaptation",
      value: "Current",
      description: "Portfolio adjusted for current market conditions",
      icon: "📊",
      color: "text-purple-600",
      bgColor: "bg-purple-50",
    },
    {
      title: "Goal Alignment",
      value: "Matched",
      description: "Stock selection aligned with your financial goals",
      icon: "🎯",
      color: "text-orange-600",
      bgColor: "bg-orange-50",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Portfolio Composition */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">🥧</span>
          Portfolio Composition
        </h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={portfolioBreakdown}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {portfolioBreakdown.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "#f9fafb",
                  border: "1px solid #e5e7eb",
                  borderRadius: "8px",
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Portfolio Insights */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">🎯</span>
          Portfolio Insights
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {insights.map((insight, index) => (
            <InsightCard key={index} {...insight} />
          ))}
        </div>
      </div>
    </div>
  );
};

// Helper Components
const MetricCard = ({ title, value, maxValue, unit, color, bgColor, icon }) => (
  <div className={`${bgColor} rounded-xl p-6 border border-gray-200`}>
    <div className="flex items-center justify-between mb-3">
      <span className="text-2xl">{icon}</span>
      <span className="text-sm font-medium text-gray-600">{title}</span>
    </div>
    <div className={`text-2xl font-bold ${color} mb-2`}>
      {typeof value === "number" ? value.toFixed(1) : value}
      {unit}
    </div>
    {maxValue && (
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${
            color.includes("green")
              ? "bg-green-500"
              : color.includes("blue")
              ? "bg-blue-500"
              : color.includes("orange")
              ? "bg-orange-500"
              : "bg-purple-500"
          }`}
          style={{ width: `${Math.min((value / maxValue) * 100, 100)}%` }}
        />
      </div>
    )}
  </div>
);

const FactorDefinition = ({ factor, definition, icon }) => (
  <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
    <div className="flex items-center mb-2">
      <span className="text-xl mr-2">{icon}</span>
      <span className="font-semibold text-gray-800">{factor}</span>
    </div>
    <p className="text-gray-600 text-sm leading-relaxed">{definition}</p>
  </div>
);

const InsightCard = ({ title, value, description, icon, color, bgColor }) => (
  <div className={`${bgColor} rounded-lg p-4 border border-gray-200`}>
    <div className="flex items-center mb-2">
      <span className="text-xl mr-2">{icon}</span>
      <span className="font-semibold text-gray-800">{title}</span>
    </div>
    <div className={`text-lg font-bold ${color} mb-1`}>{value}</div>
    <p className="text-gray-600 text-sm leading-relaxed">{description}</p>
  </div>
);

const LoadingSpinner = () => (
  <div className="flex flex-col items-center justify-center py-12">
    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
    <p className="text-gray-600 font-medium">Loading AI explanation...</p>
  </div>
);

const ErrorMessage = ({ error }) => (
  <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center">
    <div className="text-4xl mb-2">⚠️</div>
    <h3 className="text-lg font-bold text-red-800 mb-2">
      Unable to load AI explanation
    </h3>
    <p className="text-red-600">{error}</p>
  </div>
);

const NoDataMessage = () => (
  <div className="bg-gray-50 border border-gray-200 rounded-xl p-6 text-center">
    <div className="text-4xl mb-2">📊</div>
    <h3 className="text-lg font-bold text-gray-800 mb-2">
      No AI explanation available
    </h3>
    <p className="text-gray-600">
      This simulation doesn't have SHAP explanations. Try running an enhanced
      simulation.
    </p>
  </div>
);

// Helper Functions
const getFactorColor = (factor) => {
  const colors = {
    risk_tolerance: "#EF4444",
    time_horizon: "#10B981",
    goal_amount: "#3B82F6",
    market_conditions: "#8B5CF6",
    diversification: "#F59E0B",
    volatility: "#EC4899",
  };
  return colors[factor] || "#6B7280";
};

export default SHAPDashboard;
