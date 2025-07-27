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

// SHAP Dashboard that works with your existing portfolio data structure
const SHAPDashboard = ({ portfolioData }) => {
  const [activeTab, setActiveTab] = useState("overview");
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);

  // Extract SHAP data from portfolio results
  const shapData = portfolioData?.results?.shap_explanation;
  const hasShapData = Boolean(shapData);

  useEffect(() => {
    if (hasShapData) {
      // Prepare chart data from feature_contributions
      const featureContributions = shapData.feature_contributions || {};
      const data = Object.entries(featureContributions).map(
        ([factor, importance]) => ({
          factor: formatFactorName(factor),
          importance: parseFloat(importance),
          color: getFactorColor(factor),
          description: getFactorDescription(factor),
        })
      );
      setChartData(
        data.sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
      );
    }
  }, [hasShapData, shapData]);

  if (!hasShapData) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center">
        <div className="text-4xl mb-2">📊</div>
        <h3 className="text-lg font-bold text-red-800 mb-2">
          No AI explanation available
        </h3>
        <p className="text-red-600">
          This simulation doesn't have SHAP explanations.
        </p>
      </div>
    );
  }

  return (
    <div className="w-full space-y-6">
      {/* Tab Navigation */}
      <div className="bg-gray-50 rounded-xl p-2">
        <div className="flex space-x-2">
          {[
            { id: "overview", label: "Overview", icon: "📊" },
            { id: "factors", label: "Key Factors", icon: "🔍" },
            { id: "insights", label: "AI Insights", icon: "💡" },
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
        {activeTab === "overview" && (
          <OverviewTab shapData={shapData} portfolioData={portfolioData} />
        )}
        {activeTab === "factors" && (
          <FactorsTab chartData={chartData} shapData={shapData} />
        )}
        {activeTab === "insights" && (
          <InsightsTab shapData={shapData} portfolioData={portfolioData} />
        )}
      </div>
    </div>
  );
};

// Overview Tab Component
const OverviewTab = ({ shapData, portfolioData }) => {
  const portfolioQuality = shapData.portfolio_quality_score || 0;
  const transparencyMetrics = shapData.transparency_metrics || {};
  const goalAnalysis = portfolioData?.results?.goal_analysis || {};
  const marketRegime = portfolioData?.results?.market_regime || {};

  const metrics = [
    {
      title: "Portfolio Quality Score",
      value: portfolioQuality,
      maxValue: 100,
      unit: "/100",
      color: "text-green-600",
      bgColor: "bg-green-50",
      icon: "🎯",
    },
    {
      title: "Explanation Strength",
      value: transparencyMetrics.explanation_strength || 0,
      unit: "%",
      color: "text-blue-600",
      bgColor: "bg-blue-50",
      icon: "🧠",
    },
    {
      title: "Goal Feasibility",
      value: goalAnalysis.feasibility_rating || 0,
      maxValue: 5,
      unit: "/5",
      color: "text-purple-600",
      bgColor: "bg-purple-50",
      icon: "🎯",
    },
    {
      title: "Market Confidence",
      value: (marketRegime.confidence || 0) * 100,
      maxValue: 100,
      unit: "%",
      color: "text-orange-600",
      bgColor: "bg-orange-50",
      icon: "📈",
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
          {Object.entries(shapData.human_readable_explanation || {}).map(
            ([key, explanation], index) => (
              <div
                key={index}
                className="bg-gray-50 rounded-lg p-4 border-l-4 border-blue-500"
              >
                <div className="font-semibold text-gray-800 mb-2">
                  {formatFactorName(key)}
                </div>
                <p className="text-gray-600 leading-relaxed">{explanation}</p>
              </div>
            )
          )}
        </div>
      </div>

      {/* Market Context */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">📊</span>
          Market Context
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {marketRegime.regime?.replace("_", " ").toUpperCase() || "N/A"}
            </div>
            <div className="text-sm text-gray-600">Market Regime</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {marketRegime.trend_score || 0}/5
            </div>
            <div className="text-sm text-gray-600">Trend Score</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {marketRegime.current_vix?.toFixed(1) || "N/A"}
            </div>
            <div className="text-sm text-gray-600">VIX Level</div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Factors Tab Component
const FactorsTab = ({ chartData, shapData }) => {
  return (
    <div className="space-y-6">
      {/* Feature Importance Chart */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">📊</span>
          Feature Importance Analysis
        </h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
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
                formatter={(value, name) => [value.toFixed(3), "Impact Score"]}
                labelFormatter={(label) => `Factor: ${label}`}
                contentStyle={{
                  backgroundColor: "#f9fafb",
                  border: "1px solid #e5e7eb",
                  borderRadius: "8px",
                }}
              />
              <Bar dataKey="importance" radius={[4, 4, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.importance >= 0 ? "#10B981" : "#EF4444"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Factor Explanations */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">💡</span>
          Factor Impact Analysis
        </h3>
        <div className="space-y-4">
          {chartData.map((factor, index) => (
            <div
              key={index}
              className={`rounded-lg p-4 border-l-4 ${
                factor.importance >= 0
                  ? "bg-green-50 border-green-500"
                  : "bg-red-50 border-red-500"
              }`}
            >
              <div className="flex justify-between items-start mb-2">
                <span className="font-semibold text-gray-800">
                  {factor.factor}
                </span>
                <span
                  className={`font-bold ${
                    factor.importance >= 0 ? "text-green-600" : "text-red-600"
                  }`}
                >
                  {factor.importance >= 0 ? "+" : ""}
                  {factor.importance.toFixed(3)}
                </span>
              </div>
              <p className="text-sm text-gray-600">{factor.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Insights Tab Component
const InsightsTab = ({ shapData, portfolioData }) => {
  const goalAnalysis = portfolioData?.results?.goal_analysis || {};
  const feasibilityAssessment =
    portfolioData?.results?.feasibility_assessment || {};

  return (
    <div className="space-y-6">
      {/* Goal Analysis */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">🎯</span>
          Goal Achievement Analysis
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-gray-700 mb-3">
              Required Return
            </h4>
            <div className="text-3xl font-bold text-blue-600 mb-2">
              {(goalAnalysis.required_return_percent || 0).toFixed(1)}%
            </div>
            <p className="text-sm text-gray-600">
              Annual return needed to reach your goal
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-gray-700 mb-3">
              Feasibility Score
            </h4>
            <div className="text-3xl font-bold text-green-600 mb-2">
              {feasibilityAssessment.feasibility_score || 0}%
            </div>
            <p className="text-sm text-gray-600">
              Likelihood of achieving your goal
            </p>
          </div>
        </div>
      </div>

      {/* Key Insights */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">💡</span>
          Key AI Insights
        </h3>
        <div className="space-y-4">
          <InsightCard
            title="Portfolio Strategy"
            insight="AI selected growth-focused assets to maximize long-term potential"
            icon="📈"
            color="bg-blue-50 text-blue-800"
          />
          <InsightCard
            title="Risk Management"
            insight="Balanced approach considering your risk tolerance and timeline"
            icon="⚖️"
            color="bg-green-50 text-green-800"
          />
          <InsightCard
            title="Market Timing"
            insight="Strategy adapted for current strong bull market conditions"
            icon="🎯"
            color="bg-purple-50 text-purple-800"
          />
          <InsightCard
            title="Goal Alignment"
            insight="Portfolio specifically designed to achieve your £10,000 target"
            icon="🏆"
            color="bg-orange-50 text-orange-800"
          />
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

const InsightCard = ({ title, insight, icon, color }) => (
  <div className={`${color} rounded-lg p-4 border border-gray-200`}>
    <div className="flex items-center mb-2">
      <span className="text-xl mr-3">{icon}</span>
      <span className="font-semibold">{title}</span>
    </div>
    <p className="text-sm leading-relaxed">{insight}</p>
  </div>
);

// Helper Functions
const formatFactorName = (factor) => {
  const formatMap = {
    risk_score: "Risk Tolerance",
    target_value_log: "Target Amount",
    timeframe: "Investment Timeframe",
    required_return: "Required Return",
    monthly_contribution: "Monthly Contributions",
    market_volatility: "Market Volatility",
    market_trend_score: "Market Trend",
  };
  return (
    formatMap[factor] ||
    factor.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
  );
};

const getFactorColor = (factor) => {
  const colors = {
    risk_score: "#EF4444",
    target_value_log: "#3B82F6",
    timeframe: "#10B981",
    required_return: "#8B5CF6",
    monthly_contribution: "#F59E0B",
    market_volatility: "#EC4899",
    market_trend_score: "#06B6D4",
  };
  return colors[factor] || "#6B7280";
};

const getFactorDescription = (factor) => {
  const descriptions = {
    risk_score: "Your risk tolerance affects asset allocation strategy",
    target_value_log: "Target amount influences growth requirements",
    timeframe: "Investment period affects strategy and risk capacity",
    required_return: "Return needed determines portfolio aggressiveness",
    monthly_contribution: "Regular contributions impact growth potential",
    market_volatility: "Current market conditions affect timing",
    market_trend_score: "Market momentum influences asset selection",
  };
  return descriptions[factor] || "Factor impact on portfolio recommendations";
};

export default SHAPDashboard;
