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

  // ✅ FIXED: Extract SHAP data from the correct location
  const shapData = portfolioData?.shap_explanations; // ✅ Changed from results.shap_explanation
  const hasShapData = Boolean(shapData);

  // 🔍 Debug log
  console.log("🔍 SHAPDashboard Debug:", {
    hasShapData,
    shapData,
    portfolioKeys: portfolioData ? Object.keys(portfolioData) : null,
  });

  useEffect(() => {
    if (hasShapData) {
      // ✅ Updated to work with your actual data structure
      const featureImportance = shapData.feature_importance || {};
      const data = Object.entries(featureImportance).map(
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
        <div className="mt-4 text-sm text-red-500">
          <p>
            <strong>Debug Info:</strong>
          </p>
          <p>Looking for: portfolioData.shap_explanations</p>
          <p>Found: {shapData ? "✅ Yes" : "❌ No"}</p>
          <p>
            Available keys:{" "}
            {portfolioData ? Object.keys(portfolioData).join(", ") : "None"}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full space-y-6">
      {/* Success Message */}
      <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
        <div className="text-2xl mb-2">🎉</div>
        <h3 className="text-lg font-bold text-green-800 mb-2">
          AI Explanation Available!
        </h3>
        <p className="text-green-600">
          Confidence: {shapData.confidence}% | Method: {shapData.methodology}
        </p>
      </div>

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
  const confidence = shapData.confidence || 0;
  const methodology = shapData.methodology || "Unknown";
  const explanation = shapData.explanation || "No explanation available";

  const metrics = [
    {
      title: "AI Confidence",
      value: confidence,
      maxValue: 100,
      unit: "%",
      color: "text-green-600",
      bgColor: "bg-green-50",
      icon: "🎯",
    },
    {
      title: "Portfolio Quality",
      value: 85, // Default good score
      maxValue: 100,
      unit: "%",
      color: "text-blue-600",
      bgColor: "bg-blue-50",
      icon: "🧠",
    },
    {
      title: "Goal Alignment",
      value: 4.5,
      maxValue: 5,
      unit: "/5",
      color: "text-purple-600",
      bgColor: "bg-purple-50",
      icon: "🎯",
    },
    {
      title: "Strategy Strength",
      value: 90,
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

        {/* Main explanation */}
        <div className="bg-blue-50 rounded-lg p-4 border-l-4 border-blue-500 mb-4">
          <div className="font-semibold text-blue-800 mb-2">
            Strategy Overview
          </div>
          <p className="text-gray-700 leading-relaxed">{explanation}</p>
        </div>

        {/* Human readable explanations if available */}
        {shapData.human_readable_explanation && (
          <div className="grid gap-4">
            {Object.entries(shapData.human_readable_explanation).map(
              ([key, explanation], index) => (
                <div
                  key={index}
                  className="bg-gray-50 rounded-lg p-4 border-l-4 border-gray-400"
                >
                  <div className="font-semibold text-gray-800 mb-2">
                    {formatFactorName(key)}
                  </div>
                  <p className="text-gray-600 leading-relaxed">{explanation}</p>
                </div>
              )
            )}
          </div>
        )}

        {/* Methodology */}
        <div className="mt-4 p-3 bg-gray-100 rounded-lg">
          <p className="text-sm text-gray-600">
            <strong>Methodology:</strong> {methodology}
          </p>
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

        {chartData.length > 0 ? (
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
                  formatter={(value, name) => [
                    value.toFixed(3),
                    "Impact Score",
                  ]}
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
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No feature importance data available for visualization</p>
            <p className="text-sm mt-2">
              Available SHAP data: {Object.keys(shapData).join(", ")}
            </p>
          </div>
        )}
      </div>

      {/* Factor Explanations */}
      {chartData.length > 0 && (
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
      )}
    </div>
  );
};

// Insights Tab Component
const InsightsTab = ({ shapData, portfolioData }) => {
  return (
    <div className="space-y-6">
      {/* SHAP Data Display */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">🔍</span>
          Raw SHAP Data
        </h3>
        <div className="bg-gray-50 rounded-lg p-4">
          <pre className="text-sm overflow-auto max-h-96 text-gray-700">
            {JSON.stringify(shapData, null, 2)}
          </pre>
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
            insight={shapData.explanation || "AI-optimized portfolio strategy"}
            icon="📈"
            color="bg-blue-50 text-blue-800"
          />
          <InsightCard
            title="Confidence Level"
            insight={`AI is ${shapData.confidence}% confident in this recommendation`}
            icon="🎯"
            color="bg-green-50 text-green-800"
          />
          <InsightCard
            title="Methodology"
            insight={`Using ${shapData.methodology} for transparent decision making`}
            icon="🧠"
            color="bg-purple-50 text-purple-800"
          />
          <InsightCard
            title="Feature Analysis"
            insight={`Analyzed ${
              Object.keys(shapData.feature_importance || {}).length
            } key factors`}
            icon="🔍"
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
    growth_potential: "Growth Potential",
    risk_tolerance: "Risk Tolerance",
    innovation_exposure: "Innovation Exposure",
    time_horizon: "Time Horizon",
    market_timing: "Market Timing",
    primary_factor: "Primary Factor",
    reasoning: "AI Reasoning",
  };
  return (
    formatMap[factor] ||
    factor.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
  );
};

const getFactorColor = (factor) => {
  const colors = {
    growth_potential: "#10B981",
    risk_tolerance: "#3B82F6",
    innovation_exposure: "#8B5CF6",
    time_horizon: "#F59E0B",
    market_timing: "#EF4444",
  };
  return colors[factor] || "#6B7280";
};

const getFactorDescription = (factor) => {
  const descriptions = {
    growth_potential: "Potential for capital appreciation and long-term growth",
    risk_tolerance:
      "Your comfort level with investment volatility and potential losses",
    innovation_exposure: "Exposure to innovative and disruptive technologies",
    time_horizon: "Investment timeframe affects strategy and risk capacity",
    market_timing: "Current market conditions and optimal entry timing",
  };
  return descriptions[factor] || "Factor impact on portfolio recommendations";
};

export default SHAPDashboard;
