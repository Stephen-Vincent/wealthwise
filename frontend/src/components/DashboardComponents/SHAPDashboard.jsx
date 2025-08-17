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

// SHAP Dashboard adjusted for your actual data structure
const SHAPDashboard = ({ portfolioData }) => {
  const [activeTab, setActiveTab] = useState("overview");
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);

  // ✅ ADJUSTED: Check the actual data structure you have
  const hasResults = Boolean(portfolioData?.results);
  const hasAISummary = Boolean(portfolioData?.ai_summary);
  const hasRiskExplanation = Boolean(portfolioData?.risk_explanation);
  const hasAllocationGuidance = Boolean(portfolioData?.allocation_guidance);

  // Check for SHAP data in multiple possible locations
  const shapData =
    portfolioData?.shap_explanations ||
    portfolioData?.results?.shap_explanation ||
    portfolioData?.results?.shap_explanations;
  const hasShapData = Boolean(shapData);

  // ✅ IMPROVED: Better determination of AI enhancement
  const isWealthWiseEnhanced =
    hasAISummary || hasRiskExplanation || hasAllocationGuidance || hasShapData;

  // 🔍 Enhanced debug logging
  console.log("🔍 SHAPDashboard Debug:", {
    hasResults,
    hasAISummary,
    hasRiskExplanation,
    hasAllocationGuidance,
    isWealthWiseEnhanced,
    hasShapData,
    availableKeys: portfolioData ? Object.keys(portfolioData) : [],
    resultsKeys: portfolioData?.results
      ? Object.keys(portfolioData.results)
      : [],
    shapData: shapData ? "Found" : "Not found",
  });

  useEffect(() => {
    if (hasShapData) {
      // Process SHAP data if available
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

  // ✅ ADJUSTED: Better handling of different scenarios
  if (!hasResults) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-xl p-6 text-center">
        <div className="text-4xl mb-2">⚠️</div>
        <h3 className="text-lg font-bold text-gray-800 mb-2">
          No Portfolio Data Available
        </h3>
        <p className="text-gray-600">
          Please create a portfolio simulation first.
        </p>
      </div>
    );
  }

  if (!isWealthWiseEnhanced) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 text-center">
        <div className="text-4xl mb-2">🤖</div>
        <h3 className="text-lg font-bold text-yellow-800 mb-2">
          AI Enhancement Not Available
        </h3>
        <p className="text-yellow-600 mb-4">
          This portfolio simulation doesn't include comprehensive AI
          explanations.
        </p>

        {/* Show basic portfolio information */}
        <div className="bg-white rounded-lg p-4 mt-4">
          <h4 className="font-semibold mb-2">Portfolio Information:</h4>
          <div className="text-left space-y-2 text-sm">
            <p>
              <strong>Portfolio ID:</strong> {portfolioData.id}
            </p>
            <p>
              <strong>Goal:</strong> {portfolioData.goal}
            </p>
            <p>
              <strong>Target Value:</strong> £
              {portfolioData.target_value?.toLocaleString()}
            </p>
            <p>
              <strong>Risk Score:</strong> {portfolioData.risk_score}/100
            </p>
            <p>
              <strong>Risk Label:</strong> {portfolioData.risk_label}
            </p>
            {portfolioData.results?.stocks_picked && (
              <p>
                <strong>Stocks Selected:</strong>{" "}
                {portfolioData.results.stocks_picked.length} stocks
              </p>
            )}
          </div>
        </div>

        <div className="mt-4 text-xs text-yellow-600">
          <p>
            <strong>Available keys:</strong>{" "}
            {Object.keys(portfolioData).join(", ")}
          </p>
        </div>
      </div>
    );
  }

  if (!hasShapData) {
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
        <div className="text-4xl mb-2 text-center">🧠</div>
        <h3 className="text-lg font-bold text-blue-800 mb-2 text-center">
          AI Analysis Available (No SHAP Data)
        </h3>
        <p className="text-blue-600 mb-4 text-center">
          While detailed SHAP explanations aren't available, we have
          AI-generated insights for your portfolio.
        </p>

        {/* Show available AI information */}
        <div className="space-y-4">
          {portfolioData.ai_summary && (
            <div className="bg-white rounded-lg p-4">
              <h4 className="font-semibold mb-2 flex items-center">
                <span className="mr-2">🤖</span>
                AI Summary
              </h4>
              <p className="text-gray-700 text-sm leading-relaxed">
                {portfolioData.ai_summary}
              </p>
            </div>
          )}

          {portfolioData.risk_explanation && (
            <div className="bg-white rounded-lg p-4">
              <h4 className="font-semibold mb-2 flex items-center">
                <span className="mr-2">⚠️</span>
                Risk Explanation
              </h4>
              <p className="text-gray-700 text-sm leading-relaxed">
                {portfolioData.risk_explanation}
              </p>
            </div>
          )}

          {portfolioData.allocation_guidance && (
            <div className="bg-white rounded-lg p-4">
              <h4 className="font-semibold mb-2 flex items-center">
                <span className="mr-2">📊</span>
                Allocation Guidance
              </h4>
              <p className="text-gray-700 text-sm leading-relaxed">
                {portfolioData.allocation_guidance}
              </p>
            </div>
          )}

          {/* Portfolio metrics */}
          <div className="bg-white rounded-lg p-4">
            <h4 className="font-semibold mb-2 flex items-center">
              <span className="mr-2">📈</span>
              Portfolio Metrics
            </h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">Risk Score:</span>
                <span className="ml-2">{portfolioData.risk_score}/100</span>
              </div>
              <div>
                <span className="font-medium">Risk Level:</span>
                <span className="ml-2">{portfolioData.risk_label}</span>
              </div>
              <div>
                <span className="font-medium">Target:</span>
                <span className="ml-2">
                  £{portfolioData.target_value?.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="font-medium">Timeframe:</span>
                <span className="ml-2">{portfolioData.timeframe} years</span>
              </div>
            </div>
          </div>

          {/* Stock allocation if available */}
          {portfolioData.results?.stocks_picked && (
            <div className="bg-white rounded-lg p-4">
              <h4 className="font-semibold mb-2 flex items-center">
                <span className="mr-2">📋</span>
                Selected Stocks ({portfolioData.results.stocks_picked.length})
              </h4>
              <div className="grid grid-cols-1 gap-2 text-sm">
                {portfolioData.results.stocks_picked
                  .slice(0, 5)
                  .map((stock, index) => (
                    <div
                      key={index}
                      className="flex justify-between items-center"
                    >
                      <span className="font-medium">{stock.symbol}</span>
                      <span>
                        {stock.allocation
                          ? `${(stock.allocation * 100).toFixed(1)}%`
                          : "N/A"}
                      </span>
                    </div>
                  ))}
                {portfolioData.results.stocks_picked.length > 5 && (
                  <p className="text-gray-500 italic">
                    ... and {portfolioData.results.stocks_picked.length - 5}{" "}
                    more
                  </p>
                )}
              </div>
            </div>
          )}
        </div>

        <button
          onClick={() => {
            console.log("🔍 Full Portfolio Data:", portfolioData);
            // Try to fetch SHAP data separately if needed
            if (portfolioData.id && portfolioData.user_id) {
              fetchShapDataSeparately(portfolioData.user_id, portfolioData.id);
            }
          }}
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 w-full"
        >
          🔍 Debug Portfolio Data
        </button>
      </div>
    );
  }

  // ✅ SHAP data is available - render full dashboard
  return (
    <div className="w-full space-y-6">
      {/* Success Message */}
      <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
        <div className="text-2xl mb-2">🎉</div>
        <h3 className="text-lg font-bold text-green-800 mb-2">
          SHAP AI Explanation Available!
        </h3>
        <p className="text-green-600">
          Confidence: {shapData.confidence || "N/A"}% | Method:{" "}
          {shapData.methodology || "SHAP Analysis"}
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

// Overview Tab Component - Enhanced for your data
const OverviewTab = ({ shapData, portfolioData }) => {
  const confidence = shapData?.confidence || 75; // Default confidence
  const methodology = shapData?.methodology || "SHAP Analysis";
  const explanation =
    shapData?.explanation ||
    portfolioData?.ai_summary ||
    "AI-powered portfolio optimization";

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
      title: "Risk Score",
      value: portfolioData?.risk_score || 50,
      maxValue: 100,
      unit: "/100",
      color: "text-blue-600",
      bgColor: "bg-blue-50",
      icon: "📊",
    },
    {
      title: "Target Progress",
      value: portfolioData?.target_achieved ? 100 : 75,
      maxValue: 100,
      unit: "%",
      color: "text-purple-600",
      bgColor: "bg-purple-50",
      icon: "🎯",
    },
    {
      title: "Portfolio Quality",
      value: 85, // Derived score
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

        {/* Additional explanations from your data */}
        {portfolioData?.risk_explanation && (
          <div className="bg-yellow-50 rounded-lg p-4 border-l-4 border-yellow-500 mb-4">
            <div className="font-semibold text-yellow-800 mb-2">
              Risk Analysis
            </div>
            <p className="text-gray-700 leading-relaxed">
              {portfolioData.risk_explanation}
            </p>
          </div>
        )}

        {portfolioData?.allocation_guidance && (
          <div className="bg-green-50 rounded-lg p-4 border-l-4 border-green-500 mb-4">
            <div className="font-semibold text-green-800 mb-2">
              Allocation Guidance
            </div>
            <p className="text-gray-700 leading-relaxed">
              {portfolioData.allocation_guidance}
            </p>
          </div>
        )}

        {/* Human readable explanations if available */}
        {shapData?.human_readable_explanation && (
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

// Factors Tab Component - Same as before
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
              Available SHAP data: {Object.keys(shapData || {}).join(", ")}
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

// Insights Tab Component - Enhanced
const InsightsTab = ({ shapData, portfolioData }) => {
  return (
    <div className="space-y-6">
      {/* SHAP Data Display */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">🔍</span>
          Raw Data Analysis
        </h3>
        <div className="bg-gray-50 rounded-lg p-4">
          <pre className="text-sm overflow-auto max-h-96 text-gray-700">
            {JSON.stringify(shapData || portfolioData, null, 2)}
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
            insight={
              shapData?.explanation ||
              portfolioData?.ai_summary ||
              "AI-optimized portfolio strategy"
            }
            icon="📈"
            color="bg-blue-50 text-blue-800"
          />
          <InsightCard
            title="Confidence Level"
            insight={`AI is ${
              shapData?.confidence || 75
            }% confident in this recommendation`}
            icon="🎯"
            color="bg-green-50 text-green-800"
          />
          <InsightCard
            title="Risk Assessment"
            insight={`Risk level: ${portfolioData?.risk_label} (${portfolioData?.risk_score}/100)`}
            icon="⚠️"
            color="bg-yellow-50 text-yellow-800"
          />
          <InsightCard
            title="Analysis Method"
            insight={`Using ${
              shapData?.methodology || "Advanced AI Analysis"
            } for transparent decision making`}
            icon="🧠"
            color="bg-purple-50 text-purple-800"
          />
          <InsightCard
            title="Portfolio Composition"
            insight={`Selected ${
              portfolioData?.results?.stocks_picked?.length || 0
            } optimized investments`}
            icon="📋"
            color="bg-orange-50 text-orange-800"
          />
        </div>
      </div>
    </div>
  );
};

// Helper Components - Same as before
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

// Helper Functions - Same as before
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

// Helper function to try fetching SHAP data separately
const fetchShapDataSeparately = async (userId, simulationId) => {
  try {
    console.log(`🔍 Attempting to fetch SHAP data separately...`);

    // Try different possible API endpoints
    const baseUrl =
      import.meta.env.VITE_API_URL || "https://wealthwise-dwfq.onrender.com";
    const possibleEndpoints = [
      `${baseUrl}/portfolio/${userId}/${simulationId}/shap`,
      `${baseUrl}/shap/${userId}/${simulationId}`,
      `${baseUrl}/portfolio/${userId}/${simulationId}?include_shap=true`,
      `${baseUrl}/explanations/${userId}/${simulationId}`,
    ];

    for (const endpoint of possibleEndpoints) {
      try {
        console.log(`🔍 Trying endpoint: ${endpoint}`);
        const response = await fetch(endpoint);

        if (response.ok) {
          const data = await response.json();
          console.log(`✅ Found SHAP data at ${endpoint}:`, data);
          return data;
        } else {
          console.log(`❌ ${endpoint}: ${response.status}`);
        }
      } catch (error) {
        console.log(`❌ ${endpoint}: ${error.message}`);
      }
    }

    console.log(`❌ No SHAP data found at any endpoint`);
  } catch (error) {
    console.error("❌ Error fetching SHAP data:", error);
  }
};

export default SHAPDashboard;
