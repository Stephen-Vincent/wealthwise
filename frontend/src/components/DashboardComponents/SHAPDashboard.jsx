// Complete SHAPDashboard component using PortfolioContext
import React, { useState, useMemo } from "react";
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
  Area,
  AreaChart,
  Legend,
} from "recharts";
import { usePortfolio } from "../../context/PortfolioContext";

const SHAPDashboard = () => {
  const {
    portfolioData,
    shapData,
    hasShapData,
    chartData,
    enhancedData,
    loading,
    error,
  } = usePortfolio();

  const [activeTab, setActiveTab] = useState("summary");

  if (loading) {
    return (
      <div className="bg-white border rounded-xl p-8 text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <div className="text-lg text-gray-700">
          Loading your portfolio data...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-8 text-center">
        <div className="text-6xl mb-4">Error</div>
        <h3 className="text-xl font-bold text-red-800 mb-3">
          Couldn't load portfolio data
        </h3>
        <p className="text-red-700">{error}</p>
      </div>
    );
  }

  if (!hasShapData) {
    return (
      <div className="space-y-4">
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-8 text-center">
          <div className="text-6xl mb-4">Info</div>
          <h3 className="text-xl font-bold text-blue-800 mb-3">
            AI Explanation Not Available
          </h3>
          <p className="text-blue-700 text-lg">
            We couldn't generate an AI explanation for this portfolio
            recommendation.
          </p>
          <p className="text-blue-600 mt-2">
            This might happen with some types of investment strategies.
          </p>
        </div>

        <details className="bg-gray-50 border rounded-lg p-4">
          <summary className="cursor-pointer font-medium text-gray-700">
            Debug Information
          </summary>
          <div className="mt-3 space-y-2 text-sm">
            <p>
              <strong>Portfolio Data Available:</strong>{" "}
              {!!portfolioData ? "Yes" : "No"}
            </p>
            <p>
              <strong>Chart Data Available:</strong>{" "}
              {!!chartData ? "Yes" : "No"}
            </p>
            <p>
              <strong>Enhanced Data Available:</strong>{" "}
              {!!enhancedData ? "Yes" : "No"}
            </p>
            <p>
              <strong>Has SHAP Explanations Flag:</strong>{" "}
              {portfolioData?.has_shap_explanations ? "Yes" : "No"}
            </p>

            {portfolioData && (
              <div className="mt-3">
                <p>
                  <strong>Available Keys:</strong>
                </p>
                <code className="block bg-white p-2 rounded text-xs">
                  {Object.keys(portfolioData).join(", ")}
                </code>
              </div>
            )}

            {chartData && (
              <div className="mt-3">
                <p>
                  <strong>Chart Data Keys:</strong>
                </p>
                <code className="block bg-white p-2 rounded text-xs">
                  {Object.keys(chartData).join(", ")}
                </code>
              </div>
            )}
          </div>
        </details>
      </div>
    );
  }

  return (
    <div className="w-full space-y-6">
      <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
        <div className="text-2xl mb-2">Success</div>
        <h3 className="text-lg font-bold text-green-800 mb-2">
          SHAP AI Explanation Available!
        </h3>
        <p className="text-green-600">
          Our AI has generated detailed explanations for your portfolio
          recommendations.
        </p>
        {chartData && (
          <p className="text-green-600 text-sm mt-1">
            Interactive charts are ready to explore.
          </p>
        )}
      </div>

      <div className="bg-gray-50 rounded-xl p-2">
        <div className="flex space-x-2">
          {[
            { id: "summary", label: "Overview", icon: "Chart" },
            { id: "factors", label: "Key Factors", icon: "Search" },
            { id: "visualizations", label: "Charts", icon: "Graph" },
            { id: "insights", label: "AI Insights", icon: "Light" },
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

      <div className="min-h-96">
        {activeTab === "summary" && (
          <SummaryTab
            shapData={shapData}
            portfolioData={portfolioData}
            chartData={chartData}
          />
        )}
        {activeTab === "factors" && (
          <FactorsTab chartData={chartData} shapData={shapData} />
        )}
        {activeTab === "visualizations" && (
          <VisualizationsTab
            chartData={chartData}
            enhancedData={enhancedData}
            portfolioData={portfolioData}
          />
        )}
        {activeTab === "insights" && (
          <InsightsTab
            shapData={shapData}
            portfolioData={portfolioData}
            chartData={chartData}
          />
        )}
      </div>
    </div>
  );
};

// Summary Tab
const SummaryTab = ({ shapData, portfolioData, chartData }) => {
  const confidence = shapData?.confidence_score || shapData?.confidence || 75;
  const methodology = shapData?.methodology || "SHAP Analysis";
  const portfolioQualityScore = shapData?.portfolio_quality_score || 85;

  const metrics = [
    {
      title: "AI Confidence",
      value: confidence,
      maxValue: 100,
      unit: "%",
      color: "text-green-600",
      bgColor: "bg-green-50",
      icon: "Target",
    },
    {
      title: "Portfolio Quality",
      value: portfolioQualityScore,
      maxValue: 100,
      unit: "/100",
      color: "text-blue-600",
      bgColor: "bg-blue-50",
      icon: "Brain",
    },
    {
      title: "Risk Score",
      value: portfolioData?.risk_score || 50,
      maxValue: 100,
      unit: "/100",
      color: "text-orange-600",
      bgColor: "bg-orange-50",
      icon: "Chart",
    },
    {
      title: "Target Progress",
      value: portfolioData?.target_achieved ? 100 : 75,
      maxValue: 100,
      unit: "%",
      color: "text-purple-600",
      bgColor: "bg-purple-50",
      icon: "Target",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric, index) => (
          <MetricCard key={index} {...metric} />
        ))}
      </div>

      {chartData?.portfolio_composition && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">
            Portfolio Allocation
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData.portfolio_composition}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ symbol, allocation }) =>
                    `${symbol}: ${allocation}%`
                  }
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="allocation"
                >
                  {chartData.portfolio_composition.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`${value}%`, "Allocation"]} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4">
          AI Decision Summary
        </h3>

        {shapData?.human_readable_explanation && (
          <div className="space-y-4">
            {Object.entries(shapData.human_readable_explanation).map(
              ([key, explanation], index) => (
                <div
                  key={index}
                  className="bg-blue-50 rounded-lg p-4 border-l-4 border-blue-500"
                >
                  <div className="font-semibold text-blue-800 mb-2">
                    {formatFactorName(key)}
                  </div>
                  <p className="text-gray-700 leading-relaxed">{explanation}</p>
                </div>
              )
            )}
          </div>
        )}

        <div className="mt-4 p-3 bg-gray-100 rounded-lg">
          <p className="text-sm text-gray-600">
            <strong>Methodology:</strong> {methodology}
          </p>
        </div>
      </div>
    </div>
  );
};

// Factors Tab
const FactorsTab = ({ chartData, shapData }) => {
  const factorImportanceData = useMemo(() => {
    if (!chartData?.feature_importance) {
      const rawImportance = shapData?.feature_contributions || {};
      return Object.entries(rawImportance)
        .map(([factor, importance]) => ({
          factor: formatFactorName(factor),
          importance: Number(importance) || 0,
          isPositive: Number(importance) >= 0,
          description: getFactorDescription(factor),
        }))
        .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
    }

    return chartData.feature_importance.map((item) => ({
      ...item,
      importance: Number(item.importance) || 0,
    }));
  }, [chartData, shapData]);

  const getBarColor = (entry) => {
    return entry.importance >= 0 ? "#10B981" : "#EF4444";
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4">
          Feature Importance Analysis
        </h3>

        {factorImportanceData.length > 0 ? (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={factorImportanceData}
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
                  formatter={(value) => [
                    Number(value).toFixed(3),
                    "Impact Score",
                  ]}
                  labelFormatter={(label) => `Factor: ${label}`}
                />
                <Bar dataKey="importance" radius={[4, 4, 0, 0]}>
                  {factorImportanceData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getBarColor(entry)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No feature importance data available for visualization</p>
          </div>
        )}
      </div>

      {factorImportanceData.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">
            Factor Impact Analysis
          </h3>
          <div className="space-y-4">
            {factorImportanceData.map((factor, index) => (
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
                <p className="text-sm text-gray-600">
                  {factor.description || getFactorDescription(factor.factor)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Visualizations Tab
const VisualizationsTab = ({ chartData, enhancedData, portfolioData }) => {
  if (!chartData && !enhancedData) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-8 text-center">
        <div className="text-4xl mb-4">Warning</div>
        <h3 className="text-lg font-semibold text-yellow-800 mb-2">
          Charts Not Available
        </h3>
        <p className="text-yellow-700">
          Chart data could not be loaded. This might be due to an API issue or
          missing data.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {chartData?.goal_analysis && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">
            Goal Achievement Progress
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                £{chartData.goal_analysis.target_value?.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Target Value</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                £{chartData.goal_analysis.projected_value?.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Projected Value</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {(chartData.goal_analysis.progress_percentage || 0).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Progress</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {chartData.goal_analysis.target_achieved ? "Yes" : "No"}
              </div>
              <div className="text-sm text-gray-600">Target Achieved</div>
            </div>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div
              className="bg-blue-600 h-4 rounded-full transition-all duration-500"
              style={{
                width: `${Math.min(
                  chartData.goal_analysis.progress_percentage || 0,
                  100
                )}%`,
              }}
            />
          </div>
        </div>
      )}

      {chartData?.performance_timeline?.portfolio && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">
            Portfolio Performance Over Time
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chartData.performance_timeline.portfolio.slice(-60)}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 12 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis
                  tickFormatter={(value) => `£${(value / 1000).toFixed(0)}K`}
                />
                <Tooltip
                  formatter={(value, name) => [
                    name === "value"
                      ? `£${value.toLocaleString()}`
                      : `${value.toFixed(2)}%`,
                    name === "value" ? "Portfolio Value" : "Return %",
                  ]}
                  labelFormatter={(label) => `Date: ${label}`}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="#3B82F6"
                  fill="#3B82F6"
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
        <h4 className="font-semibold text-blue-800 mb-2">
          Visualization Summary
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-blue-600">
              {Object.keys(chartData || {}).length}
            </div>
            <div className="text-sm text-blue-700">Chart Types</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-blue-600">Interactive</div>
            <div className="text-sm text-blue-700">React Charts</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-blue-600">Real-time</div>
            <div className="text-sm text-blue-700">Data Updates</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-blue-600">AI Powered</div>
            <div className="text-sm text-blue-700">SHAP Analysis</div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Insights Tab
const InsightsTab = ({ shapData, portfolioData, chartData }) => {
  const generateInsights = () => {
    const insights = [];
    const confidence = shapData?.confidence_score || shapData?.confidence || 75;
    const portfolioQuality = shapData?.portfolio_quality_score || 85;

    insights.push({
      icon: "Target",
      title: "AI Confidence Level",
      description: `Our AI is ${confidence}% confident in this portfolio recommendation, indicating ${
        confidence >= 80
          ? "high reliability"
          : confidence >= 60
          ? "good reliability"
          : "moderate reliability"
      } in the strategy.`,
    });

    insights.push({
      icon: "Star",
      title: "Portfolio Quality Score",
      description: `Your portfolio scored ${portfolioQuality.toFixed(
        2
      )}/100 for quality, suggesting ${
        portfolioQuality >= 85
          ? "excellent diversification and risk management"
          : portfolioQuality >= 70
          ? "good balance between risk and returns"
          : "room for improvement in optimization"
      }.`,
    });

    if (
      chartData?.feature_importance &&
      chartData.feature_importance.length > 0
    ) {
      const topFactor = chartData.feature_importance[0];
      const isPositive = topFactor.importance >= 0;

      insights.push({
        icon: "TrendingUp",
        title: "Primary Decision Factor",
        description: `${topFactor.feature} had the ${
          isPositive ? "most positive" : "most challenging"
        } impact on your portfolio design with an impact score of ${topFactor.importance.toFixed(
          3
        )}.`,
      });
    }

    if (chartData?.goal_analysis) {
      const targetAchieved = chartData.goal_analysis.target_achieved;
      insights.push({
        icon: "Trophy",
        title: "Goal Achievement Outlook",
        description: `Based on your inputs and current market conditions, ${
          targetAchieved
            ? "you are on track to achieve your financial goal"
            : "you may need to adjust your strategy to reach your financial goal"
        }.`,
      });
    }

    return insights;
  };

  const insights = generateInsights();

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-6">
          Key AI Insights
        </h3>

        <div className="space-y-6">
          {insights.map((insight, index) => (
            <div
              key={index}
              className="flex items-start space-x-4 p-4 bg-blue-50 rounded-lg border border-blue-100"
            >
              <div className="text-2xl flex-shrink-0">
                {getIconForInsight(insight.icon)}
              </div>
              <div className="flex-1">
                <h4 className="font-semibold text-blue-900 mb-2 text-lg">
                  {insight.title}
                </h4>
                <p className="text-blue-800 leading-relaxed">
                  {insight.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4">
          What This Means for You
        </h3>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-green-50 rounded-lg p-4 border border-green-200">
            <div className="flex items-center mb-2">
              <span className="text-xl mr-2">Strengths</span>
            </div>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Portfolio matches your risk comfort level</li>
              <li>• Strategy aligned with your timeline</li>
              <li>• AI-optimized for current market conditions</li>
              <li>• Diversification helps manage risk</li>
            </ul>
          </div>

          <div className="bg-amber-50 rounded-lg p-4 border border-amber-200">
            <div className="flex items-center mb-2">
              <span className="text-xl mr-2">Opportunities</span>
            </div>
            <ul className="text-sm text-amber-700 space-y-1">
              <li>• Regular monitoring recommended</li>
              <li>• Consider increasing contributions if possible</li>
              <li>• Rebalance as market conditions change</li>
              <li>• Review strategy annually</li>
            </ul>
          </div>
        </div>
      </div>

      {chartData && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">
            Advanced Analytics Summary
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-800">
                {chartData.portfolio_composition?.length || 0}
              </div>
              <div className="text-sm text-gray-600">Assets in Portfolio</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-800">
                {chartData.feature_importance?.length || 0}
              </div>
              <div className="text-sm text-gray-600">Factors Analyzed</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-800">
                {chartData.shap_waterfall?.length || 0}
              </div>
              <div className="text-sm text-gray-600">SHAP Features</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper Components and Functions
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

const formatFactorName = (factor) => {
  const formatMap = {
    risk_score: "Your Risk Comfort Level",
    target_value_log: "Your Goal Amount",
    timeframe: "Your Timeline",
    required_return: "Growth You Need",
    monthly_contribution: "Your Monthly Savings",
    market_volatility: "Current Market Stability",
    market_trend_score: "Market Momentum",
    "Required Growth Rate": "Growth You Need",
    "Market Trend": "Market Momentum",
    "Time Horizon": "Your Timeline",
    "Monthly Investment": "Your Monthly Savings",
    "Risk Tolerance": "Your Risk Comfort Level",
    "Investment Goal": "Your Goal Amount",
    "Market Volatility": "Current Market Stability",
  };
  return (
    formatMap[factor] ||
    factor.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
  );
};

const getFactorDescription = (factor) => {
  const descriptions = {
    risk_score:
      "How comfortable you are with your investments going up and down affects what we recommend",
    target_value_log:
      "The amount you want to reach determines how aggressively we need to invest",
    timeframe:
      "How long you have to invest affects the types of investments we can choose",
    required_return:
      "The growth rate you need influences how much risk we take in your portfolio",
    monthly_contribution:
      "How much you save each month affects your overall investment strategy",
    market_volatility:
      "Current market conditions influence the timing and types of investments we select",
    market_trend_score:
      "Whether markets are going up or down affects our investment choices",
    "Required Growth Rate":
      "The annual growth rate needed to reach your target influences portfolio aggressiveness",
    "Market Trend":
      "Whether markets are trending up or down affects investment timing",
    "Time Horizon":
      "How long you have to invest affects the types of investments we can choose",
    "Monthly Investment":
      "Amount you can invest each month affects your overall strategy",
    "Risk Tolerance": "Your comfort level with investment risk and volatility",
    "Investment Goal": "The financial goal you want to achieve",
    "Market Volatility": "Current market uncertainty and volatility levels",
  };
  return (
    descriptions[factor] ||
    "This factor influenced how we built your investment portfolio"
  );
};

const getIconForInsight = (iconName) => {
  const icons = {
    Target: "Target",
    Star: "Star",
    TrendingUp: "Trending Up",
    Scale: "Scale",
    Waves: "Waves",
    Trophy: "Trophy",
  };
  return icons[iconName] || "Insight";
};

const COLORS = [
  "#0088FE",
  "#00C49F",
  "#FFBB28",
  "#FF8042",
  "#8884D8",
  "#82CA9D",
  "#FFC658",
  "#FF7C7C",
  "#8DD1E1",
  "#D084D0",
];

export default SHAPDashboard;
