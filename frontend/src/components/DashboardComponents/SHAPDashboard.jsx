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
  Area,
  AreaChart,
  Legend,
  Cell,
} from "recharts";
import {
  Target,
  Brain,
  BarChart3,
  TrendingUp,
  Search,
  Lightbulb,
  Star,
  Trophy,
  Activity,
} from "lucide-react";

// Mock usePortfolio hook for demonstration
const usePortfolio = () => ({
  portfolioData: {
    risk_score: 65,
    target_achieved: false,
    has_shap_explanations: true,
  },
  shapData: {
    confidence_score: 85,
    portfolio_quality_score: 88,
    methodology: "SHAP Analysis",
    feature_contributions: {
      risk_score: 0.15,
      target_value_log: -0.08,
      timeframe: 0.12,
      required_return: -0.05,
      monthly_contribution: 0.18,
      market_volatility: -0.03,
      market_trend_score: 0.09,
    },
    human_readable_explanation: {
      risk_score:
        "Your moderate risk tolerance allows for a balanced portfolio that can grow while protecting your money.",
      monthly_contribution:
        "Your regular monthly savings create a strong foundation for reaching your financial goals.",
      timeframe:
        "Having 10 years to invest gives us flexibility to choose growth-focused investments.",
    },
  },
  hasShapData: true,
  chartData: {
    feature_importance: [
      {
        factor: "Monthly Contribution",
        importance: 0.18,
        description: "Your regular savings amount",
      },
      {
        factor: "Risk Tolerance",
        importance: 0.15,
        description: "Your comfort with investment risk",
      },
      {
        factor: "Time Horizon",
        importance: 0.12,
        description: "Years until you need the money",
      },
      {
        factor: "Market Trend",
        importance: 0.09,
        description: "Current market direction",
      },
      {
        factor: "Target Value",
        importance: -0.08,
        description: "Amount you want to reach",
      },
      {
        factor: "Required Return",
        importance: -0.05,
        description: "Growth rate needed",
      },
      {
        factor: "Market Volatility",
        importance: -0.03,
        description: "Current market uncertainty",
      },
    ],
    performance_timeline: {
      portfolio: [
        { date: "2024-01", value: 10000, return: 0 },
        { date: "2024-02", value: 10200, return: 2.0 },
        { date: "2024-03", value: 10150, return: 1.5 },
        { date: "2024-04", value: 10300, return: 3.0 },
        { date: "2024-05", value: 10450, return: 4.5 },
        { date: "2024-06", value: 10400, return: 4.0 },
      ],
    },
  },
  enhancedData: {},
  loading: false,
  error: null,
});

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
        <div className="text-6xl mb-4">⚠️</div>
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
          <div className="text-6xl mb-4">ℹ️</div>
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
      </div>
    );
  }

  return (
    <div className="w-full space-y-6">
      <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
        <div className="text-2xl mb-2">✅</div>
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
            { id: "summary", label: "Overview", icon: <BarChart3 size={16} /> },
            { id: "factors", label: "Key Factors", icon: <Search size={16} /> },
            {
              id: "visualizations",
              label: "Charts",
              icon: <Activity size={16} />,
            },
            {
              id: "insights",
              label: "AI Insights",
              icon: <Lightbulb size={16} />,
            },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 px-4 py-3 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center ${
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

// Summary Tab - Only showing 4 metric boxes, no pie chart
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
      icon: <Target size={24} />,
    },
    {
      title: "Portfolio Quality",
      value: portfolioQualityScore,
      maxValue: 100,
      unit: "/100",
      color: "text-blue-600",
      bgColor: "bg-blue-50",
      icon: <Brain size={24} />,
    },
    {
      title: "Risk Score",
      value: portfolioData?.risk_score || 50,
      maxValue: 100,
      unit: "/100",
      color: "text-orange-600",
      bgColor: "bg-orange-50",
      icon: <BarChart3 size={24} />,
    },
    {
      title: "Target Progress",
      value: portfolioData?.target_achieved ? 100 : 75,
      maxValue: 100,
      unit: "%",
      color: "text-purple-600",
      bgColor: "bg-purple-50",
      icon: <Target size={24} />,
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric, index) => (
          <MetricCard key={index} {...metric} />
        ))}
      </div>

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
      </div>
    </div>
  );
};

// Factors Tab - Fixed data source issue
const FactorsTab = ({ chartData, shapData }) => {
  const factorImportanceData = useMemo(() => {
    // First try chartData.feature_importance
    if (
      chartData?.feature_importance &&
      chartData.feature_importance.length > 0
    ) {
      return chartData.feature_importance.map((item) => ({
        ...item,
        importance: Number(item.importance) || 0,
        isPositive: Number(item.importance) >= 0,
      }));
    }

    // Fallback to shapData.feature_contributions
    if (shapData?.feature_contributions) {
      return Object.entries(shapData.feature_contributions)
        .map(([factor, importance]) => ({
          factor: formatFactorName(factor),
          importance: Number(importance) || 0,
          isPositive: Number(importance) >= 0,
          description: getFactorDescription(factor),
        }))
        .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
    }

    return [];
  }, [chartData, shapData]);

  const getBarColor = (entry, index, data) => {
    const sortedData = [...data].sort(
      (a, b) => Math.abs(b.importance) - Math.abs(a.importance)
    );
    const sortedIndex = sortedData.findIndex((item) => item === entry);
    const intensity = Math.max(0.3, 1 - sortedIndex * 0.15); // Light to dark based on importance rank

    if (entry.importance >= 0) {
      // Positive values - different shades of green
      const greenShades = [
        `rgba(16, 185, 129, ${intensity})`, // emerald-500
        `rgba(34, 197, 94, ${intensity})`, // green-500
        `rgba(22, 163, 74, ${intensity})`, // green-600
        `rgba(21, 128, 61, ${intensity})`, // green-700
      ];
      return greenShades[sortedIndex % greenShades.length];
    } else {
      // Negative values - different shades of red
      const redShades = [
        `rgba(239, 68, 68, ${intensity})`, // red-500
        `rgba(220, 38, 38, ${intensity})`, // red-600
        `rgba(185, 28, 28, ${intensity})`, // red-700
        `rgba(153, 27, 27, ${intensity})`, // red-800
      ];
      return redShades[sortedIndex % redShades.length];
    }
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
                    <Cell
                      key={`cell-${index}`}
                      fill={getBarColor(entry, index, factorImportanceData)}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No feature importance data available for visualization</p>
            <p className="text-xs mt-2">
              Debug: chartData keys:{" "}
              {chartData ? Object.keys(chartData).join(", ") : "none"}
            </p>
            <p className="text-xs">
              shapData keys:{" "}
              {shapData ? Object.keys(shapData).join(", ") : "none"}
            </p>
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

// Visualizations Tab - Removed goal achievement progress section
const VisualizationsTab = ({ chartData, enhancedData, portfolioData }) => {
  if (!chartData && !enhancedData) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-8 text-center">
        <div className="text-4xl mb-4">⚠️</div>
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
    </div>
  );
};

// Insights Tab - Added proper icons
const InsightsTab = ({ shapData, portfolioData, chartData }) => {
  const generateInsights = () => {
    const insights = [];
    const confidence = shapData?.confidence_score || shapData?.confidence || 75;
    const portfolioQuality = shapData?.portfolio_quality_score || 85;

    insights.push({
      icon: <Target size={24} />,
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
      icon: <Star size={24} />,
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
        icon: <TrendingUp size={24} />,
        title: "Primary Decision Factor",
        description: `${topFactor.factor || topFactor.feature} had the ${
          isPositive ? "most positive" : "most challenging"
        } impact on your portfolio design with an impact score of ${topFactor.importance.toFixed(
          3
        )}.`,
      });
    }

    insights.push({
      icon: <Trophy size={24} />,
      title: "Investment Strategy Outlook",
      description: `Based on your inputs and current market conditions, your portfolio is designed to balance growth potential with risk management over your investment timeline.`,
    });

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
              <div className="text-blue-600 flex-shrink-0">{insight.icon}</div>
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
              <span className="text-xl mr-2">💪</span>
              <span className="font-semibold text-green-800">Strengths</span>
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
              <span className="text-xl mr-2">🚀</span>
              <span className="font-semibold text-amber-800">
                Opportunities
              </span>
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
                {chartData.feature_importance?.length || 0}
              </div>
              <div className="text-sm text-gray-600">Factors Analyzed</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-800">
                {chartData.performance_timeline?.portfolio?.length || 0}
              </div>
              <div className="text-sm text-gray-600">Data Points</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-800">AI Powered</div>
              <div className="text-sm text-gray-600">SHAP Analysis</div>
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
      <span className={color}>{icon}</span>
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

export default SHAPDashboard;
