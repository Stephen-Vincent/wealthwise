// frontend/src/components/DashboardComponents/SHAPDashboard.jsx

import React, { useState, useEffect, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

/**
 * Props (JS-only):
 * - simulationId        -> if provided, fetches /api/simulations/{id}
 * - portfolioData       -> if provided, uses this directly (no fetch)
 * - apiBase             -> defaults to "/api"
 * - withCredentials     -> defaults to true
 */
const SHAPDashboard = ({
  simulationId,
  portfolioData: portfolioDataProp,
  apiBase = "/api",
  withCredentials = true,
}) => {
  const [portfolioData, setPortfolioData] = useState(portfolioDataProp || null);
  const [loading, setLoading] = useState(!!simulationId && !portfolioDataProp);
  const [error, setError] = useState(null);

  // Fetch when we have an id and no data prop
  useEffect(() => {
    let alive = true;
    if (!simulationId || portfolioDataProp) return;

    async function run() {
      try {
        setLoading(true);
        setError(null);

        // IMPORTANT: detail route has NO trailing slash
        const url = `${apiBase}/simulations/${simulationId}`;
        const res = await fetch(url, {
          method: "GET",
          headers: { Accept: "application/json" },
          credentials: withCredentials ? "include" : "same-origin",
        });
        if (!res.ok) {
          const msg = await res.text().catch(() => "");
          throw new Error(
            `Failed to load simulation ${simulationId}: ${res.status} ${msg}`
          );
        }
        const json = await res.json();
        if (alive) setPortfolioData(json);
      } catch (e) {
        if (alive)
          setError(e && e.message ? e.message : "Failed to fetch simulation");
      } finally {
        if (alive) setLoading(false);
      }
    }

    run();
    return () => {
      alive = false;
    };
  }, [simulationId, apiBase, withCredentials, portfolioDataProp]);

  // Prefer prop if provided
  useEffect(() => {
    if (portfolioDataProp) setPortfolioData(portfolioDataProp);
  }, [portfolioDataProp]);

  if (loading) {
    return (
      <div className="bg-white border rounded-xl p-8 text-center">
        <div className="text-lg text-gray-700">Loading your portfolio…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-8 text-center">
        <div className="text-6xl mb-4">🚧</div>
        <h3 className="text-xl font-bold text-red-800 mb-3">
          Couldn’t load portfolio
        </h3>
        <p className="text-red-700">{String(error)}</p>
      </div>
    );
  }

  // ======== existing logic, just pointed at portfolioData state ========
  const shapData = useMemo(() => {
    return (
      portfolioData?.results?.shap_explanation || // singular
      portfolioData?.results?.shap_explanations || // plural (if used)
      portfolioData?.shap_explanation || // fallback if flattened
      null
    );
  }, [portfolioData]);

  const hasShapData = useMemo(() => {
    return Object.keys(shapData || {}).length > 0;
  }, [shapData]);

  const [activeTab, setActiveTab] = useState("summary");
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (!hasShapData) {
      setChartData([]);
      return;
    }
    const rawImportance =
      shapData.feature_importance || shapData.feature_contributions || {};
    const entries = Object.entries(rawImportance)
      .map(([factor, importance]) => {
        const num = Number(importance);
        const importanceNum = Number.isFinite(num) ? num : 0;
        return {
          factor: formatFactorName(factor),
          importance: importanceNum,
          color: getFactorColor(factor),
          description: getFactorDescription(factor),
          simpleExplanation: getSimpleExplanation(factor, importanceNum),
        };
      })
      .filter((d) => Number.isFinite(d.importance));

    entries.sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
    setChartData(entries);
  }, [hasShapData, shapData]);

  if (!hasShapData) {
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-8 text-center">
        <div className="text-6xl mb-4">🤖</div>
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
    );
  }

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          🤖 How Our AI Built Your Portfolio
        </h2>
        <p className="text-gray-600 text-lg max-w-2xl mx-auto">
          Our AI considered many factors when creating your personalized
          investment plan. Here's how it made its decisions in simple terms.
        </p>
      </div>

      {/* Tabs */}
      <div className="bg-gray-50 rounded-xl p-3">
        <div className="flex space-x-2">
          {[
            {
              id: "summary",
              label: "Quick Summary",
              icon: "📋",
              desc: "The basics",
            },
            {
              id: "factors",
              label: "What Influenced Your Plan",
              icon: "🎯",
              desc: "Key factors",
            },
            {
              id: "insights",
              label: "Personalized Insights",
              icon: "💡",
              desc: "For you",
            },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 px-4 py-4 rounded-lg font-semibold transition-all duration-200 text-center ${
                activeTab === tab.id
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-gray-600 hover:bg-gray-200"
              }`}
            >
              <div className="text-xl mb-1">{tab.icon}</div>
              <div className="text-sm font-bold">{tab.label}</div>
              <div className="text-xs opacity-75">{tab.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="min-h-96">
        {activeTab === "summary" && (
          <SummaryTab shapData={shapData} portfolioData={portfolioData} />
        )}
        {activeTab === "factors" && (
          <FactorsTab chartData={chartData} shapData={shapData} />
        )}
        {activeTab === "insights" && (
          <InsightsTab portfolioData={portfolioData} />
        )}
      </div>
    </div>
  );
};

// ——— rest of your original helpers/components unchanged ———

const SummaryTab = ({ shapData, portfolioData }) => {
  const portfolioQuality = Number(shapData.portfolio_quality_score) || 0;
  const goalAnalysis = portfolioData?.results?.goal_analysis || {};
  const marketRegime = portfolioData?.results?.market_regime || {};

  const getQualityMessage = (score) => {
    if (score >= 80)
      return {
        text: "Excellent match for your goals!",
        color: "text-green-600",
        bg: "bg-green-50",
      };
    if (score >= 60)
      return {
        text: "Good fit for your situation",
        color: "text-blue-600",
        bg: "bg-blue-50",
      };
    if (score >= 40)
      return {
        text: "Reasonable approach",
        color: "text-yellow-600",
        bg: "bg-yellow-50",
      };
    return { text: "Basic strategy", color: "text-gray-600", bg: "bg-gray-50" };
  };

  const qualityMsg = getQualityMessage(portfolioQuality);
  const marketTrendScore = Number(marketRegime?.trend_score) || 3;
  const vix = Number(marketRegime?.current_vix);
  const vixDisplay = Number.isFinite(vix) ? vix.toFixed(0) : "20";
  const mktType = (marketRegime?.regime || "balanced")
    .replace(/_/g, " ")
    .toUpperCase();

  const metrics = [
    {
      title: "How Good Is This Plan?",
      subtitle: "Our AI's confidence in your portfolio",
      value: portfolioQuality,
      maxValue: 100,
      unit: "%",
      color: qualityMsg.color,
      bgColor: qualityMsg.bg,
      icon: "🎯",
      explanation: qualityMsg.text,
    },
    {
      title: "Can You Reach Your Goal?",
      subtitle: "Likelihood of success",
      value: (Number(goalAnalysis.feasibility_rating) || 0) * 20,
      maxValue: 100,
      unit: "%",
      color: "text-purple-600",
      bgColor: "bg-purple-50",
      icon: "🏆",
      explanation: "Based on your timeline and contributions",
    },
    {
      title: "Market Conditions",
      subtitle: "How favorable are current conditions",
      value: (Number(marketRegime.confidence) || 0) * 100,
      maxValue: 100,
      unit: "%",
      color: "text-orange-600",
      bgColor: "bg-orange-50",
      icon: "📈",
      explanation: "Current market environment for investing",
    },
  ];

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {metrics.map((m, i) => (
          <SimpleMetricCard key={i} {...m} />
        ))}
      </div>

      <div className="bg-white rounded-xl border-2 border-blue-200 p-8">
        <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
          <span className="text-2xl mr-3">🤖</span> What Our AI Is Thinking
        </h3>
        <div className="space-y-4">
          {Object.entries(shapData.human_readable_explanation || {}).map(
            ([key, explanation], idx) => (
              <div
                key={idx}
                className="bg-blue-50 rounded-lg p-6 border-l-4 border-blue-500"
              >
                <div className="font-semibold text-blue-800 mb-3 text-lg">
                  {formatFactorName(key)}
                </div>
                <p className="text-gray-700 leading-relaxed text-base">
                  {String(explanation)}
                </p>
              </div>
            )
          )}
          {Object.keys(shapData.human_readable_explanation || {}).length ===
            0 && (
            <div className="bg-blue-50 rounded-lg p-6 text-center">
              <div className="text-4xl mb-3">🤖</div>
              <p className="text-blue-700 text-lg">
                Our AI analyzed your situation and created a personalized
                strategy…
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-3">🌍</span> Current Market Environment
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center bg-white rounded-lg p-4">
            <div className="text-2xl font-bold text-blue-600 mb-1">
              {mktType}
            </div>
            <div className="text-sm text-gray-600">Market Type</div>
            <div className="text-xs text-gray-500 mt-1">
              How markets are behaving
            </div>
          </div>
          <div className="text-center bg-white rounded-lg p-4">
            <div className="text-2xl font-bold text-green-600 mb-1">
              {marketTrendScore}/5
            </div>
            <div className="text-sm text-gray-600">Trend Strength</div>
            <div className="text-xs text-gray-500 mt-1">Market momentum</div>
          </div>
          <div className="text-center bg-white rounded-lg p-4">
            <div className="text-2xl font-bold text-orange-600 mb-1">
              {vixDisplay}
            </div>
            <div className="text-sm text-gray-600">Fear Level</div>
            <div className="text-xs text-gray-500 mt-1">
              Market anxiety (lower is calmer)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const FactorsTab = ({ chartData }) => (
  <div className="space-y-8">
    <div className="text-center">
      <h3 className="text-xl font-bold text-gray-800 mb-3">
        What Influenced Your Investment Plan
      </h3>
      <p className="text-gray-600 max-w-2xl mx-auto">
        Our AI looked at many things about your situation. Here are the factors
        that had the biggest impact.
      </p>
    </div>

    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <h4 className="text-lg font-bold text-gray-800 mb-6 flex items-center">
        <span className="mr-3">📊</span> Impact of Different Factors
        <span className="ml-auto text-sm font-normal text-gray-500">
          Bigger bars = more important
        </span>
      </h4>
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
              formatter={(value) => [
                `${value > 0 ? "Helped" : "Limited"} (${Math.abs(
                  Number(value) || 0
                ).toFixed(3)})`,
                "Impact",
              ]}
              labelFormatter={(label) => `${label}`}
              contentStyle={{
                backgroundColor: "#f9fafb",
                border: "1px solid #e5e7eb",
                borderRadius: "8px",
              }}
            />
            <Bar dataKey="importance" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex justify-center mt-4 space-x-6 text-sm">
        <div className="flex items-center">
          <div className="w-4 h-4 bg-green-500 rounded mr-2"></div>
          <span>Helped your portfolio</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-4 bg-orange-500 rounded mr-2"></div>
          <span>Created challenges</span>
        </div>
      </div>
    </div>

    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <h4 className="text-lg font-bold text-gray-800 mb-6 flex items-center">
        <span className="mr-3">💭</span> What Each Factor Means for You
      </h4>
      <div className="space-y-4">
        {chartData.map((factor, index) => {
          const positive = factor.importance >= 0;
          return (
            <div
              key={index}
              className={`rounded-lg p-5 border-l-4 ${
                positive
                  ? "bg-green-50 border-green-500"
                  : "bg-orange-50 border-orange-500"
              }`}
            >
              <div className="flex justify-between items-start mb-3">
                <span className="font-semibold text-gray-800 text-lg">
                  {factor.factor}
                </span>
                <div className="text-right">
                  <span
                    className={`font-bold text-sm ${
                      positive ? "text-green-600" : "text-orange-600"
                    }`}
                  >
                    {positive ? "✓ Helpful" : "⚠ Challenge"}
                  </span>
                  <div className="text-xs text-gray-500">
                    Impact:{" "}
                    {Math.abs(Number(factor.importance) || 0).toFixed(3)}
                  </div>
                </div>
              </div>
              <p className="text-gray-700 mb-2 leading-relaxed">
                {factor.description}
              </p>
              <p className="text-sm text-gray-600 italic">
                {factor.simpleExplanation}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  </div>
);

const InsightsTab = ({ portfolioData }) => {
  const goalAnalysis = portfolioData?.results?.goal_analysis || {};
  const feasibilityAssessment =
    portfolioData?.results?.feasibility_assessment || {};

  const requiredReturn = Number(goalAnalysis.required_return_percent) || 0;
  const feasibilityScore = Number(feasibilityAssessment.feasibility_score) || 0;

  const getReturnMessage = (returnRate) => {
    if (returnRate > 15)
      return {
        text: "Very ambitious - needs strong growth",
        color: "text-red-600",
        icon: "🚀",
      };
    if (returnRate > 10)
      return {
        text: "Ambitious - requires growth investments",
        color: "text-orange-600",
        icon: "📈",
      };
    if (returnRate > 7)
      return {
        text: "Moderate - balanced approach works",
        color: "text-blue-600",
        icon: "⚖️",
      };
    return {
      text: "Conservative - steady growth is enough",
      color: "text-green-600",
      icon: "🐢",
    };
  };

  const getFeasibilityMessage = (score) => {
    if (score >= 80)
      return {
        text: "Very likely to succeed!",
        color: "text-green-600",
        icon: "🎯",
      };
    if (score >= 60)
      return {
        text: "Good chances of success",
        color: "text-blue-600",
        icon: "👍",
      };
    if (score >= 40)
      return {
        text: "Possible with discipline",
        color: "text-yellow-600",
        icon: "💪",
      };
    return {
      text: "Challenging - consider adjusting goals",
      color: "text-red-600",
      icon: "🤔",
    };
  };

  const returnMsg = getReturnMessage(requiredReturn);
  const feasibilityMsg = getFeasibilityMessage(feasibilityScore);

  return (
    <div className="space-y-8">
      <div className="bg-white rounded-xl border border-gray-200 p-8">
        <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
          <span className="mr-3">🎯</span>
          Your Personal Investment Journey
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="text-center">
            <div className="mb-4">
              <span className="text-4xl">{returnMsg.icon}</span>
            </div>
            <h4 className="font-semibold text-gray-700 mb-3 text-lg">
              Growth Needed Each Year
            </h4>
            <div className={`text-4xl font-bold mb-3 ${returnMsg.color}`}>
              {requiredReturn.toFixed(1)}%
            </div>
            <p className="text-sm text-gray-600 mb-2">
              This is how much your investments need to grow annually
            </p>
            <p className={`text-sm font-medium ${returnMsg.color}`}>
              {returnMsg.text}
            </p>
          </div>
          <div className="text-center">
            <div className="mb-4">
              <span className="text-4xl">{feasibilityMsg.icon}</span>
            </div>
            <h4 className="font-semibold text-gray-700 mb-3 text-lg">
              Your Success Probability
            </h4>
            <div className={`text-4xl font-bold mb-3 ${feasibilityMsg.color}`}>
              {Math.round(feasibilityScore)}%
            </div>
            <p className="text-sm text-gray-600 mb-2">
              Based on your situation and market history
            </p>
            <p className={`text-sm font-medium ${feasibilityMsg.color}`}>
              {feasibilityMsg.text}
            </p>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
          <span className="mr-3">💡</span>
          What This Means for You
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InsightCard
            title="Your Investment Style"
            insight="Our AI chose a strategy that matches your comfort with risk and your timeline"
            icon="🎨"
            color="bg-blue-50 text-blue-800 border-blue-200"
          />
          <InsightCard
            title="Risk & Reward Balance"
            insight="We balanced the growth you need with the level of ups and downs you can handle"
            icon="⚖️"
            color="bg-green-50 text-green-800 border-green-200"
          />
          <InsightCard
            title="Market Timing"
            insight="Your portfolio is designed for current market conditions and your long-term goals"
            icon="⏰"
            color="bg-purple-50 text-purple-800 border-purple-200"
          />
          <InsightCard
            title="Goal-Focused Design"
            insight="Every investment choice was made to help you reach your specific financial target"
            icon="🎯"
            color="bg-orange-50 text-orange-800 border-orange-200"
          />
        </div>
      </div>
    </div>
  );
};

const SimpleMetricCard = ({
  title,
  subtitle,
  value,
  maxValue,
  unit,
  color,
  bgColor,
  icon,
  explanation,
}) => (
  <div className={`${bgColor} rounded-xl p-6 border-2 border-gray-100`}>
    <div className="text-center">
      <div className="text-4xl mb-3">{icon}</div>
      <h4 className="font-bold text-gray-800 mb-1">{title}</h4>
      <p className="text-sm text-gray-600 mb-4">{subtitle}</p>
      <div className={`text-3xl font-bold ${color} mb-2`}>
        {typeof value === "number" ? Math.round(value) : value}
        {unit}
      </div>
      <p className="text-sm text-gray-700 font-medium">{explanation}</p>
      {maxValue && (
        <div className="w-full bg-gray-200 rounded-full h-3 mt-4">
          <div
            className={`h-3 rounded-full transition-all duration-500 ${
              color.includes("green")
                ? "bg-green-500"
                : color.includes("blue")
                ? "bg-blue-500"
                : color.includes("orange")
                ? "bg-orange-500"
                : "bg-purple-500"
            }`}
            style={{
              width: `${Math.min(
                ((Number(value) || 0) / maxValue) * 100,
                100
              )}%`,
            }}
          />
        </div>
      )}
    </div>
  </div>
);

const InsightCard = ({ title, insight, icon, color }) => (
  <div className={`${color} rounded-lg p-5 border`}>
    <div className="flex items-center mb-3">
      <span className="text-2xl mr-3">{icon}</span>
      <span className="font-bold text-lg">{title}</span>
    </div>
    <p className="text-sm leading-relaxed">{insight}</p>
  </div>
);

// Helpers
const formatFactorName = (factor) => {
  const formatMap = {
    risk_score: "Your Risk Comfort Level",
    target_value_log: "Your Goal Amount",
    timeframe: "Your Timeline",
    required_return: "Growth You Need",
    monthly_contribution: "Your Monthly Savings",
    market_volatility: "Current Market Stability",
    market_trend_score: "Market Momentum",
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
  };
  return (
    descriptions[factor] ||
    "This factor influenced how we built your investment portfolio"
  );
};

const getSimpleExplanation = (factor, importance) => {
  const isPositive = (Number(importance) || 0) >= 0;
  const explanations = {
    risk_score: isPositive
      ? "Your comfort with risk allowed us to include more growth investments"
      : "Your preference for stability meant we chose more conservative investments",
    target_value_log: isPositive
      ? "Your goal amount worked well with our investment timeline"
      : "Your target required us to be more aggressive than usual",
    timeframe: isPositive
      ? "Your timeline gave us good flexibility in choosing investments"
      : "Your shorter timeline limited our investment options",
    required_return: isPositive
      ? "The returns you need are achievable with our strategy"
      : "You need high returns, which required taking more risk",
    monthly_contribution: isPositive
      ? "Your regular savings help build wealth steadily over time"
      : "Higher contributions would help reach your goal more easily",
    market_volatility: isPositive
      ? "Current market conditions are favorable for your strategy"
      : "Market uncertainty made us more cautious with your money",
    market_trend_score: isPositive
      ? "Market trends are working in favor of your investment plan"
      : "Market conditions created some challenges for your strategy",
  };
  return (
    explanations[factor] ||
    (isPositive
      ? "This factor supported your investment strategy"
      : "This factor created some constraints for your plan")
  );
};

export default SHAPDashboard;
