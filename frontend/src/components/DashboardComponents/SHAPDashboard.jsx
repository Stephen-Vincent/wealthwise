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

const SHAPDashboard = ({
  simulationId,
  portfolioData: portfolioDataProp,
  apiBase = import.meta.env.VITE_API_BASE_URL || "/api",
  withCredentials = true,
}) => {
  const [portfolioData, setPortfolioData] = useState(portfolioDataProp || null);
  const [loading, setLoading] = useState(!!simulationId && !portfolioDataProp);
  const [error, setError] = useState(null);
  const [chartImages, setChartImages] = useState({});
  const [chartLoading, setChartLoading] = useState(false);

  console.log("🔍 SHAPDashboard Debug:", {
    simulationId,
    hasPortfolioDataProp: !!portfolioDataProp,
    portfolioData: portfolioData,
    loading,
    error,
  });

  // 1) Fetch the simulation
  useEffect(() => {
    let alive = true;
    if (!simulationId || portfolioDataProp) return;

    (async () => {
      try {
        setLoading(true);
        setError(null);

        const res = await fetch(`${apiBase}/simulations/${simulationId}`, {
          method: "GET",
          headers: { Accept: "application/json" },
          credentials: withCredentials ? "include" : "same-origin",
        });
        if (!res.ok)
          throw new Error(
            `Failed to load simulation ${simulationId}: ${res.status}`
          );
        const sim = await res.json();

        // 2) If SHAP missing, fetch SHAP explanation and merge
        const hasShap = !!sim?.results?.shap_explanation;
        if (!hasShap) {
          try {
            const shapRes = await fetch(
              `${apiBase}/shap/simulation/${simulationId}/explanation`,
              {
                method: "GET",
                headers: { Accept: "application/json" },
                credentials: withCredentials ? "include" : "same-origin",
              }
            );
            if (shapRes.ok) {
              const shapJson = await shapRes.json();
              // Merge into results
              const merged = {
                ...sim,
                results: {
                  ...(sim.results || {}),
                  shap_explanation: shapJson?.shap_data || {},
                  goal_analysis:
                    shapJson?.goal_analysis ?? sim?.results?.goal_analysis,
                  market_regime:
                    shapJson?.market_regime ?? sim?.results?.market_regime,
                },
              };
              if (alive) setPortfolioData(merged);
            } else {
              // Keep the original sim if SHAP endpoint 404s
              if (alive) setPortfolioData(sim);
            }
          } catch {
            if (alive) setPortfolioData(sim);
          }
        } else {
          if (alive) setPortfolioData(sim);
        }
      } catch (e) {
        if (alive) setError(e?.message || "Failed to fetch simulation");
      } finally {
        if (alive) setLoading(false);
      }
    })();

    return () => {
      alive = false;
    };
  }, [simulationId, apiBase, withCredentials, portfolioDataProp]);

  // Prefer prop if provided
  useEffect(() => {
    if (portfolioDataProp) setPortfolioData(portfolioDataProp);
  }, [portfolioDataProp]);

  // === Enhanced SHAP data detection ===
  const shapData = useMemo(() => {
    const data =
      portfolioData?.shap_explanation ||
      portfolioData?.results?.shap_explanation ||
      portfolioData?.results?.shap_explanations ||
      null;

    console.log("📊 SHAP Data Detection:", {
      topLevel: !!portfolioData?.shap_explanation,
      resultsLevel: !!portfolioData?.results?.shap_explanation,
      resultsLevelPlural: !!portfolioData?.results?.shap_explanations,
      finalData: data,
      dataKeys: data ? Object.keys(data) : [],
    });

    return data;
  }, [portfolioData]);

  const hasShapData = useMemo(() => {
    const hasData = shapData && Object.keys(shapData || {}).length > 0;
    console.log("✅ Has SHAP Data:", hasData);
    return hasData;
  }, [shapData]);

  // Fetch visualization charts when SHAP data is available
  useEffect(() => {
    if (simulationId && hasShapData && !chartLoading) {
      setChartLoading(true);

      // Fetch all charts at once
      fetch(`${apiBase}/shap/simulation/${simulationId}/all-charts`, {
        method: "GET",
        headers: { Accept: "application/json" },
        credentials: withCredentials ? "include" : "same-origin",
      })
        .then((res) => res.json())
        .then((data) => {
          console.log("📊 Charts fetched:", data);
          if (data.charts) {
            setChartImages(data.charts);
          }
          if (data.errors) {
            console.warn("Chart generation errors:", data.errors);
          }
        })
        .catch((err) => {
          console.error("Failed to fetch charts:", err);
          // Try individual chart endpoints as fallback
          fetchIndividualCharts();
        })
        .finally(() => {
          setChartLoading(false);
        });
    }
  }, [simulationId, hasShapData, apiBase, withCredentials]);

  const fetchIndividualCharts = async () => {
    const chartEndpoints = [
      { key: "shap_waterfall", endpoint: "shap-chart" },
      { key: "portfolio_composition", endpoint: "portfolio-chart" },
      { key: "risk_return", endpoint: "risk-return-chart" },
      { key: "market_regime", endpoint: "market-regime-chart" },
    ];

    const newChartImages = {};

    for (const { key, endpoint } of chartEndpoints) {
      try {
        const res = await fetch(
          `${apiBase}/shap/simulation/${simulationId}/${endpoint}`,
          {
            method: "GET",
            headers: { Accept: "application/json" },
            credentials: withCredentials ? "include" : "same-origin",
          }
        );

        if (res.ok) {
          const data = await res.json();
          if (data.chart_url) {
            newChartImages[key] = data.chart_url;
          }
        }
      } catch (err) {
        console.warn(`Failed to fetch ${key} chart:`, err);
      }
    }

    if (Object.keys(newChartImages).length > 0) {
      setChartImages(newChartImages);
    }
  };

  const [activeTab, setActiveTab] = useState("summary");
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (!hasShapData) {
      console.log("❌ No SHAP data for chart generation");
      setChartData([]);
      return;
    }

    console.log("📈 Generating chart data from SHAP:", shapData);

    const rawImportance =
      shapData.feature_importance ||
      shapData.feature_contributions ||
      shapData.shap_values ||
      {};

    console.log("📊 Raw importance data:", rawImportance);

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

    console.log("📊 Processed chart data:", entries);
    setChartData(entries);
  }, [hasShapData, shapData]);

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
          Couldn't load portfolio
        </h3>
        <p className="text-red-700">{String(error)}</p>
      </div>
    );
  }

  if (!hasShapData) {
    return (
      <div className="space-y-4">
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

        {/* Debug information */}
        <details className="bg-gray-50 border rounded-lg p-4">
          <summary className="cursor-pointer font-medium text-gray-700">
            🔍 Debug Information
          </summary>
          <div className="mt-3 space-y-2 text-sm">
            <p>
              <strong>Portfolio Data Available:</strong>{" "}
              {!!portfolioData ? "✅" : "❌"}
            </p>
            <p>
              <strong>Top-level SHAP:</strong>{" "}
              {!!portfolioData?.shap_explanation ? "✅" : "❌"}
            </p>
            <p>
              <strong>Results SHAP:</strong>{" "}
              {!!portfolioData?.results?.shap_explanation ? "✅" : "❌"}
            </p>
            <p>
              <strong>WealthWise Enhanced:</strong>{" "}
              {portfolioData?.wealthwise_enhanced ? "✅" : "❌"}
            </p>
            <p>
              <strong>Has SHAP Flag:</strong>{" "}
              {portfolioData?.has_shap_explanations ? "✅" : "❌"}
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

            {portfolioData?.results && (
              <div className="mt-3">
                <p>
                  <strong>Results Keys:</strong>
                </p>
                <code className="block bg-white p-2 rounded text-xs">
                  {Object.keys(portfolioData.results).join(", ")}
                </code>
              </div>
            )}

            {shapData && (
              <div className="mt-3">
                <p>
                  <strong>SHAP Data Preview:</strong>
                </p>
                <pre className="bg-white p-2 rounded text-xs overflow-auto max-h-32">
                  {JSON.stringify(shapData, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </details>
      </div>
    );
  }

  // SHAP data is available - render the dashboard
  return (
    <div className="w-full space-y-6">
      {/* Success message */}
      <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
        <div className="text-2xl mb-2">🎉</div>
        <h3 className="text-lg font-bold text-green-800 mb-2">
          SHAP AI Explanation Available!
        </h3>
        <p className="text-green-600">
          Our AI has generated detailed explanations for your portfolio
          recommendations.
        </p>
        {Object.keys(chartImages).length > 0 && (
          <p className="text-green-600 text-sm mt-1">
            Professional visualizations are ready to view.
          </p>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="bg-gray-50 rounded-xl p-2">
        <div className="flex space-x-2">
          {[
            { id: "summary", label: "Overview", icon: "📊" },
            { id: "factors", label: "Key Factors", icon: "🔍" },
            { id: "visualizations", label: "Charts", icon: "📈" },
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
        {activeTab === "summary" && (
          <SummaryTab shapData={shapData} portfolioData={portfolioData} />
        )}
        {activeTab === "factors" && (
          <FactorsTab chartData={chartData} shapData={shapData} />
        )}
        {activeTab === "visualizations" && (
          <VisualizationsTab
            chartImages={chartImages}
            chartLoading={chartLoading}
            simulationId={simulationId}
            apiBase={apiBase}
            withCredentials={withCredentials}
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

// New Visualizations Tab Component
const VisualizationsTab = ({
  chartImages,
  chartLoading,
  simulationId,
  apiBase,
  withCredentials,
}) => {
  const [refreshing, setRefreshing] = useState(false);

  const refreshCharts = async () => {
    setRefreshing(true);
    try {
      const res = await fetch(
        `${apiBase}/shap/simulation/${simulationId}/all-charts`,
        {
          method: "GET",
          headers: { Accept: "application/json" },
          credentials: withCredentials ? "include" : "same-origin",
        }
      );

      if (res.ok) {
        const data = await res.json();
        // Force reload images by adding timestamp
        const timestamp = Date.now();
        const refreshedImages = {};
        Object.entries(data.charts || {}).forEach(([key, url]) => {
          refreshedImages[key] = `${url}?t=${timestamp}`;
        });
        window.location.reload(); // Simple way to refresh charts
      }
    } catch (err) {
      console.error("Failed to refresh charts:", err);
    } finally {
      setRefreshing(false);
    }
  };

  if (chartLoading) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-8 text-center">
        <div className="animate-spin text-4xl mb-4">⚙️</div>
        <h3 className="text-lg font-semibold text-gray-700 mb-2">
          Generating Professional Charts
        </h3>
        <p className="text-gray-600">
          Our AI is creating detailed visualizations of your portfolio
          recommendations...
        </p>
      </div>
    );
  }

  const chartConfigs = [
    {
      key: "shap_waterfall",
      title: "SHAP Decision Waterfall",
      description:
        "Shows exactly how each factor influenced the AI's portfolio recommendations",
      icon: "📊",
    },
    {
      key: "portfolio_composition",
      title: "Portfolio Allocation",
      description: "Visual breakdown of your recommended stock allocations",
      icon: "🥧",
    },
    {
      key: "risk_return",
      title: "Risk vs Return Profile",
      description: "Where your portfolio sits in the risk-return spectrum",
      icon: "📈",
    },
    {
      key: "market_regime",
      title: "Market Conditions Analysis",
      description:
        "Current market environment and how it affects your strategy",
      icon: "🌊",
    },
  ];

  const availableCharts = chartConfigs.filter(
    (config) => chartImages[config.key]
  );

  return (
    <div className="space-y-6">
      {/* Header with refresh button */}
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-xl font-bold text-gray-800">
            Professional AI Visualizations
          </h3>
          <p className="text-gray-600">
            High-quality charts explaining your portfolio recommendations
          </p>
        </div>
        <button
          onClick={refreshCharts}
          disabled={refreshing}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {refreshing ? "Refreshing..." : "Refresh Charts"}
        </button>
      </div>

      {availableCharts.length === 0 ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-8 text-center">
          <div className="text-4xl mb-4">⚠️</div>
          <h3 className="text-lg font-semibold text-yellow-800 mb-2">
            Charts Not Available
          </h3>
          <p className="text-yellow-700">
            Professional visualizations couldn't be generated at this time.
          </p>
          <button
            onClick={refreshCharts}
            className="mt-4 px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700"
          >
            Try Generating Charts
          </button>
        </div>
      ) : (
        <div className="grid gap-6">
          {availableCharts.map((config) => (
            <div
              key={config.key}
              className="bg-white rounded-xl border border-gray-200 p-6"
            >
              <div className="flex items-center mb-4">
                <span className="text-2xl mr-3">{config.icon}</span>
                <div>
                  <h4 className="text-lg font-semibold text-gray-800">
                    {config.title}
                  </h4>
                  <p className="text-gray-600 text-sm">{config.description}</p>
                </div>
              </div>

              <div className="bg-gray-50 rounded-lg p-4">
                <img
                  src={chartImages[config.key]}
                  alt={config.title}
                  className="w-full h-auto rounded border"
                  onError={(e) => {
                    e.target.style.display = "none";
                    e.target.nextSibling.style.display = "block";
                  }}
                />
                <div className="hidden text-center py-8 text-gray-500">
                  <p>Chart could not be displayed</p>
                  <p className="text-sm">URL: {chartImages[config.key]}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Chart Statistics */}
      {availableCharts.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
          <h4 className="font-semibold text-blue-800 mb-2">
            Visualization Summary
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-blue-600">
                {availableCharts.length}
              </div>
              <div className="text-sm text-blue-700">Charts Generated</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-600">✅</div>
              <div className="text-sm text-blue-700">Professional Quality</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-600">🤖</div>
              <div className="text-sm text-blue-700">AI Generated</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-600">📊</div>
              <div className="text-sm text-blue-700">SHAP Explainable</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Summary Tab Component (unchanged)
const SummaryTab = ({ shapData, portfolioData }) => {
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
      icon: "🎯",
    },
    {
      title: "Portfolio Quality",
      value: portfolioQualityScore,
      maxValue: 100,
      unit: "/100",
      color: "text-blue-600",
      bgColor: "bg-blue-50",
      icon: "🧠",
    },
    {
      title: "Risk Score",
      value: portfolioData?.risk_score || 50,
      maxValue: 100,
      unit: "/100",
      color: "text-orange-600",
      bgColor: "bg-orange-50",
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

        {/* Human readable explanations */}
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

// Factors Tab Component (unchanged)
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
                <Bar
                  dataKey="importance"
                  radius={[4, 4, 0, 0]}
                  fill="#3B82F6"
                />
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
                <p className="text-sm text-gray-600 mb-2">
                  {factor.description}
                </p>
                <p className="text-sm font-medium text-gray-700">
                  {factor.simpleExplanation}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Insights Tab Component (unchanged)
const InsightsTab = ({ shapData, portfolioData, chartData }) => {
  // Generate user-friendly insights
  const generateInsights = () => {
    const insights = [];

    // Portfolio Strategy Insight
    const confidence = shapData?.confidence_score || shapData?.confidence || 75;
    const portfolioQuality = shapData?.portfolio_quality_score || 85;

    insights.push({
      icon: "🎯",
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
      icon: "⭐",
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

    // Top factors insights
    if (chartData && chartData.length > 0) {
      const topFactor = chartData[0];
      const isPositive = topFactor.importance >= 0;

      insights.push({
        icon: "📈",
        title: "Primary Decision Factor",
        description: `${topFactor.factor} had the ${
          isPositive ? "most positive" : "most challenging"
        } impact on your portfolio design. ${topFactor.simpleExplanation}`,
      });

      // Risk vs Timeline insight
      const riskFactor = chartData.find((f) =>
        f.factor.toLowerCase().includes("risk")
      );
      const timeFactor = chartData.find(
        (f) =>
          f.factor.toLowerCase().includes("timeline") ||
          f.factor.toLowerCase().includes("timeframe")
      );

      if (riskFactor && timeFactor) {
        insights.push({
          icon: "⚖️",
          title: "Risk & Time Balance",
          description: `Your risk comfort level and investment timeline work ${
            (riskFactor.importance > 0 && timeFactor.importance > 0) ||
            (riskFactor.importance < 0 && timeFactor.importance < 0)
              ? "well together"
              : "against each other"
          } in this portfolio strategy.`,
        });
      }
    }

    // Market conditions insight
    const marketVolatility = chartData?.find((f) =>
      f.factor.toLowerCase().includes("volatility")
    );
    const marketTrend = chartData?.find((f) =>
      f.factor.toLowerCase().includes("trend")
    );

    if (marketVolatility || marketTrend) {
      const marketImpact =
        marketVolatility?.importance || marketTrend?.importance || 0;
      insights.push({
        icon: "🌊",
        title: "Market Conditions Impact",
        description: `Current market conditions are ${
          marketImpact > 0 ? "favorable" : "challenging"
        } for your investment strategy, ${
          marketImpact > 0 ? "supporting" : "requiring adjustments to"
        } your portfolio allocation.`,
      });
    }

    // Goal achievement insight
    const goalAnalysis =
      shapData?.goal_analysis || portfolioData?.results?.goal_analysis;
    if (goalAnalysis) {
      insights.push({
        icon: "🏆",
        title: "Goal Achievement Outlook",
        description:
          typeof goalAnalysis === "string"
            ? goalAnalysis
            : "Based on your inputs and market conditions, your financial goals appear achievable with this strategy.",
      });
    }

    return insights;
  };

  const insights = generateInsights();

  return (
    <div className="space-y-6">
      {/* Key AI Insights */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
          <span className="mr-2">💡</span>
          Key AI Insights
        </h3>

        <div className="space-y-6">
          {insights.map((insight, index) => (
            <div
              key={index}
              className="flex items-start space-x-4 p-4 bg-blue-50 rounded-lg border border-blue-100"
            >
              <div className="text-2xl flex-shrink-0">{insight.icon}</div>
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

      {/* Next Steps */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">🚀</span>
          What This Means for You
        </h3>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-green-50 rounded-lg p-4 border border-green-200">
            <div className="flex items-center mb-2">
              <span className="text-xl mr-2">✅</span>
              <h4 className="font-semibold text-green-800">Strengths</h4>
            </div>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Portfolio matches your risk comfort level</li>
              <li>• Strategy aligned with your timeline</li>
              <li>• AI-optimized for current market conditions</li>
            </ul>
          </div>

          <div className="bg-amber-50 rounded-lg p-4 border border-amber-200">
            <div className="flex items-center mb-2">
              <span className="text-xl mr-2">📈</span>
              <h4 className="font-semibold text-amber-800">Opportunities</h4>
            </div>
            <ul className="text-sm text-amber-700 space-y-1">
              <li>• Regular monitoring recommended</li>
              <li>• Consider increasing contributions if possible</li>
              <li>• Rebalance as market conditions change</li>
            </ul>
          </div>
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

// Helper Functions
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
