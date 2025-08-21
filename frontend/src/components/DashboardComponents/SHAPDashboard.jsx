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
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  Area,
  AreaChart,
  ComposedChart,
  Legend,
} from "recharts";

const SHAPDashboard = ({
  simulationId,
  portfolioData: portfolioDataProp,
  apiBase = import.meta.env.VITE_API_BASE_URL || "/api",
  withCredentials = true,
}) => {
  const [portfolioData, setPortfolioData] = useState(portfolioDataProp || null);
  const [chartData, setChartData] = useState(null);
  const [enhancedData, setEnhancedData] = useState(null);
  const [loading, setLoading] = useState(!!simulationId && !portfolioDataProp);
  const [error, setError] = useState(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [debugInfo, setDebugInfo] = useState({}); // Add debug state

  // Resolve API base
  const resolvedApiBase = useMemo(() => {
    const env = (import.meta.env.VITE_API_URL || apiBase || "").trim();
    if (!env) return "/api";
    return env.replace(/\/+$/, "");
  }, [apiBase]);

  // Derive simulation ID
  const derivedSimulationId =
    simulationId ??
    portfolioDataProp?.id ??
    portfolioDataProp?.simulation_id ??
    portfolioData?.id ??
    portfolioData?.simulation_id ??
    null;

  console.log("SHAPDashboard Debug:", {
    propSimulationId: simulationId,
    derivedSimulationId,
    hasPortfolioDataProp: !!portfolioDataProp,
    portfolioData,
    loading,
    error,
    apiBase: resolvedApiBase,
    chartData,
    enhancedData,
  });

  // Enhanced debugging helper function
  const debugApiResponse = async (url, response, responseText, parsedData) => {
    const debug = {
      url,
      status: response.status,
      statusText: response.statusText,
      ok: response.ok,
      headers: Object.fromEntries(response.headers.entries()),
      responseTextLength: responseText?.length || 0,
      responseTextPreview: responseText?.substring(0, 200) || "",
      isValidJson: false,
      parsedDataKeys: [],
      parsedDataStructure: null,
      expectedKeys: ["success", "chart_data", "enhanced_data"],
    };

    try {
      if (parsedData) {
        debug.isValidJson = true;
        debug.parsedDataKeys = Object.keys(parsedData);
        debug.parsedDataStructure = {
          hasSuccess: "success" in parsedData,
          successValue: parsedData.success,
          hasChartData: "chart_data" in parsedData,
          hasEnhancedData: "enhanced_data" in parsedData,
          hasError: "error" in parsedData,
          errorValue: parsedData.error,
          topLevelKeys: Object.keys(parsedData),
        };

        if (parsedData.chart_data) {
          debug.chartDataKeys = Object.keys(parsedData.chart_data);
        }
        if (parsedData.enhanced_data) {
          debug.enhancedDataKeys = Object.keys(parsedData.enhanced_data);
        }
      }
    } catch (e) {
      debug.parseError = e.message;
    }

    console.group(`🔍 API Debug: ${url}`);
    console.log("📊 Response Status:", debug.status, debug.ok ? "✅" : "❌");
    console.log("📋 Headers:", debug.headers);
    console.log("📄 Response Preview:", debug.responseTextPreview);
    console.log("🔧 Parsed Data Structure:", debug.parsedDataStructure);
    console.log("🎯 Expected vs Received Keys:", {
      expected: debug.expectedKeys,
      received: debug.parsedDataKeys,
      missing: debug.expectedKeys.filter(
        (key) => !debug.parsedDataKeys.includes(key)
      ),
    });
    console.groupEnd();

    setDebugInfo((prev) => ({ ...prev, [url]: debug }));
    return debug;
  };

  // Helper functions
  const looksLikeSpaHtmlResponse = (res) => {
    const ct = res.headers.get("content-type") || "";
    return res.ok && ct.includes("text/html");
  };

  async function safeJson(res) {
    const ct = res.headers.get("content-type") || "";
    const responseText = await res.text();

    console.log(`🔍 Raw response for ${res.url}:`, {
      contentType: ct,
      textLength: responseText.length,
      textPreview: responseText.substring(0, 300),
      startsWithBrace: responseText.startsWith("{"),
      endsWithBrace: responseText.endsWith("}"),
    });

    if (!ct.includes("application/json") && !responseText.startsWith("{")) {
      console.error(
        `❌ Expected JSON but got "${ct}". Response:`,
        responseText.slice(0, 200)
      );
      throw new Error(
        `Expected JSON but got "${ct}". First bytes: ${responseText.slice(
          0,
          80
        )}`
      );
    }

    try {
      const parsed = JSON.parse(responseText);
      console.log(`✅ Successfully parsed JSON:`, parsed);
      await debugApiResponse(res.url, res, responseText, parsed);
      return parsed;
    } catch (parseError) {
      console.error(`❌ JSON parse failed:`, parseError);
      console.log(`📄 Full response text:`, responseText);
      await debugApiResponse(res.url, res, responseText, null);
      throw new Error(`Invalid JSON response: ${parseError.message}`);
    }
  }

  // Fetch simulation data
  useEffect(() => {
    let alive = true;
    if (!derivedSimulationId || portfolioDataProp) return;

    (async () => {
      try {
        setLoading(true);
        setError(null);

        const res = await fetch(
          `${resolvedApiBase}/simulations/${derivedSimulationId}`,
          {
            method: "GET",
            headers: { Accept: "application/json" },
            credentials: withCredentials ? "include" : "same-origin",
          }
        );

        console.log(`🔍 Simulation fetch response:`, res.status, res.ok);

        if (!res.ok)
          throw new Error(
            `Failed to load simulation ${derivedSimulationId}: ${res.status}`
          );

        const sim = await safeJson(res);
        console.log(`📊 Simulation data received:`, sim);

        // If SHAP missing, fetch SHAP explanation and merge
        const hasShap = !!sim?.results?.shap_explanation;
        console.log(`🔍 Has SHAP in simulation:`, hasShap);

        if (!hasShap) {
          try {
            const shapUrl = `${resolvedApiBase}/shap-visualization/simulation/${derivedSimulationId}/explanation`;
            console.log(`🔍 Fetching SHAP from:`, shapUrl);

            const shapRes = await fetch(shapUrl, {
              method: "GET",
              headers: { Accept: "application/json" },
              credentials: withCredentials ? "include" : "same-origin",
            });

            console.log(`🔍 SHAP fetch response:`, shapRes.status, shapRes.ok);

            if (shapRes.ok) {
              const shapJson = await safeJson(shapRes);
              console.log(`📊 SHAP data received:`, shapJson);

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
              console.log(`🔄 Merged simulation data:`, merged);
              if (alive) setPortfolioData(merged);
            } else {
              console.warn(`⚠️ SHAP fetch failed, using original sim data`);
              if (alive) setPortfolioData(sim);
            }
          } catch (shapError) {
            console.error(`❌ SHAP fetch error:`, shapError);
            if (alive) setPortfolioData(sim);
          }
        } else {
          if (alive) setPortfolioData(sim);
        }
      } catch (e) {
        console.error(`❌ Simulation fetch failed:`, e);
        if (alive) setError(e?.message || "Failed to fetch simulation");
      } finally {
        if (alive) setLoading(false);
      }
    })();

    return () => {
      alive = false;
    };
  }, [
    derivedSimulationId,
    resolvedApiBase,
    withCredentials,
    portfolioDataProp,
  ]);

  // Prefer prop if provided
  useEffect(() => {
    if (portfolioDataProp) {
      console.log(`🔄 Using portfolio data from props:`, portfolioDataProp);
      setPortfolioData(portfolioDataProp);
    }
  }, [portfolioDataProp]);

  // SHAP data detection
  const shapData = useMemo(() => {
    const data =
      portfolioData?.shap_explanation ||
      portfolioData?.results?.shap_explanation ||
      null;

    console.log("🔍 SHAP Data Detection:", {
      topLevel: !!portfolioData?.shap_explanation,
      resultsLevel: !!portfolioData?.results?.shap_explanation,
      finalData: data,
      dataKeys: data ? Object.keys(data) : [],
      dataPreview: data ? JSON.stringify(data).substring(0, 200) : null,
    });

    return data;
  }, [portfolioData]);

  const hasShapData = useMemo(() => {
    const hasData = shapData && Object.keys(shapData || {}).length > 0;
    console.log("🎯 Has SHAP Data:", hasData, shapData);
    return hasData;
  }, [shapData]);

  // Enhanced chart data fetching with better error handling
  useEffect(() => {
    if (!derivedSimulationId || !hasShapData) {
      console.log(`⏭️ Skipping chart data fetch:`, {
        hasSimId: !!derivedSimulationId,
        hasShapData,
      });
      return;
    }

    let cancelled = false;
    setChartLoading(true);

    const loadChartData = async () => {
      try {
        console.log(
          `🔄 Fetching chart data for simulation:`,
          derivedSimulationId
        );

        // Fetch comprehensive chart data
        const chartUrl = `${resolvedApiBase}/shap-visualization/simulation/${derivedSimulationId}/chart-data`;
        console.log(`🔍 Chart data URL:`, chartUrl);

        const chartRes = await fetch(chartUrl, {
          method: "GET",
          headers: { Accept: "application/json" },
          credentials: withCredentials ? "include" : "same-origin",
        });

        console.log("📊 Chart data response:", {
          status: chartRes.status,
          ok: chartRes.ok,
          statusText: chartRes.statusText,
          url: chartRes.url,
        });

        // Check if response looks like SPA HTML
        if (looksLikeSpaHtmlResponse(chartRes)) {
          console.warn(
            "⚠️ Received HTML response instead of JSON for chart data"
          );
          throw new Error("Received HTML response instead of JSON");
        }

        if (chartRes.ok) {
          try {
            const chartDataResult = await safeJson(chartRes);
            console.log("📊 Chart data received:", chartDataResult);

            // Check expected structure
            if (chartDataResult && typeof chartDataResult === "object") {
              if (
                chartDataResult.success === true &&
                chartDataResult.chart_data
              ) {
                console.log("✅ Chart data has expected structure");
                if (!cancelled) {
                  setChartData(chartDataResult.chart_data);
                }
              } else if (chartDataResult.success === false) {
                console.error(
                  "❌ Chart data API returned success: false",
                  chartDataResult
                );
                throw new Error(
                  chartDataResult.error || "Chart data API request failed"
                );
              } else {
                console.warn(
                  "⚠️ Unexpected chart data structure:",
                  chartDataResult
                );
                // Try to use the data anyway if it looks valid
                if (chartDataResult.chart_data && !cancelled) {
                  setChartData(chartDataResult.chart_data);
                } else if (
                  Object.keys(chartDataResult).length > 0 &&
                  !cancelled
                ) {
                  // Maybe the whole response is the chart data
                  setChartData(chartDataResult);
                }
              }
            } else {
              console.error(
                "❌ Chart data is not an object:",
                typeof chartDataResult,
                chartDataResult
              );
            }
          } catch (jsonError) {
            console.error("❌ Chart data JSON parse failed:", jsonError);
            throw jsonError;
          }
        } else {
          console.error(
            "❌ Chart data fetch failed:",
            chartRes.status,
            chartRes.statusText
          );
          throw new Error(
            `Chart data fetch failed: ${chartRes.status} ${chartRes.statusText}`
          );
        }

        // Fetch enhanced portfolio data
        const enhancedUrl = `${resolvedApiBase}/shap-visualization/simulation/${derivedSimulationId}/enhanced-data`;
        console.log(`🔍 Enhanced data URL:`, enhancedUrl);

        const enhancedRes = await fetch(enhancedUrl, {
          method: "GET",
          headers: { Accept: "application/json" },
          credentials: withCredentials ? "include" : "same-origin",
        });

        console.log("📈 Enhanced data response:", {
          status: enhancedRes.status,
          ok: enhancedRes.ok,
          statusText: enhancedRes.statusText,
        });

        if (enhancedRes.ok && !looksLikeSpaHtmlResponse(enhancedRes)) {
          try {
            const enhancedDataResult = await safeJson(enhancedRes);
            console.log("📈 Enhanced data received:", enhancedDataResult);

            if (
              enhancedDataResult &&
              enhancedDataResult.success &&
              enhancedDataResult.enhanced_data &&
              !cancelled
            ) {
              setEnhancedData(enhancedDataResult.enhanced_data);
            } else if (enhancedDataResult && !enhancedDataResult.success) {
              console.warn(
                "⚠️ Enhanced data API returned success: false",
                enhancedDataResult
              );
            }
          } catch (jsonError) {
            console.error("❌ Enhanced data JSON parse failed:", jsonError);
          }
        } else {
          console.warn(
            "⚠️ Enhanced data fetch failed or returned HTML:",
            enhancedRes.status
          );
        }
      } catch (error) {
        console.error("❌ Failed to fetch chart data:", error);
        console.error("Error details:", {
          message: error.message,
          stack: error.stack,
          name: error.name,
        });
      } finally {
        if (!cancelled) {
          setChartLoading(false);
        }
      }
    };

    loadChartData();

    return () => {
      cancelled = true;
    };
  }, [derivedSimulationId, hasShapData, resolvedApiBase, withCredentials]);

  const [activeTab, setActiveTab] = useState("summary");

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

        {/* Enhanced Debug information */}
        <details className="bg-gray-50 border rounded-lg p-4">
          <summary className="cursor-pointer font-medium text-gray-700">
            Debug Information
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

            {/* API Debug Info */}
            {Object.keys(debugInfo).length > 0 && (
              <div className="mt-3">
                <p>
                  <strong>API Debug Info:</strong>
                </p>
                <pre className="bg-white p-2 rounded text-xs overflow-auto max-h-48">
                  {JSON.stringify(debugInfo, null, 2)}
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
        {chartData && (
          <p className="text-green-600 text-sm mt-1">
            Interactive charts are ready to explore.
          </p>
        )}
      </div>

      {/* Enhanced Debug Panel */}
      <details className="bg-gray-50 border rounded-lg p-4">
        <summary className="cursor-pointer font-medium text-gray-700">
          Chart Data Debug
        </summary>
        <div className="mt-3 space-y-2 text-sm">
          <p>
            <strong>Chart Data Available:</strong> {!!chartData ? "✅" : "❌"}
          </p>
          <p>
            <strong>Enhanced Data Available:</strong>{" "}
            {!!enhancedData ? "✅" : "❌"}
          </p>
          <p>
            <strong>Chart Loading:</strong> {chartLoading ? "🔄" : "✅"}
          </p>

          {chartData && (
            <div>
              <p>
                <strong>Chart Data Keys:</strong>
              </p>
              <code className="block bg-white p-2 rounded text-xs">
                {Object.keys(chartData).join(", ")}
              </code>
              <details className="mt-2">
                <summary className="cursor-pointer text-xs text-gray-600">
                  View Chart Data Structure
                </summary>
                <pre className="bg-white p-2 rounded text-xs overflow-auto max-h-32 mt-1">
                  {JSON.stringify(chartData, null, 2)}
                </pre>
              </details>
            </div>
          )}

          {enhancedData && (
            <div>
              <p>
                <strong>Enhanced Data Keys:</strong>
              </p>
              <code className="block bg-white p-2 rounded text-xs">
                {Object.keys(enhancedData).join(", ")}
              </code>
            </div>
          )}

          {Object.keys(debugInfo).length > 0 && (
            <details className="mt-2">
              <summary className="cursor-pointer text-xs text-gray-600">
                View API Debug Info
              </summary>
              <pre className="bg-white p-2 rounded text-xs overflow-auto max-h-48 mt-1">
                {JSON.stringify(debugInfo, null, 2)}
              </pre>
            </details>
          )}
        </div>
      </details>

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
            chartLoading={chartLoading}
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

// Rest of your component code remains the same...
// [Include all the other components: SummaryTab, FactorsTab, etc.]

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

      {/* Portfolio Composition Chart */}
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

// Factors Tab - Fixed Bar chart coloring
const FactorsTab = ({ chartData, shapData }) => {
  // Process factor importance data
  const factorImportanceData = useMemo(() => {
    if (!chartData?.factor_importance) {
      // Fallback to processing SHAP data directly
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
    return chartData.factor_importance;
  }, [chartData, shapData]);

  // Custom bar color function
  const getBarColor = (entry) => {
    return entry.importance >= 0 ? "#10B981" : "#EF4444";
  };

  return (
    <div className="space-y-6">
      {/* Feature Importance Chart */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">📊</span>
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
                  contentStyle={{
                    backgroundColor: "#f9fafb",
                    border: "1px solid #e5e7eb",
                    borderRadius: "8px",
                  }}
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

      {/* SHAP Waterfall Chart */}
      {chartData?.shap_waterfall && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
            <span className="mr-2">🌊</span>
            SHAP Decision Waterfall
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData.shap_waterfall.contributions}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="feature"
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis />
                <Tooltip />
                <Bar dataKey="contribution">
                  {chartData.shap_waterfall.contributions.map(
                    (entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.contribution >= 0 ? "#10B981" : "#EF4444"}
                      />
                    )
                  )}
                </Bar>
                <Line
                  type="monotone"
                  dataKey="cumulative"
                  stroke="#3B82F6"
                  strokeWidth={2}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Factor Explanations */}
      {factorImportanceData.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
            <span className="mr-2">💡</span>
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
                <p className="text-sm text-gray-600">{factor.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Visualizations Tab
const VisualizationsTab = ({
  chartData,
  enhancedData,
  chartLoading,
  portfolioData,
}) => {
  if (chartLoading) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-8 text-center">
        <div className="animate-spin text-4xl mb-4">⚙️</div>
        <h3 className="text-lg font-semibold text-gray-700 mb-2">
          Loading Interactive Charts
        </h3>
        <p className="text-gray-600">Preparing your data visualizations...</p>
      </div>
    );
  }

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
      {/* Risk Return Analysis */}
      {chartData?.risk_return_analysis && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">
            Risk vs Return Analysis
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                data={[
                  chartData.risk_return_analysis.portfolio,
                  ...chartData.risk_return_analysis.benchmark,
                ]}
              >
                <CartesianGrid />
                <XAxis
                  type="number"
                  dataKey="volatility"
                  name="Risk"
                  label={{
                    value: "Risk (%)",
                    position: "insideBottom",
                    offset: -20,
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="expected_return"
                  name="Return"
                  label={{
                    value: "Expected Return (%)",
                    angle: -90,
                    position: "insideLeft",
                  }}
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  formatter={(value, name) => [
                    `${(value * 100).toFixed(2)}%`,
                    name === "volatility" ? "Risk" : "Expected Return",
                  ]}
                  labelFormatter={(label, payload) =>
                    payload?.[0]?.payload?.label || "Portfolio Asset"
                  }
                />
                <Scatter
                  name="Assets"
                  data={[
                    chartData.risk_return_analysis.portfolio,
                    ...chartData.risk_return_analysis.benchmark,
                  ]}
                  fill="#3B82F6"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Market Regime Analysis */}
      {chartData?.market_regime && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">
            Market Conditions
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={Object.entries(
                  chartData.market_regime.regime_probabilities || {}
                ).map(([name, value]) => ({
                  name,
                  probability: value,
                  isCurrent: name === chartData.market_regime.current_regime,
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip
                  formatter={(value) => [
                    `${(value * 100).toFixed(1)}%`,
                    "Probability",
                  ]}
                />
                <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                  {Object.entries(
                    chartData.market_regime.regime_probabilities || {}
                  ).map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={
                        entry[0] === chartData.market_regime.current_regime
                          ? "#3B82F6"
                          : "#9CA3AF"
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Goal Analysis */}
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
                £{chartData.goal_analysis.current_value?.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Current Value</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {(chartData.goal_analysis.probability_of_success * 100).toFixed(
                  0
                )}
                %
              </div>
              <div className="text-sm text-gray-600">Success Probability</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {chartData.goal_analysis.time_horizon} years
              </div>
              <div className="text-sm text-gray-600">Time Horizon</div>
            </div>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div
              className="bg-blue-600 h-4 rounded-full transition-all duration-500"
              style={{
                width: `${Math.min(
                  chartData.goal_analysis.progress_percentage,
                  100
                )}%`,
              }}
            />
          </div>
          <div className="text-center mt-2 text-sm text-gray-600">
            {chartData.goal_analysis.progress_percentage.toFixed(1)}% progress
            toward goal
          </div>
        </div>
      )}

      {/* Performance Timeline */}
      {enhancedData?.portfolio_timeline && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">
            Portfolio Performance Over Time
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={enhancedData.portfolio_timeline.slice(-60)}>
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

      {/* Individual Stock Performance */}
      {enhancedData?.individual_stock_performance && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">
            Individual Stock Analysis
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={enhancedData.individual_stock_performance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="symbol" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar
                  dataKey="allocation"
                  fill="#3B82F6"
                  name="Portfolio Weight %"
                />
                <Bar
                  dataKey="risk_metrics.volatility"
                  fill="#10B981"
                  name="Volatility"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Chart Statistics */}
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

    // Factor-based insights
    if (
      chartData?.factor_importance &&
      chartData.factor_importance.length > 0
    ) {
      const topFactor = chartData.factor_importance[0];
      const isPositive = topFactor.importance >= 0;

      insights.push({
        icon: "TrendingUp",
        title: "Primary Decision Factor",
        description: `${topFactor.factor} had the ${
          isPositive ? "most positive" : "most challenging"
        } impact on your portfolio design with an impact score of ${topFactor.importance.toFixed(
          3
        )}.`,
      });
    }

    // Goal analysis insights
    if (chartData?.goal_analysis) {
      const successProb = chartData.goal_analysis.probability_of_success * 100;
      insights.push({
        icon: "Trophy",
        title: "Goal Achievement Outlook",
        description: `Based on your inputs and current market conditions, you have a ${successProb.toFixed(
          0
        )}% probability of achieving your financial goal of £${chartData.goal_analysis.target_value?.toLocaleString()}.`,
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
              <li>• Diversification helps manage risk</li>
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
              <li>• Review strategy annually</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Advanced Analytics */}
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
                {chartData.factor_importance?.length || 0}
              </div>
              <div className="text-sm text-gray-600">Factors Analyzed</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-800">
                {chartData.shap_waterfall?.contributions?.length || 0}
              </div>
              <div className="text-sm text-gray-600">SHAP Features</div>
            </div>
          </div>
        </div>
      )}
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

const getIconForInsight = (iconName) => {
  const icons = {
    Target: "🎯",
    Star: "⭐",
    TrendingUp: "📈",
    Scale: "⚖️",
    Waves: "🌊",
    Trophy: "🏆",
  };
  return icons[iconName] || "💡";
};

// Chart color constants
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
