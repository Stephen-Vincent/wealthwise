import React from "react";
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

// 🔀 Utility to read either snake_case or camelCase
const pick = (obj, ...keys) => {
  for (const k of keys) {
    const parts = k.split(".");
    let val = obj;
    for (const p of parts) val = val?.[p];
    if (val !== undefined && val !== null) return val;
  }
  return undefined;
};

/**
 * 🔧 SHAP Debug + Recovery Utilities (integrated)
 */
export function debugShapDataFlow(portfolioData, location = "unknown") {
  console.log(`🔍 SHAP Debug at ${location}:`, {
    hasShapExplanations: portfolioData?.has_shap_explanations,
    wealthwiseEnhanced: portfolioData?.wealthwise_enhanced,
    resultsShapExplanation: portfolioData?.results?.shap_explanation,
    resultsHasShap: !!portfolioData?.results?.shap_explanation,
    shapInResults: Object.keys(portfolioData?.results || {}).filter((key) =>
      key.toLowerCase().includes("shap")
    ),
    allResultsKeys: Object.keys(portfolioData?.results || {}),
    humanReadableExplanations:
      portfolioData?.results?.shap_explanation?.human_readable_explanation,
    portfolioQualityScore:
      portfolioData?.results?.shap_explanation?.portfolio_quality_score,
    rawResultsObject: portfolioData?.results,
  });

  const searchForShap = (obj, path = "") => {
    if (!obj || typeof obj !== "object") return [];
    let findings = [];
    for (const [key, value] of Object.entries(obj)) {
      const currentPath = path ? `${path}.${key}` : key;
      if (key.toLowerCase().includes("shap")) {
        findings.push({ path: currentPath, value });
      }
      if (typeof value === "object" && value !== null) {
        findings = findings.concat(searchForShap(value, currentPath));
      }
    }
    return findings;
  };

  const shapFindings = searchForShap(portfolioData);
  console.log("🔍 All SHAP-related data found:", shapFindings);
  return shapFindings;
}

export function getShapExplanationData(portfolioData) {
  const possiblePaths = [
    portfolioData?.results?.shap_explanation,
    portfolioData?.shap_explanation,
    portfolioData?.results?.shapExplanation,
    portfolioData?.shapExplanation,
    portfolioData?.results?.shap_data,
    portfolioData?.shap_data,
    portfolioData?.results?.shapData,
    portfolioData?.shapData,
    portfolioData?.explanation,
    portfolioData?.results?.explanation,
  ];
  for (const path of possiblePaths) {
    if (path && typeof path === "object") {
      console.log("✅ Found SHAP data at path:", path);
      return path;
    }
  }
  console.log("❌ No SHAP data found in any expected location");
  return null;
}

export async function verifyBackendShapData(simulationId) {
  try {
    const response = await fetch(`/api/simulations/${simulationId}`);
    const data = await response.json();
    console.log("🔍 Backend SHAP Verification:", {
      simulationId,
      hasShapInResults: !!data?.results?.shap_explanation,
      hasShapFlag: data?.has_shap_explanations,
      wealthwiseEnhanced: data?.wealthwise_enhanced,
      rawResults: data?.results,
    });
    return data;
  } catch (error) {
    console.error("❌ Failed to verify backend SHAP data:", error);
    return null;
  }
}

export function recoverShapData(portfolioData) {
  const searchDeep = (obj, targetKey) => {
    if (!obj || typeof obj !== "object") return null;
    if (obj[targetKey]) return obj[targetKey];
    for (const value of Object.values(obj)) {
      if (typeof value === "object" && value !== null) {
        const found = searchDeep(value, targetKey);
        if (found) return found;
      }
    }
    return null;
  };

  const shapData =
    searchDeep(portfolioData, "shap_explanation") ||
    searchDeep(portfolioData, "shap_data") ||
    searchDeep(portfolioData, "explanation");

  if (shapData) {
    console.log("✅ Recovered SHAP data from deep search:", shapData);
    return shapData;
  }
  console.log("❌ Could not recover SHAP data");
  return null;
}

/**
 * 🔁 Hook: centralizes SHAP loading + debug + fallback fetching
 */
export function useShapDashboard(portfolioData) {
  const [shapData, setShapData] = React.useState(null);
  const [debugInfo, setDebugInfo] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);

  const hasIndicators = Boolean(
    pick(
      portfolioData,
      "has_shap_explanations",
      "hasShapExplanations",
      "hasShapExplanation"
    ) ||
      pick(portfolioData, "wealthwise_enhanced", "wealthwiseEnhanced") ||
      pick(portfolioData, "methodology", "analysisMethodology")
  );

  React.useEffect(() => {
    let isActive = true;
    async function run() {
      if (!portfolioData) return;

      // 1) Debug
      const dbg = debugShapDataFlow(portfolioData, "useShapDashboard");
      if (!isActive) return;
      setDebugInfo(dbg);

      // 2) Try inline
      let explanation = getShapExplanationData(portfolioData);

      // 3) Try recover
      if (!explanation) explanation = recoverShapData(portfolioData);

      // 4) If still nothing but indicators present, try dedicated endpoint
      if (!explanation && hasIndicators && portfolioData?.id) {
        try {
          setLoading(true);
          setError(null);
          const baseUrl =
            import.meta.env.VITE_API_URL ||
            "https://wealthwise-dwfq.onrender.com";
          const resp = await fetch(
            `${baseUrl}/shap/simulation/${portfolioData.id}/explanation`
          );
          if (resp.ok) {
            const json = await resp.json();
            explanation = json?.shap_data || json || null;
            console.log("✅ SHAP data fetched from dedicated endpoint:", json);
          } else {
            setError(`SHAP data not available (${resp.status})`);
            console.log(`❌ SHAP endpoint returned ${resp.status}`);
          }
        } catch (e) {
          setError(e?.message || "Failed to fetch SHAP data");
          console.error("❌ Error fetching SHAP data:", e);
        } finally {
          setLoading(false);
        }
      }

      // 5) Verify backend if still nothing
      if (!explanation && portfolioData?.id) {
        verifyBackendShapData(portfolioData.id);
      }

      if (!isActive) return;
      setShapData(explanation || null);
    }

    run();
    return () => {
      isActive = false;
    };
  }, [portfolioData, hasIndicators]);

  return {
    shapData,
    debugInfo,
    loading,
    error,
    hasShap:
      !!shapData ||
      Boolean(
        pick(
          portfolioData,
          "has_shap_explanations",
          "hasShapExplanations",
          "hasShapExplanation"
        )
      ),
    shouldShow:
      !!shapData ||
      Boolean(pick(portfolioData, "wealthwise_enhanced", "wealthwiseEnhanced")),
  };
}

/**
 * 🎛️ SHAP Dashboard (integrated with hook + utils)
 */
const SHAPDashboard = ({ portfolioData }) => {
  const [activeTab, setActiveTab] = React.useState("overview");
  const [chartData, setChartData] = React.useState([]);

  const hasResults = Boolean(portfolioData?.results);
  const isWealthWiseEnhanced = Boolean(
    pick(portfolioData, "wealthwise_enhanced", "wealthwiseEnhanced")
  );

  const { shapData, loading, error, hasShap, shouldShow } =
    useShapDashboard(portfolioData);

  const hasShapIndicators = Boolean(
    pick(
      portfolioData,
      "has_shap_explanations",
      "hasShapExplanations",
      "hasShapExplanation"
    ) || pick(portfolioData, "methodology", "analysisMethodology")
  );

  const hasShapData = Boolean(shapData);

  // 🔍 Component-level debug snapshot
  console.log("🔍 SHAPDashboard Debug:", {
    hasResults,
    isWealthWiseEnhanced,
    hasShapData,
    hasShapIndicators,
    methodology: pick(portfolioData, "methodology", "analysisMethodology"),
    hasShapExplanationsFlag: pick(
      portfolioData,
      "has_shap_explanations",
      "hasShapExplanations",
      "hasShapExplanation"
    ),
    portfolioId: portfolioData?.id,
    loading,
    error,
    shapDataKeys: shapData ? Object.keys(shapData) : null,
  });

  // Build chart data from SHAP feature_importance
  React.useEffect(() => {
    if (hasShapData) {
      const featureImportance =
        shapData.feature_importance || shapData.featureImportance || {};
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
    } else {
      setChartData([]);
    }
  }, [hasShapData, shapData]);

  // ✅ Loading state
  if (loading) {
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-6 text-center">
        <div className="text-4xl mb-2">⏳</div>
        <h3 className="text-lg font-bold text-blue-800 mb-2">
          Loading SHAP Analysis...
        </h3>
        <p className="text-blue-600">
          Fetching AI explanation data from the backend...
        </p>
      </div>
    );
  }

  // ✅ Error state (only when indicators imply SHAP exists)
  if (error && hasShapIndicators) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center">
        <div className="text-4xl mb-2">❌</div>
        <h3 className="text-lg font-bold text-red-800 mb-2">
          Failed to Load SHAP Data
        </h3>
        <p className="text-red-600 mb-4">Error: {error}</p>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          🔄 Retry
        </button>
      </div>
    );
  }

  // ✅ No portfolio results yet
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

  // ✅ Not WealthWise enhanced
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
        <BasicPortfolioInfo portfolioData={portfolioData} />
      </div>
    );
  }

  // ✅ Indicators say SHAP exists, but none returned yet
  if (!hasShapData && hasShapIndicators) {
    return (
      <div className="bg-orange-50 border border-orange-200 rounded-xl p-6">
        <div className="text-4xl mb-2 text-center">🧠</div>
        <h3 className="text-lg font-bold text-orange-800 mb-2 text-center">
          SHAP Analysis Generated (Data Missing from API)
        </h3>
        <p className="text-orange-600 mb-4 text-center">
          The backend generated SHAP explanations, but they weren't included in
          the API response yet.
        </p>
        <ShapIndicatorsPanel portfolioData={portfolioData} />
      </div>
    );
  }

  // ✅ AI analysis available but no SHAP at all
  if (!hasShapData) {
    return <AIAnalysisFallback portfolioData={portfolioData} />;
  }

  // ✅ Full SHAP dashboard
  return (
    <div className="w-full space-y-6">
      <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
        <div className="text-2xl mb-2">🎉</div>
        <h3 className="text-lg font-bold text-green-800 mb-2">
          SHAP AI Explanation Available!
        </h3>
        <p className="text-green-600">
          Confidence:{" "}
          {pick(
            shapData,
            "confidence_score",
            "confidence",
            "confidenceScore"
          ) || "N/A"}
          % | Method: {shapData.methodology || "SHAP Analysis"}
        </p>
      </div>

      {/* Tabs */}
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

      {process.env.NODE_ENV === "development" && (
        <details className="mt-2 p-4 bg-gray-100 rounded-lg">
          <summary className="cursor-pointer font-medium text-gray-700">
            Debug Info
          </summary>
          <pre className="mt-2 text-xs text-gray-600 overflow-auto">
            {JSON.stringify(
              {
                hasShapExplanations: portfolioData?.has_shap_explanations,
                wealthwiseEnhanced: portfolioData?.wealthwise_enhanced,
                shapKeys: shapData ? Object.keys(shapData) : [],
              },
              null,
              2
            )}
          </pre>
        </details>
      )}
    </div>
  );
};

/**
 * 🧱 Subcomponents & helpers
 */
const BasicPortfolioInfo = ({ portfolioData }) => (
  <div className="bg-white rounded-lg p-4 mt-4 text-left">
    <h4 className="font-semibold mb-2">Portfolio Information:</h4>
    <div className="space-y-2 text-sm">
      <p>
        <strong>Portfolio ID:</strong> {portfolioData.id}
      </p>
      <p>
        <strong>Goal:</strong>{" "}
        {pick(portfolioData, "goal", "investment_goal", "investmentGoal")}
      </p>
      <p>
        <strong>Target Value:</strong> £
        {pick(portfolioData, "target_value", "targetValue")?.toLocaleString?.()}
      </p>
      <p>
        <strong>Risk Score:</strong>{" "}
        {pick(portfolioData, "risk_score", "riskScore")}/100
      </p>
      <p>
        <strong>Risk Label:</strong>{" "}
        {pick(portfolioData, "risk_label", "riskLabel")}
      </p>
      {pick(portfolioData, "results.stocks_picked", "results.stocksPicked") && (
        <p>
          <strong>Stocks Selected:</strong>{" "}
          {pick(
            portfolioData,
            "results.stocks_picked.length",
            "results.stocksPicked.length"
          )}{" "}
          stocks
        </p>
      )}
    </div>
    <div className="mt-4 text-xs text-yellow-600">
      <p>
        <strong>Available keys:</strong> {Object.keys(portfolioData).join(", ")}
      </p>
    </div>
  </div>
);

const ShapIndicatorsPanel = ({ portfolioData }) => (
  <div className="bg-white rounded-lg p-4 mb-4">
    <h4 className="font-semibold mb-2 flex items-center">
      <span className="mr-2">🔍</span>
      SHAP Analysis Indicators
    </h4>
    <div className="text-sm space-y-1">
      <p>
        <strong>Has SHAP Explanations:</strong>{" "}
        {portfolioData?.has_shap_explanations ? "✅ Yes" : "❌ No"}
      </p>
      <p>
        <strong>Analysis Method:</strong>{" "}
        {portfolioData?.methodology || "Not specified"}
      </p>
      <p>
        <strong>WealthWise Enhanced:</strong>{" "}
        {portfolioData?.wealthwise_enhanced ? "✅ Yes" : "❌ No"}
      </p>
    </div>
  </div>
);

const AIAnalysisFallback = ({ portfolioData }) => {
  const aiSummary = pick(portfolioData, "ai_summary", "aiSummary");
  const riskExplanation = pick(
    portfolioData,
    "risk_explanation",
    "riskExplanation"
  );
  const allocationGuidance = pick(
    portfolioData,
    "allocation_guidance",
    "allocationGuidance"
  );
  const riskScore = pick(portfolioData, "risk_score", "riskScore");
  const riskLabel = pick(portfolioData, "risk_label", "riskLabel");
  const targetValue = pick(portfolioData, "target_value", "targetValue");
  const timeframe = pick(
    portfolioData,
    "timeframe",
    "timeHorizon",
    "timeHorizonYears"
  );
  const stocksPicked =
    pick(portfolioData, "results.stocks_picked", "results.stocksPicked") || [];

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
      <div className="text-4xl mb-2 text-center">🧠</div>
      <h3 className="text-lg font-bold text-blue-800 mb-2 text-center">
        AI Analysis Available (No SHAP Data)
      </h3>
      <p className="text-blue-600 mb-4 text-center">
        While detailed SHAP explanations aren't available, we have AI-generated
        insights for your portfolio.
      </p>

      <div className="space-y-4">
        {aiSummary && (
          <div className="bg-white rounded-lg p-4">
            <h4 className="font-semibold mb-2 flex items-center">
              <span className="mr-2">🤖</span>
              AI Summary
            </h4>
            <p className="text-gray-700 text-sm leading-relaxed">{aiSummary}</p>
          </div>
        )}

        {riskExplanation && (
          <div className="bg-white rounded-lg p-4">
            <h4 className="font-semibold mb-2 flex items-center">
              <span className="mr-2">⚠️</span>
              Risk Explanation
            </h4>
            <p className="text-gray-700 text-sm leading-relaxed">
              {riskExplanation}
            </p>
          </div>
        )}

        {allocationGuidance && (
          <div className="bg-white rounded-lg p-4">
            <h4 className="font-semibold mb-2 flex items-center">
              <span className="mr-2">📊</span>
              Allocation Guidance
            </h4>
            <p className="text-gray-700 text-sm leading-relaxed">
              {allocationGuidance}
            </p>
          </div>
        )}

        <div className="bg-white rounded-lg p-4">
          <h4 className="font-semibold mb-2 flex items-center">
            <span className="mr-2">📈</span>
            Portfolio Metrics
          </h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium">Risk Score:</span>
              <span className="ml-2">{riskScore}/100</span>
            </div>
            <div>
              <span className="font-medium">Risk Level:</span>
              <span className="ml-2">{riskLabel}</span>
            </div>
            <div>
              <span className="font-medium">Target:</span>
              <span className="ml-2">£{targetValue?.toLocaleString?.()}</span>
            </div>
            <div>
              <span className="font-medium">Timeframe:</span>
              <span className="ml-2">{timeframe} years</span>
            </div>
          </div>
        </div>

        {stocksPicked.length > 0 && (
          <div className="bg-white rounded-lg p-4">
            <h4 className="font-semibold mb-2 flex items-center">
              <span className="mr-2">📋</span>
              Selected Stocks ({stocksPicked.length})
            </h4>
            <div className="grid grid-cols-1 gap-2 text-sm">
              {stocksPicked.slice(0, 5).map((stock, index) => (
                <div key={index} className="flex justify-between items-center">
                  <span className="font-medium">{stock.symbol}</span>
                  <span>
                    {stock.allocation
                      ? `${(stock.allocation * 100).toFixed(1)}%`
                      : "N/A"}
                  </span>
                </div>
              ))}
              {stocksPicked.length > 5 && (
                <p className="text-gray-500 italic">
                  ... and {stocksPicked.length - 5} more
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      <button
        onClick={() => console.log("🔍 Full Portfolio Data:", portfolioData)}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 w-full"
      >
        🔍 Debug Portfolio Data
      </button>
    </div>
  );
};

const OverviewTab = ({ shapData, portfolioData }) => {
  const confidence =
    pick(shapData, "confidence_score", "confidence", "confidenceScore") || 75;
  const methodology = shapData?.methodology || "SHAP Analysis";
  const explanation =
    pick(
      shapData,
      "explanation",
      "human_readable_explanation.summary",
      "humanReadableExplanation.summary"
    ) ||
    pick(portfolioData, "ai_summary", "aiSummary") ||
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
      title: "Portfolio Quality",
      value:
        pick(shapData, "portfolio_quality_score", "portfolioQualityScore") ||
        85,
      maxValue: 100,
      unit: "/100",
      color: "text-blue-600",
      bgColor: "bg-blue-50",
      icon: "🧠",
    },
    {
      title: "Risk Score",
      value: pick(portfolioData, "risk_score", "riskScore") || 50,
      maxValue: 100,
      unit: "/100",
      color: "text-orange-600",
      bgColor: "bg-orange-50",
      icon: "📊",
    },
    {
      title: "Target Progress",
      value: pick(portfolioData, "target_achieved", "targetAchieved")
        ? 100
        : 75,
      maxValue: 100,
      unit: "%",
      color: "text-purple-600",
      bgColor: "bg-purple-50",
      icon: "🎯",
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
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-2">🧠</span>
          AI Decision Summary
        </h3>

        <div className="bg-blue-50 rounded-lg p-4 border-l-4 border-blue-500 mb-4">
          <div className="font-semibold text-blue-800 mb-2">
            Strategy Overview
          </div>
          <p className="text-gray-700 leading-relaxed">{explanation}</p>
        </div>

        {pick(portfolioData, "risk_explanation", "riskExplanation") && (
          <div className="bg-yellow-50 rounded-lg p-4 border-l-4 border-yellow-500 mb-4">
            <div className="font-semibold text-yellow-800 mb-2">
              Risk Analysis
            </div>
            <p className="text-gray-700 leading-relaxed">
              {pick(portfolioData, "risk_explanation", "riskExplanation")}
            </p>
          </div>
        )}

        {pick(portfolioData, "allocation_guidance", "allocationGuidance") && (
          <div className="bg-green-50 rounded-lg p-4 border-l-4 border-green-500 mb-4">
            <div className="font-semibold text-green-800 mb-2">
              Allocation Guidance
            </div>
            <p className="text-gray-700 leading-relaxed">
              {pick(portfolioData, "allocation_guidance", "allocationGuidance")}
            </p>
          </div>
        )}

        {shapData?.human_readable_explanation && (
          <div className="grid gap-4">
            {Object.entries(shapData.human_readable_explanation).map(
              ([key, text], index) => (
                <div
                  key={index}
                  className="bg-gray-50 rounded-lg p-4 border-l-4 border-gray-400"
                >
                  <div className="font-semibold text-gray-800 mb-2">
                    {formatFactorName(key)}
                  </div>
                  <p className="text-gray-600 leading-relaxed">{text}</p>
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

const FactorsTab = ({ chartData, shapData }) => {
  return (
    <div className="space-y-6">
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

const InsightsTab = ({ shapData, portfolioData }) => {
  return (
    <div className="space-y-6">
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

/** Helpers */
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
