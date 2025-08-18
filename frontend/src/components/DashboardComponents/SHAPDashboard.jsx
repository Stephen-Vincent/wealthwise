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

  // === your existing logic, unchanged ===
  const shapData = useMemo(() => {
    return (
      portfolioData?.results?.shap_explanation ||
      portfolioData?.results?.shap_explanations ||
      portfolioData?.shap_explanation ||
      null
    );
  }, [portfolioData]);
  const hasShapData = useMemo(
    () => Object.keys(shapData || {}).length > 0,
    [shapData]
  );

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

  // ... (keep the rest of your component exactly as you have it)
  // SummaryTab / FactorsTab / InsightsTab / helpers unchanged below
  // (omitted here for brevity)
  return (
    <div className="w-full space-y-6">
      {/* header / tabs / content exactly like your current version */}
      {/* ... */}
    </div>
  );
};

// ===== helpers (unchanged) =====
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
