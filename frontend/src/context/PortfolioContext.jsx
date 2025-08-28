/**
 * PortfolioContext.jsx
 *
 * This file defines a React Context and Provider for managing portfolio-related state,
 * handling API calls, and storing simulation data across the WealthWise app.
 * It centralizes the logic for fetching, updating, and providing portfolio and simulation data,
 * as well as user authentication and loading/error states, making this information
 * accessible to any component in the app tree.
 */
// Enhanced PortfolioContext - Centralized data management for all portfolio-related data

import { createContext, useEffect, useState, useCallback } from "react";
import { useLocation } from "react-router-dom";

const PortfolioContext = createContext();

export const PortfolioProvider = ({ children }) => {
  const [portfolioData, _setPortfolioData] = useState(null);
  const [selectedSimulationId, setSelectedSimulationId] = useState(null);
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // API base URL
  const apiBase = import.meta.env.VITE_API_URL || "/api";

  // Load token and user information from local storage on initial render
  useEffect(() => {
    const storedToken = localStorage.getItem("access_token");
    const storedUser = localStorage.getItem("user");
    if (storedToken) {
      setToken(storedToken);
    }
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch (e) {
        console.error("Error parsing stored user:", e);
      }
    }
  }, []);

  // Centralized API call function
  const apiCall = useCallback(
    async (endpoint, options = {}) => {
      const url = `${apiBase.replace(/\/+$/, "")}${endpoint}`;
      const defaultOptions = {
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          ...(token && { Authorization: `Bearer ${token}` }),
        },
        credentials: "include",
        ...options,
      };

      console.log(`API Call: ${options.method || "GET"} ${url}`);

      try {
        const response = await fetch(url, defaultOptions);

        if (!response.ok) {
          throw new Error(
            `API call failed: ${response.status} ${response.statusText}`
          );
        }

        const data = await response.json();
        console.log(`API Response (${endpoint}):`, data);
        return data;
      } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
      }
    },
    [apiBase, token]
  );

  // Fetch complete simulation data with all related information
  const fetchSimulationData = useCallback(
    async (simulationId) => {
      if (!simulationId) return null;

      setLoading(true);
      setError(null);

      try {
        console.log(`Fetching complete data for simulation ${simulationId}`);

        // Fetch main simulation data
        const simulationData = await apiCall(`/simulations/${simulationId}`);

        // --- Normalize AI summary across possible shapes ---
        const normalizedAiSummary =
          simulationData?.ai_summary ||
          simulationData?.results?.ai_analysis?.ai_summary ||
          simulationData?.results?.ai_analysis?.summary ||
          simulationData?.results?.education?.summary ||
          "";

        // --- Backfill results.ai_analysis.ai_summary for consumers that expect it ---
        if (normalizedAiSummary) {
          if (!simulationData.results) simulationData.results = {};
          if (!simulationData.results.ai_analysis)
            simulationData.results.ai_analysis = {};
          if (!simulationData.results.ai_analysis.ai_summary)
            simulationData.results.ai_analysis.ai_summary = normalizedAiSummary;
        }

        // Attach normalized summary if not present at top-level
        if (!simulationData.ai_summary && normalizedAiSummary) {
          simulationData.ai_summary = normalizedAiSummary;
        }

        // Check if we already have chart_data in the simulation response
        if (simulationData.chart_data) {
          console.log("Chart data found in simulation response");
          console.log(
            "🔎 Normalized ai_summary (pre-return):",
            simulationData.ai_summary
              ? simulationData.ai_summary.slice(0, 140) + "…"
              : "(empty)"
          );
          return simulationData;
        }

        // If no chart_data, try to fetch it separately
        let chartData = null;
        let enhancedData = null;

        try {
          console.log("Fetching additional chart data...");
          const chartResponse = await apiCall(
            `/shap-visualization/simulation/${simulationId}/chart-data`
          );
          if (chartResponse.success && chartResponse.chart_data) {
            chartData = chartResponse.chart_data;
          }
        } catch (chartError) {
          console.warn("Chart data fetch failed:", chartError.message);
        }

        try {
          console.log("Fetching enhanced data...");
          const enhancedResponse = await apiCall(
            `/shap-visualization/simulation/${simulationId}/enhanced-data`
          );
          if (enhancedResponse.success && enhancedResponse.enhanced_data) {
            enhancedData = enhancedResponse.enhanced_data;
          }
        } catch (enhancedError) {
          console.warn("Enhanced data fetch failed:", enhancedError.message);
        }

        // Build a minimal chart_data fallback if needed
        if (!chartData && simulationData?.results?.shap_explanation) {
          const se = simulationData.results.shap_explanation;
          chartData = {
            feature_importance: se.feature_importance || [],
            portfolio_composition: se.portfolio_composition || [],
            shap_waterfall: se.shap_waterfall || [],
            performance_timeline:
              simulationData?.results?.portfolio_metrics?.timeline_data ||
              simulationData?.results?.timeline ||
              [],
          };
          console.log(
            "🧩 Constructed minimal chart_data from shap_explanation"
          );
        }

        // Merge all data together
        const completeData = {
          ...simulationData,
          ai_summary: simulationData.ai_summary ?? normalizedAiSummary ?? "",
          // keep nested ai_analysis summary for older components
          results: {
            ...(simulationData.results || {}),
            ai_analysis: {
              ...((simulationData.results &&
                simulationData.results.ai_analysis) ||
                {}),
              ai_summary:
                (simulationData.results &&
                  simulationData.results.ai_analysis &&
                  simulationData.results.ai_analysis.ai_summary) ||
                normalizedAiSummary ||
                "",
            },
          },
          ...(chartData && { chart_data: chartData }),
          ...(enhancedData && { enhanced_data: enhancedData }),
        };
        console.log(
          "🔎 Final ai_summary length:",
          completeData.ai_summary ? completeData.ai_summary.length : 0
        );

        console.log("Complete simulation data assembled:", {
          hasSimulation: true,
          hasChartData: !!completeData.chart_data,
          hasEnhancedData: !!completeData.enhanced_data,
          chartDataKeys: completeData.chart_data
            ? Object.keys(completeData.chart_data)
            : "None",
        });

        return completeData;
      } catch (error) {
        console.error("Failed to fetch simulation data:", error);
        setError(error.message);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [apiCall]
  );

  // Enhanced setPortfolioData that preserves all data structures
  const setPortfolioData = useCallback((data) => {
    console.log("Setting portfolio data:", data);

    if (!data) {
      _setPortfolioData(null);
      return;
    }

    // Add explicit logging for chart_data preservation
    console.log("Chart data check:", {
      hasChartData: !!data?.chart_data,
      chartDataKeys: data?.chart_data ? Object.keys(data.chart_data) : "None",
      hasResults: !!data?.results,
      hasEnhancedData: !!data?.enhanced_data,
    });

    // Fallback logic: try nested, then results, then top-level data
    const Results = data?.results || data;

    // Ensure older components find the AI summary in results.ai_analysis
    const normalizedAi =
      data?.ai_summary ||
      data?.results?.ai_analysis?.ai_summary ||
      data?.results?.ai_analysis?.summary ||
      data?.results?.education?.summary ||
      null;

    if (normalizedAi) {
      if (!Results.ai_analysis) Results.ai_analysis = {};
      if (!Results.ai_analysis.ai_summary)
        Results.ai_analysis.ai_summary = normalizedAi;
    }

    const timeline = Results?.timeline || data?.timeline || {};
    const portfolioArray = timeline.portfolio || [];
    const stocksPicked = Results?.stocks_picked || data?.stocks_picked || [];

    let breakdown = null;

    // Create breakdown from stocks_picked using allocation field
    if (stocksPicked.length > 0) {
      console.log("Creating breakdown from stocks_picked:", stocksPicked);

      breakdown = Object.fromEntries(
        stocksPicked.map((stock) => [stock.symbol, stock.allocation || 0])
      );

      console.log("Generated breakdown:", breakdown);
    }

    // Use backend breakdown if available, otherwise use computed breakdown
    const finalBreakdown = data?.breakdown || breakdown;

    // Enhanced data object with proper structure - PRESERVE ALL ORIGINAL DATA
    const enhancedData = {
      ...data, // Keep all original fields including chart_data
      results: Results,
      ai_summary:
        data?.ai_summary ||
        data?.results?.ai_analysis?.ai_summary ||
        data?.results?.ai_analysis?.summary ||
        data?.results?.education?.summary ||
        null,
      breakdown: finalBreakdown,
      stocks_picked: stocksPicked,
      timeline: timeline,
      final_balance:
        data?.final_balance ||
        data?.performance_metrics?.ending_value ||
        Results?.portfolio_metrics?.ending_value ||
        0,

      // Explicitly ensure these are preserved
      chart_data: data?.chart_data || null,
      enhanced_data: data?.enhanced_data || null,
    };

    console.log("Enhanced data structure:", {
      simulationId: enhancedData?.id,
      hasChartData: !!enhancedData.chart_data,
      hasEnhancedData: !!enhancedData.enhanced_data,
      chartDataKeys: enhancedData.chart_data
        ? Object.keys(enhancedData.chart_data)
        : "None",
      finalBalance: enhancedData.final_balance,
      hasBreakdown: !!finalBreakdown,
    });

    console.log(
      "📝 ai_summary present:",
      !!enhancedData.ai_summary,
      "length:",
      enhancedData.ai_summary ? enhancedData.ai_summary.length : 0
    );

    _setPortfolioData(enhancedData);
  }, []);

  // Load simulation data when selectedSimulationId changes
  useEffect(() => {
    if (selectedSimulationId && token) {
      console.log(`Loading data for simulation ${selectedSimulationId}`);
      fetchSimulationData(selectedSimulationId).then((data) => {
        if (data) {
          setPortfolioData(data);
        }
      });
    }
  }, [selectedSimulationId, token, fetchSimulationData, setPortfolioData]);

  // Update the selected simulation ID and store it in local storage
  const updateSelectedSimulation = useCallback(
    (id) => {
      localStorage.setItem("selectedSimulationId", id);
      setSelectedSimulationId(id);
      // cache any existing summary (if switching within session)
      if (portfolioData?.ai_summary) {
        localStorage.setItem("cached_ai_summary", portfolioData.ai_summary);
      }
    },
    [portfolioData]
  );

  // Refresh current simulation data
  const refreshPortfolioData = useCallback(async () => {
    if (selectedSimulationId) {
      const data = await fetchSimulationData(selectedSimulationId);
      if (data) {
        setPortfolioData(data);
      }
    }
  }, [selectedSimulationId, fetchSimulationData, setPortfolioData]);

  const location = useLocation();

  // Detect simulation ID from URL path and update selectedSimulationId accordingly
  useEffect(() => {
    const pathParts = location.pathname.split("/");
    const routeSimulationId =
      pathParts.includes("dashboard") && pathParts.length >= 3
        ? pathParts[2]
        : null;

    if (routeSimulationId && routeSimulationId !== selectedSimulationId) {
      setSelectedSimulationId(routeSimulationId);
      localStorage.setItem("selectedSimulationId", routeSimulationId);
    }
  }, [location, selectedSimulationId, setSelectedSimulationId]);

  // Derived data selectors - components can use these instead of parsing portfolioData themselves
  const shapData =
    portfolioData?.chart_data ||
    portfolioData?.results?.shap_explanation ||
    null;
  const hasShapData = shapData && Object.keys(shapData).length > 0;
  const chartData = portfolioData?.chart_data || null;
  const enhancedData = portfolioData?.enhanced_data || null;
  const aiSummary =
    portfolioData?.ai_summary ||
    portfolioData?.results?.ai_analysis?.ai_summary ||
    portfolioData?.results?.ai_analysis?.summary ||
    portfolioData?.results?.education?.summary ||
    "";

  const hasAiSummary = !!(
    aiSummary &&
    typeof aiSummary === "string" &&
    aiSummary.trim().length > 0
  );

  return (
    <PortfolioContext.Provider
      value={{
        // Core data
        portfolioData,
        setPortfolioData,

        // Selection management
        selectedSimulationId,
        updateSelectedSimulation,

        // Authentication
        token,
        user,

        // Loading states
        loading,
        error,

        // Data fetching functions
        fetchSimulationData,
        refreshPortfolioData,
        apiCall,

        // Derived data selectors
        shapData,
        hasShapData,
        chartData,
        enhancedData,
        aiSummary,
        hasAiSummary,
      }}
    >
      {children}
    </PortfolioContext.Provider>
  );
};

import { useContext } from "react";
const usePortfolio = () => useContext(PortfolioContext);

export { PortfolioContext, usePortfolio };
export default PortfolioContext;
