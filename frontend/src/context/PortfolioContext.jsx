// PortfolioContext is responsible for managing and providing portfolio-related data
// including simulation results, user info, and token to all components within the dashboard.
import React, { createContext, useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

const PortfolioContext = createContext();

export const PortfolioProvider = ({ children }) => {
  const [portfolioData, _setPortfolioData] = useState(null);
  const [selectedSimulationId, setSelectedSimulationId] = useState(null);
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);

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

  // Set the portfolio data and compute a simplified breakdown of portfolio value
  const setPortfolioData = (data) => {
    console.log("📦 Full portfolioData received:", data);

    // Fallback logic: try nested, then results, then top-level data
    const Results = data?.results || data;
    const timeline = Results?.timeline || data?.timeline || {};
    const portfolioArray = timeline.portfolio || [];
    const stocksPicked = Results?.stocks_picked || data?.stocks_picked || [];

    console.log("🔍 Context debug:", {
      hasResults: !!Results,
      hasTimeline: !!timeline,
      portfolioArrayLength: portfolioArray.length,
      stocksPickedLength: stocksPicked.length,
      sampleTimelineEntry: portfolioArray[0],
      lastTimelineEntry: portfolioArray[portfolioArray.length - 1],
    });

    let breakdown = null;

    // Create breakdown from stocks_picked instead of timeline
    if (stocksPicked.length > 0) {
      console.log("📊 Creating breakdown from stocks_picked:", stocksPicked);

      breakdown = Object.fromEntries(
        stocksPicked.map((stock) => [
          stock.symbol,
          stock.final_value ||
            stock.current_value ||
            stock.shares * (stock.current_price || 0) ||
            0,
        ])
      );

      console.log("📈 Generated breakdown:", breakdown);
    } else if (portfolioArray.length > 0) {
      // If we have timeline data but no stocks_picked, try to extract from timeline
      const latestPortfolio = portfolioArray[portfolioArray.length - 1];
      console.log("📅 Latest portfolio entry:", latestPortfolio);

      // Check if timeline entry has stock data (some formats might have this)
      if (
        latestPortfolio &&
        typeof latestPortfolio === "object" &&
        !latestPortfolio.date
      ) {
        // This might be a stock breakdown object
        breakdown = Object.fromEntries(
          Object.entries(latestPortfolio)
            .filter(([key]) => key !== "date" && key !== "value")
            .map(([symbol, details]) => [
              symbol,
              details?.final_value ?? details?.value ?? 0,
            ])
        );
      } else {
        // Timeline only has date/value pairs, so create a simple breakdown
        // Use the total value and distribute among stocks if we know them
        const totalValue = latestPortfolio?.value || 0;
        if (totalValue > 0 && (data?.stocks_picked || Results?.stocks_picked)) {
          const stocks = data?.stocks_picked || Results?.stocks_picked || [];
          if (stocks.length > 0) {
            breakdown = Object.fromEntries(
              stocks.map((stock) => [
                stock.symbol,
                totalValue * (stock.allocation / 100),
              ])
            );
          }
        }
      }
    }

    // Enhanced data object with proper structure
    const enhancedData = {
      ...data,
      results: Results,
      ai_summary: data?.ai_summary || null,
      breakdown,
      stocks_picked: stocksPicked,

      // Ensure timeline is accessible at top level for calculations
      timeline: timeline,

      // Store final balance for easy access
      final_balance: (() => {
        // Try multiple sources for final balance
        if (Results?.end_value && Results.end_value > 0) {
          return Results.end_value;
        }
        if (portfolioArray.length > 0) {
          const lastEntry = portfolioArray[portfolioArray.length - 1];
          if (lastEntry?.value && lastEntry.value > 0) {
            return lastEntry.value;
          }
        }
        // Calculate from breakdown if available
        if (breakdown) {
          const totalFromBreakdown = Object.values(breakdown).reduce(
            (sum, val) => sum + (val || 0),
            0
          );
          if (totalFromBreakdown > 0) {
            return totalFromBreakdown;
          }
        }
        return 0;
      })(),
    };

    console.log("💾 Stored simulation data:", {
      simulationId: data?.id,
      userId: data?.user_id,
      fullData: "stored in portfolioData key",
      contextUpdated: true,
      finalBalance: enhancedData.final_balance,
      hasBreakdown: !!breakdown,
      breakdownTotal: breakdown
        ? Object.values(breakdown).reduce((sum, val) => sum + (val || 0), 0)
        : 0,
    });

    _setPortfolioData(enhancedData);
  };

  // Update the selected simulation ID and store it in local storage
  const updateSelectedSimulation = (id) => {
    localStorage.setItem("selectedSimulationId", id);
    setSelectedSimulationId(id);
  };

  const location = useLocation();

  // Detect simulation ID from URL path and update selectedSimulationId accordingly
  useEffect(() => {
    const pathParts = location.pathname.split("/");
    const routeSimulationId =
      pathParts.includes("dashboard") && pathParts.length >= 3
        ? pathParts[2]
        : null;

    if (routeSimulationId) {
      setSelectedSimulationId(routeSimulationId);
      localStorage.setItem("selectedSimulationId", routeSimulationId);
    }
  }, [location]);

  return (
    <PortfolioContext.Provider
      value={{
        portfolioData,
        setPortfolioData,
        updateSelectedSimulation,
        selectedSimulationId,
        token,
        user,
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
