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

    let breakdown = null;

    if (portfolioArray.length > 0) {
      const latestPortfolio = portfolioArray[portfolioArray.length - 1];

      breakdown = Object.fromEntries(
        Object.entries(latestPortfolio).map(([symbol, details]) => [
          symbol,
          details?.final_value ?? 0,
        ])
      );
    }

    _setPortfolioData({
      ...data,
      results: Results,
      ai_summary: data?.ai_summary || null,
      breakdown,
      stocks_picked: Results?.stocks_picked || [],
    });
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
