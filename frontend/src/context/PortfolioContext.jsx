import React, { createContext, useState } from "react";

const PortfolioContext = createContext();

export const PortfolioProvider = ({ children }) => {
  const [portfolioData, _setPortfolioData] = useState(null);

  const setPortfolioData = (data) => {
    let breakdown = null;

    if (data && typeof data === "object") {
      if (data.breakdown && typeof data.breakdown === "object") {
        breakdown = data.breakdown;
      } else if (data.portfolio && typeof data.portfolio === "object") {
        const entries = Object.entries(data.portfolio).filter(
          ([, info]) =>
            info && typeof info === "object" && "final_value" in info
        );

        if (entries.length > 0) {
          breakdown = Object.fromEntries(
            entries.map(([symbol, info]) => [symbol, info.final_value ?? 0])
          );
        }
      }
    }

    console.log("✅ Breakdown computed:", breakdown);
    console.log("📦 Raw data received in setPortfolioData:", data);

    _setPortfolioData({
      ...data,
      breakdown,
    });
  };

  return (
    <PortfolioContext.Provider value={{ portfolioData, setPortfolioData }}>
      {children}
    </PortfolioContext.Provider>
  );
};

export default PortfolioContext;
