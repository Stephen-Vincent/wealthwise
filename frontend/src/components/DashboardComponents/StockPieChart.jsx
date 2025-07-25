import React, { useEffect, useState, useContext } from "react";
import { Chart } from "react-google-charts";
import PortfolioContext from "../../context/PortfolioContext";

// Utility to lighten or darken a hex color by a percent
// percent > 0: lighten, percent < 0: darken
function shadeColor(color, percent) {
  // Remove '#' if present
  let hex = color.replace(/^#/, "");
  if (hex.length === 3) {
    hex = hex
      .split("")
      .map((c) => c + c)
      .join("");
  }
  let num = parseInt(hex, 16);
  let r = (num >> 16) & 0xff;
  let g = (num >> 8) & 0xff;
  let b = num & 0xff;
  r = Math.min(255, Math.max(0, Math.round(r + (255 - r) * percent)));
  g = Math.min(255, Math.max(0, Math.round(g + (255 - g) * percent)));
  b = Math.min(255, Math.max(0, Math.round(b + (255 - b) * percent)));
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}
// Category configurations for styling and grouping
const CATEGORY_CONFIG = {
  bonds: {
    label: "Bonds & Fixed Income",
    color: "#10B981",
    bgColor: "#D1FAE5",
    description: "Low-risk investments that provide steady income",
  },
  dividend_stocks: {
    label: "Dividend Stocks",
    color: "#3B82F6",
    bgColor: "#DBEAFE",
    description: "Stable companies that pay regular dividends",
  },
  utilities: {
    label: "Utilities",
    color: "#F59E0B",
    bgColor: "#FEF3C7",
    description: "Essential services like electricity and water",
  },
  large_cap_growth: {
    label: "Large Cap Growth",
    color: "#8B5CF6",
    bgColor: "#EDE9FE",
    description: "Established companies with growth potential",
  },
  broad_market: {
    label: "Market ETFs",
    color: "#6366F1",
    bgColor: "#E0E7FF",
    description: "Diversified funds tracking broad market indices",
  },
  international: {
    label: "International",
    color: "#EC4899",
    bgColor: "#FCE7F3",
    description: "Non-U.S. developed market investments",
  },
  emerging_markets: {
    label: "Emerging Markets",
    color: "#EF4444",
    bgColor: "#FEE2E2",
    description: "Higher-growth developing country investments",
  },
  technology: {
    label: "Technology",
    color: "#06B6D4",
    bgColor: "#CFFAFE",
    description: "Technology and innovation-focused investments",
  },
  high_growth: {
    label: "High Growth",
    color: "#F97316",
    bgColor: "#FFEDD5",
    description: "High-risk, high-reward growth investments",
  },
  reits: {
    label: "Real Estate",
    color: "#84CC16",
    bgColor: "#ECFCCB",
    description: "Real estate investment trusts",
  },
  financials: {
    label: "Financial Services",
    color: "#64748B",
    bgColor: "#F1F5F9",
    description: "Banks, payment processors, and financial companies",
  },
  other: {
    label: "Other",
    color: "#9CA3AF",
    bgColor: "#F3F4F6",
    description: "Other or uncategorized investments",
  },
};

const DEFAULT_CATEGORY_CONFIG = {
  label: "Other",
  color: "#9CA3AF",
  bgColor: "#F3F4F6",
  description: "Other or uncategorized investments",
};

// Instrument service for FastAPI calls
class InstrumentService {
  constructor(baseURL = `${import.meta.env.VITE_API_URL}/api`) {
    this.baseURL = baseURL;
    this.cache = new Map();
  }

  async fetchInstruments(symbols = []) {
    // For specific symbols, use bulk endpoint
    if (symbols.length > 0 && symbols.length <= 50) {
      return this.fetchBulkInstruments(symbols);
    }

    // For general fetch, use the main endpoint
    const cacheKey = "all_instruments";
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    try {
      const response = await fetch(`${this.baseURL}/instruments?limit=1000`);
      if (!response.ok) throw new Error("Failed to fetch instruments");

      const result = await response.json();
      if (result.success) {
        this.cache.set(cacheKey, result.data);
        return result.data;
      }
      throw new Error(result.error || "Unknown error");
    } catch (error) {
      console.error("Error fetching instruments:", error);
      return this.getFallbackData(symbols);
    }
  }

  async fetchBulkInstruments(symbols) {
    const symbolsString = symbols.join(",");
    const cacheKey = `bulk_${symbolsString}`;

    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    try {
      const response = await fetch(
        `${this.baseURL}/instruments/bulk/${symbolsString}`
      );
      if (!response.ok) throw new Error("Failed to fetch bulk instruments");

      const result = await response.json();
      if (result.success) {
        this.cache.set(cacheKey, result.data);
        return result.data;
      }
      throw new Error("Failed to fetch bulk instruments");
    } catch (error) {
      console.error("Error fetching bulk instruments:", error);
      return this.getFallbackData(symbols);
    }
  }

  async searchInstruments(query) {
    try {
      const response = await fetch(
        `${this.baseURL}/instruments?search=${encodeURIComponent(
          query
        )}&limit=50`
      );
      if (!response.ok) throw new Error("Failed to search instruments");

      const result = await response.json();
      return result.success ? result.data : {};
    } catch (error) {
      console.error("Error searching instruments:", error);
      return {};
    }
  }

  async getInstrumentDetails(symbol, includePrice = false) {
    try {
      const response = await fetch(
        `${this.baseURL}/instruments/${symbol}?include_price=${includePrice}`
      );
      if (!response.ok) throw new Error("Failed to get instrument details");

      const result = await response.json();
      return result.success ? result.data : null;
    } catch (error) {
      console.error("Error getting instrument details:", error);
      return null;
    }
  }

  async getCategories() {
    try {
      const response = await fetch(
        `${this.baseURL}/instruments/categories/list`
      );
      if (!response.ok) throw new Error("Failed to get categories");

      const result = await response.json();
      return result.success ? result.categories : {};
    } catch (error) {
      console.error("Error getting categories:", error);
      return {};
    }
  }

  // Fallback data in case API is unavailable
  getFallbackData(symbols) {
    const fallbackInstruments = {};
    symbols.forEach((symbol) => {
      fallbackInstruments[symbol] = {
        name: symbol,
        type: "Unknown",
        category: "other",
        description: "Investment instrument",
        risk: "Unknown",
        icon: "📊",
      };
    });
    return fallbackInstruments;
  }
}

export default function StockPieChart() {
  const { portfolioData } = useContext(PortfolioContext);
  const stocksPicked = portfolioData?.results?.stocks_picked || [];
  const [selectedInstrument, setSelectedInstrument] = useState(null);
  const [instrumentDatabase, setInstrumentDatabase] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const instrumentService = new InstrumentService();

  // Fetch instrument data when component mounts or stocks change
  useEffect(() => {
    const fetchInstrumentData = async () => {
      if (stocksPicked.length === 0) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const symbols = stocksPicked.map((stock) => stock.symbol);
        const instruments = await instrumentService.fetchInstruments(symbols);
        setInstrumentDatabase(instruments);
      } catch (err) {
        setError("Failed to load instrument data");
        console.error("Error fetching instrument data:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchInstrumentData();
  }, [stocksPicked]);

  console.log("📊 StockPieChart stocks picked:", stocksPicked);

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-8 text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <h3 className="text-lg font-semibold text-gray-700 mb-2">
          Loading Portfolio Data
        </h3>
        <p className="text-gray-500">
          Fetching instrument information from Yahoo Finance...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-8 text-center">
        <div className="text-red-400 text-6xl mb-4">⚠️</div>
        <h3 className="text-lg font-semibold text-gray-700 mb-2">
          Error Loading Data
        </h3>
        <p className="text-gray-500 mb-4">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!stocksPicked.length) {
    return (
      <div className="bg-gray-50 rounded-lg p-6 text-center">
        <div className="text-gray-400 text-6xl mb-4">📊</div>
        <h3 className="text-lg font-semibold text-gray-700 mb-2">
          No Portfolio Allocation
        </h3>
        <p className="text-gray-500">
          Complete your portfolio setup to see your allocation breakdown.
        </p>
      </div>
    );
  }

  // Enrich stock data with instrument information
  const enrichedStocks = stocksPicked.map((stock) => ({
    ...stock,
    info: instrumentDatabase[stock.symbol] || {
      name: stock.symbol,
      type: "Unknown",
      category: "other",
      description: "Investment instrument",
      risk: "Unknown",
      icon: "📊",
    },
  }));

  // Group by category for summary
  const categoryTotals = enrichedStocks.reduce((acc, stock) => {
    const category = stock.info.category;
    if (!acc[category]) {
      acc[category] = { allocation: 0, count: 0, stocks: [] };
    }
    acc[category].allocation += stock.allocation;
    acc[category].count += 1;
    acc[category].stocks.push(stock);
    return acc;
  }, {});

  const rawValues = stocksPicked.map((stock) => stock.allocation);
  const total = rawValues.reduce((sum, val) => sum + val, 0);

  if (total === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-6 text-center">
        <p className="text-gray-500">
          Portfolio breakdown has zero total value.
        </p>
      </div>
    );
  }

  // --- Google Charts Pie Chart Data Preparation ---
  // Prepare chart data as [["Stock", "Allocation"], ...]
  const chartData = [
    ["Stock", "Allocation"],
    ...enrichedStocks.map((stock) => [
      `${stock.symbol} (${(stock.allocation * 100).toFixed(1)}%)`,
      stock.allocation * 100,
    ]),
  ];

  // Prepare colors for each slice using CATEGORY_CONFIG, alternate lighter/darker for 3D effect
  const chartColors = enrichedStocks.map((stock, idx) => {
    const category = stock.info?.category || "other";
    const baseColor =
      CATEGORY_CONFIG[category]?.color || DEFAULT_CATEGORY_CONFIG.color;
    // Odd-indexed slices: lighten; Even-indexed: darken (or vice versa)
    const percent = idx % 2 === 0 ? -0.09 : 0.13;
    return shadeColor(baseColor, percent);
  });

  // Google Charts options for 3D pie chart
  // Set legend position to "right" for better vertical display, and
  // set legend.maxLines to 99 to avoid legend pagination/cycling arrows.
  // See: https://developers.google.com/chart/interactive/docs/gallery/piechart#legend
  const chartOptions = {
    is3D: true,
    legend: {
      position: "right", // Place legend on right for vertical stacking
      alignment: "center",
      textStyle: { fontSize: 14, color: "#444" },
      maxLines: 99, // Show as many legend entries vertically as possible; avoids pagination arrows
      // This is intentionally set high for portfolios with many holdings.
    },
    pieSliceText: "none", // Hide labels on slices
    tooltip: {
      text: "percentage",
      textStyle: { fontSize: 14 },
      showColorCode: true,
    },
    colors: chartColors,
    backgroundColor: "transparent",
    chartArea: { left: 0, top: 24, width: "100%", height: "80%" },
    fontName: "Inter, sans-serif",
    pieStartAngle: 60,
    pieSliceBorderColor: "#e5e7eb",
    slices: {},
  };

  return (
    <div className="mb-6">
      <h3 className="text-xl font-semibold text-center mb-6 text-gray-800">
        Portfolio Allocation
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pie Chart Card - Google Charts 3D Pie */}
        <div
          className="rounded-2xl p-10 lg:p-14 shadow-2xl"
          style={{
            background: "rgba(255,255,255,0.92)",
            boxShadow:
              "0 8px 36px 0 rgba(31,38,135,0.14), 0 2px 12px 0 rgba(0,0,0,0.07)",
            border: "none",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            minHeight: 430,
          }}
        >
          {/* Google 3D Pie Chart */}
          <div
            className="w-full flex items-center justify-center"
            style={{ minHeight: 340, position: "relative" }}
          >
            {/* 
              Google PieChart - 3D, custom colors, legend at bottom, minimalist.
              Chart data and options are prepared above.
            */}
            <Chart
              chartType="PieChart"
              width="100%"
              height="340px"
              data={chartData}
              options={chartOptions}
              // onClick/selection events could be handled here if needed
            />
          </div>
          {/* Minimal, muted helper text below pie chart */}
        </div>

        {/* Category Summary & Holdings List */}
        <div className="space-y-4">
          {/* Category Summary */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h4 className="text-lg font-semibold mb-4 text-gray-800">
              Asset Categories
            </h4>
            <div className="space-y-3">
              {Object.entries(categoryTotals)
                .sort(([, a], [, b]) => b.allocation - a.allocation)
                .map(([category, data]) => {
                  const config =
                    CATEGORY_CONFIG[category] ?? DEFAULT_CATEGORY_CONFIG;
                  const percentage = (data.allocation * 100).toFixed(1);

                  return (
                    <div
                      key={category}
                      className="flex items-center justify-between"
                    >
                      <div className="flex items-center space-x-3">
                        <div
                          className="w-4 h-4 rounded-full"
                          style={{ backgroundColor: config.color }}
                        />
                        <div>
                          <span className="font-medium text-gray-800">
                            {config.label}
                          </span>
                          <p className="text-xs text-gray-500">
                            {config.description}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="font-semibold">{percentage}%</span>
                        <p className="text-xs text-gray-500">
                          {data.count} holding{data.count !== 1 ? "s" : ""}
                        </p>
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>

          {/* Individual Holdings */}
          <div className="bg-white rounded-xl shadow-lg p-6 max-h-80 overflow-y-auto">
            <h4 className="text-lg font-semibold mb-4 text-gray-800">
              Individual Holdings
            </h4>
            <div className="space-y-3">
              {enrichedStocks
                .sort((a, b) => b.allocation - a.allocation)
                .map((stock, index) => {
                  const percentage = (stock.allocation * 100).toFixed(1);
                  const config =
                    CATEGORY_CONFIG[stock.info?.category] ||
                    DEFAULT_CATEGORY_CONFIG;

                  return (
                    <div
                      key={stock.symbol}
                      className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors border"
                      onClick={() => setSelectedInstrument(stock)}
                      style={{
                        borderLeftColor: config.color,
                        borderLeftWidth: "4px",
                      }}
                    >
                      <div className="flex items-center space-x-3">
                        <span className="text-xl">{stock.info.icon}</span>
                        <div>
                          <div className="font-medium text-gray-800">
                            {stock.symbol}
                          </div>
                          <div className="text-sm text-gray-600 truncate max-w-32">
                            {stock.info.name}
                          </div>
                          <div className="text-xs text-gray-500">
                            {stock.info.type}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">{percentage}%</div>
                        <div className="text-xs text-gray-500">
                          {stock.info.risk} Risk
                        </div>
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Instrument Modal/Card */}
      {selectedInstrument && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          {/* Glassmorphism for modal */}
          <div
            className="rounded-xl max-w-md w-full p-6 shadow-2xl max-h-96 overflow-y-auto"
            style={{
              background: "rgba(255,255,255,0.95)",
              backdropFilter: "blur(10px)",
              WebkitBackdropFilter: "blur(10px)",
              border: "1.5px solid rgba(99,102,241,0.12)",
            }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">{selectedInstrument.info.icon}</span>
                <div>
                  <h3 className="text-xl font-bold text-gray-800">
                    {selectedInstrument.symbol}
                  </h3>
                  <p className="text-sm text-gray-600">
                    {selectedInstrument.info.type}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setSelectedInstrument(null)}
                className="text-gray-400 hover:text-gray-600 text-2xl font-bold leading-none"
              >
                ×
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">
                  {selectedInstrument.info.name}
                </h4>
                <p className="text-sm text-gray-600 leading-relaxed">
                  {selectedInstrument.info.description}
                </p>
              </div>

              <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Category:</span>
                  <span className="font-medium capitalize">
                    {CATEGORY_CONFIG[selectedInstrument.info.category]?.label ||
                      selectedInstrument.info.category}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Risk Level:</span>
                  <span className="font-medium">
                    {selectedInstrument.info.risk}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Portfolio Weight:</span>
                  <span className="font-medium">
                    {(selectedInstrument.allocation * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Additional metadata if available */}
                {selectedInstrument.info.metadata && (
                  <>
                    {selectedInstrument.info.metadata.sector && (
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Sector:</span>
                        <span className="font-medium">
                          {selectedInstrument.info.metadata.sector}
                        </span>
                      </div>
                    )}
                    {selectedInstrument.info.metadata.dividendYield && (
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Dividend Yield:</span>
                        <span className="font-medium">
                          {(
                            selectedInstrument.info.metadata.dividendYield * 100
                          ).toFixed(2)}
                          %
                        </span>
                      </div>
                    )}
                    {selectedInstrument.info.metadata.currentPrice && (
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Current Price:</span>
                        <span className="font-medium">
                          $
                          {selectedInstrument.info.metadata.currentPrice.toFixed(
                            2
                          )}
                        </span>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Category description */}
              <div className="bg-blue-50 rounded-lg p-3">
                <h5 className="font-medium text-blue-800 mb-1">
                  About{" "}
                  {CATEGORY_CONFIG[selectedInstrument.info.category]?.label ||
                    selectedInstrument.info.category}
                </h5>
                <p className="text-sm text-blue-700">
                  {CATEGORY_CONFIG[selectedInstrument.info.category]
                    ?.description || "Investment category information"}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
