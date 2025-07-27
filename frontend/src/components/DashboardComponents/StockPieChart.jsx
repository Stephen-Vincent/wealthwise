import React, { useEffect, useState, useContext } from "react";
import { Chart } from "react-google-charts";
import PortfolioContext from "../../context/PortfolioContext";
import {
  enhanceStockData,
  categorizeStocks,
  CATEGORY_INFO,
} from "../../utils/stockMapping";

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

// Enhanced category configurations using stockMapping.js data
const ENHANCED_CATEGORY_CONFIG = {
  Technology: {
    label: "Technology",
    color: "#3B82F6",
    bgColor: "#DBEAFE",
    description: "Technology and innovation-focused investments",
    icon: "💻",
  },
  "Large Cap Growth": {
    label: "Large Cap Growth",
    color: "#8B5CF6",
    bgColor: "#EDE9FE",
    description: "Established companies with growth potential",
    icon: "📈",
  },
  "Emerging Markets": {
    label: "Emerging Markets",
    color: "#EF4444",
    bgColor: "#FEE2E2",
    description: "Higher-growth developing country investments",
    icon: "🌍",
  },
  Healthcare: {
    label: "Healthcare",
    color: "#10B981",
    bgColor: "#D1FAE5",
    description: "Healthcare and biotechnology companies",
    icon: "🏥",
  },
  "Digital Assets": {
    label: "Digital Assets",
    color: "#8B5CF6",
    bgColor: "#EDE9FE",
    description: "Cryptocurrency and digital asset exposure",
    icon: "₿",
  },
  "Broad Market": {
    label: "Market ETFs",
    color: "#6366F1",
    bgColor: "#E0E7FF",
    description: "Diversified funds tracking broad market indices",
    icon: "📊",
  },
  "Large Cap": {
    label: "Large Cap",
    color: "#059669",
    bgColor: "#D1FAE5",
    description: "Large established company investments",
    icon: "🏢",
  },
  Bonds: {
    label: "Bonds & Fixed Income",
    color: "#0D9488",
    bgColor: "#CCFBF1",
    description: "Fixed income and bond investments",
    icon: "📋",
  },
  Other: {
    label: "Other",
    color: "#9CA3AF",
    bgColor: "#F3F4F6",
    description: "Other or uncategorized investments",
    icon: "❓",
  },
};

const DEFAULT_CATEGORY_CONFIG = {
  label: "Other",
  color: "#9CA3AF",
  bgColor: "#F3F4F6",
  description: "Other or uncategorized investments",
  icon: "❓",
};

export default function StockPieChart() {
  const { portfolioData } = useContext(PortfolioContext);
  const stocksPicked = portfolioData?.results?.stocks_picked || [];
  const [selectedInstrument, setSelectedInstrument] = useState(null);
  const [enhancedStocks, setEnhancedStocks] = useState([]);
  const [categoryTotals, setCategoryTotals] = useState({});
  const [loading, setLoading] = useState(true);

  // Process stock data using stockMapping.js
  useEffect(() => {
    if (stocksPicked.length === 0) {
      setLoading(false);
      return;
    }

    try {
      // Use stockMapping.js to enhance stock data
      const enhanced = enhanceStockData(stocksPicked);
      setEnhancedStocks(enhanced);

      // Calculate category totals
      const categories = enhanced.reduce((acc, stock) => {
        const category = stock.category || "Other";
        if (!acc[category]) {
          acc[category] = {
            allocation: 0,
            count: 0,
            stocks: [],
            info: ENHANCED_CATEGORY_CONFIG[category] || DEFAULT_CATEGORY_CONFIG,
          };
        }
        acc[category].allocation += stock.allocation || 0;
        acc[category].count += 1;
        acc[category].stocks.push(stock);
        return acc;
      }, {});

      setCategoryTotals(categories);
      setLoading(false);
    } catch (error) {
      console.error("Error processing stock data:", error);
      setLoading(false);
    }
  }, [stocksPicked]);

  console.log("📊 StockPieChart enhanced stocks:", enhancedStocks);
  console.log("📊 Category totals:", categoryTotals);

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-8 text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <h3 className="text-lg font-semibold text-gray-700 mb-2">
          Loading Portfolio Data
        </h3>
        <p className="text-gray-500">Processing stock allocation data...</p>
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

  const rawValues = enhancedStocks.map((stock) => stock.allocation || 0);
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
    ...enhancedStocks.map((stock) => [
      `${stock.symbol} (${((stock.allocation || 0) * 100).toFixed(1)}%)`,
      (stock.allocation || 0) * 100,
    ]),
  ];

  // Prepare colors for each slice using enhanced category config
  const chartColors = enhancedStocks.map((stock, idx) => {
    const category = stock.category || "Other";
    const categoryConfig =
      ENHANCED_CATEGORY_CONFIG[category] || DEFAULT_CATEGORY_CONFIG;
    const baseColor = categoryConfig.color;
    // Alternate lighter/darker for 3D effect
    const percent = idx % 2 === 0 ? -0.09 : 0.13;
    return shadeColor(baseColor, percent);
  });

  // Google Charts options for 3D pie chart
  const chartOptions = {
    is3D: true,
    legend: {
      position: "right",
      alignment: "center",
      textStyle: { fontSize: 14, color: "#444" },
      maxLines: 99,
    },
    pieSliceText: "none",
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
          <div
            className="w-full flex items-center justify-center"
            style={{ minHeight: 340, position: "relative" }}
          >
            <Chart
              chartType="PieChart"
              width="100%"
              height="340px"
              data={chartData}
              options={chartOptions}
            />
          </div>
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
                  const config = data.info;
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
              {enhancedStocks
                .sort((a, b) => (b.allocation || 0) - (a.allocation || 0))
                .map((stock, index) => {
                  const percentage = ((stock.allocation || 0) * 100).toFixed(1);
                  const categoryConfig =
                    ENHANCED_CATEGORY_CONFIG[stock.category] ||
                    DEFAULT_CATEGORY_CONFIG;

                  return (
                    <div
                      key={stock.symbol}
                      className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors border"
                      onClick={() => setSelectedInstrument(stock)}
                      style={{
                        borderLeftColor: categoryConfig.color,
                        borderLeftWidth: "4px",
                      }}
                    >
                      <div className="flex items-center space-x-3">
                        <span className="text-xl">{categoryConfig.icon}</span>
                        <div>
                          <div className="font-medium text-gray-800">
                            {stock.symbol}
                          </div>
                          <div className="text-sm text-gray-600 truncate max-w-32">
                            {stock.name || stock.symbol}
                          </div>
                          <div className="text-xs text-gray-500">
                            {stock.category || "Other"}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">{percentage}%</div>
                        <div className="text-xs text-gray-500">
                          {stock.riskLevel || "Unknown"} Risk
                        </div>
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Instrument Modal */}
      {selectedInstrument && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
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
                <span className="text-2xl">
                  {
                    (
                      ENHANCED_CATEGORY_CONFIG[selectedInstrument.category] ||
                      DEFAULT_CATEGORY_CONFIG
                    ).icon
                  }
                </span>
                <div>
                  <h3 className="text-xl font-bold text-gray-800">
                    {selectedInstrument.symbol}
                  </h3>
                  <p className="text-sm text-gray-600">
                    {selectedInstrument.category || "Other"}
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
                  {selectedInstrument.name || selectedInstrument.symbol}
                </h4>
                <p className="text-sm text-gray-600 leading-relaxed">
                  {selectedInstrument.description ||
                    "Investment instrument information"}
                </p>
              </div>

              <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Category:</span>
                  <span className="font-medium">
                    {
                      (
                        ENHANCED_CATEGORY_CONFIG[selectedInstrument.category] ||
                        DEFAULT_CATEGORY_CONFIG
                      ).label
                    }
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Risk Level:</span>
                  <span className="font-medium">
                    {selectedInstrument.riskLevel || "Unknown"}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Portfolio Weight:</span>
                  <span className="font-medium">
                    {((selectedInstrument.allocation || 0) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Category description */}
              <div className="bg-blue-50 rounded-lg p-3">
                <h5 className="font-medium text-blue-800 mb-1">
                  About{" "}
                  {
                    (
                      ENHANCED_CATEGORY_CONFIG[selectedInstrument.category] ||
                      DEFAULT_CATEGORY_CONFIG
                    ).label
                  }
                </h5>
                <p className="text-sm text-blue-700">
                  {
                    (
                      ENHANCED_CATEGORY_CONFIG[selectedInstrument.category] ||
                      DEFAULT_CATEGORY_CONFIG
                    ).description
                  }
                </p>
              </div>

              {/* AI Enhancement indicator if available */}
              {selectedInstrument.explanation && (
                <div className="bg-green-50 rounded-lg p-3">
                  <h5 className="font-medium text-green-800 mb-1 flex items-center">
                    <span className="mr-1">🤖</span>
                    AI Selection Reason
                  </h5>
                  <p className="text-sm text-green-700">
                    {selectedInstrument.explanation.substring(0, 150)}...
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
