import { useEffect, useState, useContext } from "react";
import { Chart } from "react-google-charts";
import PortfolioContext from "../../context/PortfolioContext";
import StockInfoAPI from "../../utils/stockInfoAPI";

// Enhanced category configurations - moved from stockMapping.js
const ENHANCED_CATEGORY_CONFIG = {
  Technology: {
    label: "Technology",
    color: "#3B82F6",
    bgColor: "#DBEAFE",
    description: "Technology and innovation-focused investments",
    icon: "💻",
  },
  "Innovation/Growth": {
    label: "Innovation/Growth",
    color: "#8B5CF6",
    bgColor: "#EDE9FE",
    description: "Disruptive innovation and high-growth companies",
    icon: "🚀",
  },
  "Autonomous Technology": {
    label: "Autonomous Technology",
    color: "#F59E0B",
    bgColor: "#FEF3C7",
    description: "Autonomous technology and robotics companies",
    icon: "🤖",
  },
  "Financial Technology": {
    label: "Financial Technology",
    color: "#10B981",
    bgColor: "#D1FAE5",
    description: "Financial technology and fintech companies",
    icon: "💳",
  },
  "Large Cap Growth": {
    label: "Large Cap Growth",
    color: "#8B5CF6",
    bgColor: "#EDE9FE",
    description: "Established companies with growth potential",
    icon: "📈",
  },
  "Small Cap Growth": {
    label: "Small Cap Growth",
    color: "#F59E0B",
    bgColor: "#FEF3C7",
    description: "Smaller companies with high growth potential",
    icon: "🚀",
  },
  "Emerging Markets": {
    label: "Emerging Markets",
    color: "#EF4444",
    bgColor: "#FEE2E2",
    description: "Higher-growth developing country investments",
    icon: "🌍",
  },
  Biotechnology: {
    label: "Biotechnology",
    color: "#10B981",
    bgColor: "#D1FAE5",
    description: "Biotechnology and pharmaceutical companies",
    icon: "🧬",
  },
  Healthcare: {
    label: "Healthcare",
    color: "#10B981",
    bgColor: "#D1FAE5",
    description: "Healthcare and biotechnology companies",
    icon: "🏥",
  },
  Cryptocurrency: {
    label: "Cryptocurrency",
    color: "#F59E0B",
    bgColor: "#FEF3C7",
    description: "Cryptocurrency and digital asset investments",
    icon: "₿",
  },
  "Bitcoin/Cryptocurrency": {
    label: "Bitcoin/Crypto",
    color: "#F59E0B",
    bgColor: "#FEF3C7",
    description: "Bitcoin and cryptocurrency exposure",
    icon: "₿",
  },
  "Financial Services": {
    label: "Financial Services",
    color: "#64748B",
    bgColor: "#F1F5F9",
    description: "Banks, insurance, and financial companies",
    icon: "🏦",
  },
  "Large Cap": {
    label: "Large Cap",
    color: "#6366F1",
    bgColor: "#E0E7FF",
    description: "Large established company investments",
    icon: "🏢",
  },
  "Total Market": {
    label: "Total Market",
    color: "#6366F1",
    bgColor: "#E0E7FF",
    description: "Total market exposure ETFs",
    icon: "📊",
  },
  "Broad Market": {
    label: "Market ETFs",
    color: "#6366F1",
    bgColor: "#E0E7FF",
    description: "Diversified funds tracking broad market indices",
    icon: "📊",
  },
  Bonds: {
    label: "Bonds & Fixed Income",
    color: "#0D9488",
    bgColor: "#CCFBF1",
    description: "Fixed income and bond investments",
    icon: "📋",
  },
  "Real Estate": {
    label: "Real Estate",
    color: "#84CC16",
    bgColor: "#ECFCCB",
    description: "Real estate investment trusts",
    icon: "🏠",
  },
  Other: {
    label: "Other",
    color: "#9CA3AF",
    bgColor: "#F3F4F6",
    description: "Other or uncategorized investments",
    icon: "❓",
  },
};

const DEFAULT_CATEGORY_CONFIG = ENHANCED_CATEGORY_CONFIG["Other"];

// Utility to lighten or darken a hex color
function shadeColor(color, percent) {
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

export default function StockPieChart() {
  const { portfolioData } = useContext(PortfolioContext);
  const stocksPicked = portfolioData?.results?.stocks_picked || [];
  const [selectedInstrument, setSelectedInstrument] = useState(null);
  const [enhancedStocks, setEnhancedStocks] = useState([]);
  const [categoryTotals, setCategoryTotals] = useState({});
  const [loading, setLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState("idle"); // 'idle', 'loading', 'success', 'error'

  const stockInfoAPI = new StockInfoAPI();

  // Process stock data with API enhancement
  useEffect(() => {
    if (stocksPicked.length === 0) {
      setLoading(false);
      return;
    }

    enhanceStocksWithAPI();
  }, [stocksPicked]);

  const enhanceStocksWithAPI = async () => {
    setLoading(true);
    setApiStatus("loading");

    try {
      console.log("🔄 Fetching stock information from API...");
      const symbols = stocksPicked.map((stock) => stock.symbol);
      const apiData = await stockInfoAPI.getBatchStockInfo(symbols);

      // Enhance stocks with API data
      const enhanced = stocksPicked.map((stock) => {
        const apiInfo = apiData[stock.symbol];
        return {
          ...stock,
          name: apiInfo?.name || stock.name || stock.symbol,
          category: apiInfo?.category || "Other",
          description: apiInfo?.description || "Investment instrument",
          riskLevel: apiInfo?.riskLevel || "Unknown",
          sector: apiInfo?.sector || "Unknown",
          price: apiInfo?.price || null,
          marketCap: apiInfo?.marketCap || null,
          dividendYield: apiInfo?.dividendYield || 0,
          beta: apiInfo?.beta || null,
          isETF: apiInfo?.isETF || false,
          dataSource: apiInfo?.dataSource || "Enhanced Database",
          confidence: apiInfo?.confidence || "Medium",
        };
      });

      setEnhancedStocks(enhanced);
      calculateCategoryTotals(enhanced);
      setApiStatus("success");
      console.log("✅ API enhancement completed");
    } catch (error) {
      console.error("❌ API enhancement failed:", error);
      setApiStatus("error");

      // Still try to use the enhanced API's fallback data
      try {
        const symbols = stocksPicked.map((stock) => stock.symbol);
        const fallbackData = await stockInfoAPI.getBatchStockInfo(symbols);

        const fallbackStocks = stocksPicked.map((stock) => {
          const fallbackInfo = fallbackData[stock.symbol];
          return {
            ...stock,
            name: fallbackInfo?.name || stock.name || stock.symbol,
            category: fallbackInfo?.category || "Other",
            description: fallbackInfo?.description || "Investment instrument",
            riskLevel: fallbackInfo?.riskLevel || "Unknown",
            sector: fallbackInfo?.sector || "Unknown",
            dataSource: fallbackInfo?.dataSource || "Fallback",
            confidence: "Low",
          };
        });

        setEnhancedStocks(fallbackStocks);
        calculateCategoryTotals(fallbackStocks);
      } catch (fallbackError) {
        console.error("❌ Fallback also failed:", fallbackError);
        // Use basic stock data
        setEnhancedStocks(
          stocksPicked.map((stock) => ({
            ...stock,
            category: "Other",
            dataSource: "Basic",
            confidence: "None",
          }))
        );
        calculateCategoryTotals(stocksPicked);
      }
    } finally {
      setLoading(false);
    }
  };

  const calculateCategoryTotals = (stocks) => {
    const categories = stocks.reduce((acc, stock) => {
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
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-8 text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <h3 className="text-lg font-semibold text-gray-700 mb-2">
          {apiStatus === "loading"
            ? "Fetching Stock Information..."
            : "Loading Portfolio Data"}
        </h3>
        <p className="text-gray-500">
          {apiStatus === "loading"
            ? "Getting detailed investment information from API..."
            : "Processing allocation data..."}
        </p>
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

  // Chart data preparation
  const chartData = [
    ["Stock", "Allocation"],
    ...enhancedStocks.map((stock) => [
      `${stock.symbol} (${((stock.allocation || 0) * 100).toFixed(1)}%)`,
      (stock.allocation || 0) * 100,
    ]),
  ];

  const chartColors = enhancedStocks.map((stock, idx) => {
    const category = stock.category || "Other";
    const categoryConfig =
      ENHANCED_CATEGORY_CONFIG[category] || DEFAULT_CATEGORY_CONFIG;
    const baseColor = categoryConfig.color;
    const percent = idx % 2 === 0 ? -0.09 : 0.13;
    return shadeColor(baseColor, percent);
  });

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
  };

  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-800">
          Portfolio Allocation
        </h3>

        {/* API Status Indicator */}
        <div className="flex items-center space-x-4">
          <div
            className={`px-3 py-1 rounded-full text-xs font-medium ${
              apiStatus === "success"
                ? "bg-green-100 text-green-800"
                : apiStatus === "error"
                ? "bg-red-100 text-red-800"
                : "bg-gray-100 text-gray-800"
            }`}
          >
            {apiStatus === "success"
              ? "✅ API Enhanced"
              : apiStatus === "error"
              ? "📋 Fallback Data"
              : "⏳ Loading"}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pie Chart */}
        <div className="rounded-2xl p-10 lg:p-14 shadow-2xl bg-white min-h-96">
          <Chart
            chartType="PieChart"
            width="100%"
            height="340px"
            data={chartData}
            options={chartOptions}
          />
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
                          <div className="text-xs text-gray-500 flex items-center space-x-2">
                            <span>{stock.category || "Other"}</span>
                            {stock.dataSource === "Enhanced Database" && (
                              <span className="bg-green-100 text-green-700 px-1 rounded text-xs">
                                DB
                              </span>
                            )}
                            {stock.dataSource?.includes("API") && (
                              <span className="bg-blue-100 text-blue-700 px-1 rounded text-xs">
                                API
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">{percentage}%</div>
                        <div className="text-xs text-gray-500">
                          {stock.riskLevel || "Unknown"} Risk
                        </div>
                        {stock.price && (
                          <div className="text-xs text-blue-600">
                            ${stock.price.toFixed(2)}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Instrument Modal with API Data */}
      {selectedInstrument && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div
            className="rounded-xl max-w-lg w-full p-6 shadow-2xl max-h-96 overflow-y-auto"
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
                  <span
                    className={`px-2 py-1 rounded-full text-xs ${
                      selectedInstrument.dataSource === "Enhanced Database"
                        ? "bg-green-100 text-green-700"
                        : selectedInstrument.dataSource?.includes("API")
                        ? "bg-blue-100 text-blue-700"
                        : "bg-gray-100 text-gray-700"
                    }`}
                  >
                    {selectedInstrument.dataSource === "Enhanced Database"
                      ? "✅ Database"
                      : selectedInstrument.dataSource?.includes("API")
                      ? "🌐 Live API"
                      : "📋 Fallback"}
                  </span>
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

                {/* API-Enhanced Data */}
                {selectedInstrument.sector && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Sector:</span>
                    <span className="font-medium">
                      {selectedInstrument.sector}
                    </span>
                  </div>
                )}
                {selectedInstrument.price && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Current Price:</span>
                    <span className="font-medium text-green-600">
                      ${selectedInstrument.price.toFixed(2)}
                    </span>
                  </div>
                )}
                {selectedInstrument.marketCap && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Market Cap:</span>
                    <span className="font-medium">
                      ${(selectedInstrument.marketCap / 1e9).toFixed(1)}B
                    </span>
                  </div>
                )}
                {selectedInstrument.beta && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Beta:</span>
                    <span className="font-medium">
                      {selectedInstrument.beta.toFixed(2)}
                    </span>
                  </div>
                )}
                {selectedInstrument.dividendYield > 0 && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Dividend Yield:</span>
                    <span className="font-medium text-blue-600">
                      {selectedInstrument.dividendYield.toFixed(2)}%
                    </span>
                  </div>
                )}
              </div>

              {/* Category Description */}
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
