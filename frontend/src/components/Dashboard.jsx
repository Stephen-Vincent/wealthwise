/**
 * Dashboard.jsx
 * --------------
 * Main Dashboard component for the WealthWise application.
 *
 * - Pulls portfolio data and related information from context.
 * - Manages loading and error states for robust user experience.
 * - Handles sidebar navigation and integrates scroll navigation via refs for each key section.
 * - Renders the main dashboard sections: summary cards, portfolio graphs, SHAP (AI explainability) analysis,
 *   AI portfolio summary, stock pie chart, and dashboard action buttons.
 * - Integrates a responsive sidebar with toggling for mobile/desktop layouts.
 * - Provides enhanced feedback when SHAP (explainable AI) data is available, and includes development/debug panels.
 */
import { useRef, useState, useEffect } from "react";
import Sidebar from "./DashboardComponents/Sidebar";
import Header from "./DashboardComponents/Header";
import SummaryCards from "./DashboardComponents/SummaryCards";
import PortfolioGraph from "./DashboardComponents/PortfolioGraph";
import StockPieChart from "./DashboardComponents/StockPieChart";
import AIPortfolioSummary from "./DashboardComponents/AIPortfolioSummary";
import DashboardButtons from "./DashboardComponents/DashboardButtons";
import SHAPDashboard from "./DashboardComponents/SHAPDashboard";
import { useNavigate } from "react-router-dom";
import { usePortfolio } from "../context/PortfolioContext";

export default function Dashboard() {
  const navigate = useNavigate();

  // Use centralized portfolio data from context
  const {
    portfolioData,
    loading,
    error,
    shapData,
    hasShapData,
    chartData,
    enhancedData,
    refreshPortfolioData,
  } = usePortfolio();

  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Add refs for each dashboard section
  const summaryRef = useRef(null);
  const graphRef = useRef(null);
  const aiSummaryRef = useRef(null);
  const shapRef = useRef(null);
  const pieChartRef = useRef(null);
  const buttonsRef = useRef(null);

  // Debug logging for SHAP data (development only)
  useEffect(() => {
    if (portfolioData && process.env.NODE_ENV === "development") {
      console.log("Dashboard Data Debug:", {
        hasPortfolioData: !!portfolioData,
        hasShapData,
        hasChartData: !!chartData,
        hasEnhancedData: !!enhancedData,
        shapDataKeys: shapData ? Object.keys(shapData) : "None",
        chartDataKeys: chartData ? Object.keys(chartData) : "None",
      });
    }
  }, [portfolioData, hasShapData, chartData, enhancedData, shapData]);

  // Loading state
  if (loading) {
    return (
      <div className="flex min-h-screen bg-gray-50 items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your portfolio data...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex min-h-screen bg-gray-50 items-center justify-center">
        <div className="text-center max-w-md">
          <div className="text-6xl mb-4">⚠️</div>
          <h2 className="text-xl font-bold text-red-800 mb-2">
            Error Loading Data
          </h2>
          <p className="text-red-600 mb-4">{error}</p>
          <button
            onClick={refreshPortfolioData}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  // No data state
  if (!portfolioData) {
    return (
      <div className="flex min-h-screen bg-gray-50 items-center justify-center">
        <div className="text-center max-w-md">
          <div className="text-6xl mb-4">📊</div>
          <h2 className="text-xl font-bold text-gray-800 mb-2">
            No Portfolio Data
          </h2>
          <p className="text-gray-600 mb-4">
            No simulation data found. Please create a portfolio first.
          </p>
          <button
            onClick={() => navigate("/onboarding")}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Create Portfolio
          </button>
        </div>
      </div>
    );
  }

  const handleSliceClick = (label) => {
    navigate(`/stock/${label}`);
  };

  // Menu click handler
  const scrollToSection = (ref) => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    setSidebarOpen(false);
  };

  const sectionRefs = {
    summaryRef,
    graphRef,
    shapRef,
    aiSummaryRef,
    pieChartRef,
  };

  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* Sidebar */}
      <div
        className={`
          fixed lg:static top-0 left-0 z-40
          lg:w-1/6 w-80
          h-screen lg:sticky lg:top-0
          transition-transform duration-300 ease-in-out
          ${
            sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
          }
        `}
      >
        <Sidebar
          scrollToSection={scrollToSection}
          sectionRefs={sectionRefs}
          onClose={() => setSidebarOpen(false)}
        />
      </div>

      {/* Main Content */}
      <main className="flex-1 w-full lg:w-5/6 relative">
        {/* Header */}
        <Header portfolioData={portfolioData} />

        {/* Enhanced Portfolio Badge - Show if SHAP data exists */}
        {hasShapData && (
          <div className="sticky top-0 z-30 bg-gradient-to-r from-green-500 to-blue-600 text-white py-2 px-4 shadow-md">
            <div className="flex items-center justify-center space-x-2">
              <span className="text-sm font-medium">AI-Enhanced Portfolio</span>
              <span className="bg-white bg-opacity-20 rounded-full px-2 py-1 text-xs">
                Explainable AI Available
              </span>
            </div>
          </div>
        )}

        {/* Development Debug Panel */}
        {process.env.NODE_ENV === "development" && (
          <div className="sticky top-0 z-20 bg-yellow-100 border-b border-yellow-300 p-2">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-yellow-800">
                Debug Panel (Dev Mode)
              </span>
              <div className="flex space-x-2">
                <button
                  onClick={refreshPortfolioData}
                  className="px-2 py-1 bg-blue-500 text-white rounded text-xs hover:bg-blue-600"
                >
                  Refresh Data
                </button>
                <span className="px-2 py-1 bg-gray-200 rounded text-xs">
                  SHAP: {hasShapData ? "Available" : "Not Available"}
                </span>
                <span className="px-2 py-1 bg-gray-200 rounded text-xs">
                  Charts: {chartData ? "Available" : "Not Available"}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Floating Menu Button */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="
            fixed top-8 left-4 z-50
            w-12 h-12 bg-blue-600 hover:bg-blue-700
            text-white rounded-full shadow-lg
            flex items-center justify-center
            transition-colors duration-200
            lg:hidden
          "
          aria-label={sidebarOpen ? "Close menu" : "Open menu"}
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            {sidebarOpen ? (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            ) : (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            )}
          </svg>
        </button>

        {/* Dashboard Content */}
        <div className="p-4 md:p-6 lg:p-8 space-y-6 md:space-y-8">
          {/* Summary Cards */}
          <section ref={summaryRef}>
            <SummaryCards portfolioData={portfolioData} />
          </section>

          {/* Portfolio Graph */}
          <section ref={graphRef}>
            <PortfolioGraph portfolioData={portfolioData} />
          </section>

          {/* SHAP Explanation Section - Show if SHAP data exists */}
          {hasShapData ? (
            <section ref={shapRef}>
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4 text-white">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-xl font-bold">
                        AI Decision Explanation
                      </h2>
                      <p className="text-blue-100 text-sm">
                        Understand exactly why our AI recommended this portfolio
                        for you
                      </p>
                    </div>
                    <div className="bg-white bg-opacity-20 rounded-lg px-3 py-1">
                      <span className="text-xs font-medium">Enhanced AI</span>
                    </div>
                  </div>
                </div>

                <div className="p-6">
                  {/* Pass only the data, let SHAPDashboard consume from context */}
                  <SHAPDashboard />
                </div>
              </div>
            </section>
          ) : (
            /* Debug Section when SHAP data is not available */
            <section ref={shapRef}>
              <div className="bg-yellow-50 border-2 border-yellow-300 rounded-xl p-6">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="text-lg font-bold text-yellow-800">
                    SHAP Data Not Available
                  </h3>
                  <button
                    onClick={refreshPortfolioData}
                    className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
                  >
                    Refresh Data
                  </button>
                </div>

                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Portfolio Data:</strong>{" "}
                    {portfolioData ? "Available" : "Not Available"}
                  </p>
                  <p>
                    <strong>SHAP Data:</strong>{" "}
                    {hasShapData ? "Available" : "Not Available"}
                  </p>
                  <p>
                    <strong>Chart Data:</strong>{" "}
                    {chartData ? "Available" : "Not Available"}
                  </p>
                  <p>
                    <strong>Enhanced Features:</strong>{" "}
                    {portfolioData?.has_shap_explanations
                      ? "Enabled"
                      : "Disabled"}
                  </p>

                  {portfolioData && (
                    <details className="mt-4 p-3 bg-white rounded border">
                      <summary className="cursor-pointer font-medium text-gray-700">
                        Portfolio Data Structure
                      </summary>
                      <div className="mt-2 text-xs">
                        <p>
                          <strong>Top-level keys:</strong>{" "}
                          {Object.keys(portfolioData).join(", ")}
                        </p>
                        {portfolioData.results && (
                          <p>
                            <strong>Results keys:</strong>{" "}
                            {Object.keys(portfolioData.results).join(", ")}
                          </p>
                        )}
                        {chartData && (
                          <p>
                            <strong>Chart data keys:</strong>{" "}
                            {Object.keys(chartData).join(", ")}
                          </p>
                        )}
                      </div>
                    </details>
                  )}

                  <div className="mt-4 p-3 bg-blue-50 rounded border border-blue-200">
                    <p className="text-blue-800 font-medium">
                      Possible Solutions:
                    </p>
                    <ul className="mt-2 text-blue-700 text-xs space-y-1">
                      <li>
                        • Backend might not be generating SHAP data for this
                        simulation
                      </li>
                      <li>
                        • SHAP visualization endpoints might not be responding
                      </li>
                      <li>
                        • Data might be in a different format than expected
                      </li>
                      <li>
                        • Try refreshing the data or creating a new simulation
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* AI Portfolio Summary */}
          <section ref={aiSummaryRef}>
            <AIPortfolioSummary portfolioData={portfolioData} />
          </section>

          {/* Stock Pie Chart */}
          <section ref={pieChartRef}>
            <StockPieChart
              data={portfolioData}
              onSliceClick={handleSliceClick}
            />
          </section>

          {/* Dashboard Buttons */}
          <section ref={buttonsRef}>
            <DashboardButtons />
          </section>
        </div>
      </main>
    </div>
  );
}
