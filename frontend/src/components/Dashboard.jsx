import { useRef, useState } from "react";
import Sidebar from "./DashboardComponents/Sidebar";
import Header from "./DashboardComponents/Header";
import SummaryCards from "./DashboardComponents/SummaryCards";
import PortfolioGraph from "./DashboardComponents/PortfolioGraph";
import StockPieChart from "./DashboardComponents/StockPieChart";
import AIPortfolioSummary from "./DashboardComponents/AIPortfolioSummary";
import DashboardButtons from "./DashboardComponents/DashboardButtons";
import SHAPDashboard from "./DashboardComponents/SHAPDashboard";
import { useNavigate } from "react-router-dom";
import { useContext } from "react";
import { PortfolioContext } from "../context/PortfolioContext";

export default function Dashboard() {
  const navigate = useNavigate();
  const { portfolioData } = useContext(PortfolioContext);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Add refs for each dashboard section
  const summaryRef = useRef(null);
  const graphRef = useRef(null);
  const aiSummaryRef = useRef(null);
  const shapRef = useRef(null); // ✅ Add SHAP ref
  const pieChartRef = useRef(null);
  const buttonsRef = useRef(null);

  if (!portfolioData)
    return (
      <div className="p-4 md:p-8 text-red-500">
        Failed to load simulation data. Please try again.
      </div>
    );

  // ✅ FIXED: Check for SHAP data in the correct location
  const hasShapExplanation = Boolean(portfolioData?.shap_explanations);

  // 🔍 Debug log to see what we have
  console.log("🔍 SHAP Debug:", {
    hasShapExplanation,
    shapData: portfolioData?.shap_explanations,
    wealthwiseEnhanced: portfolioData?.wealthwise_enhanced,
  });

  const handleSliceClick = (label) => {
    navigate(`/stock/${label}`);
  };

  // Menu click handler
  const scrollToSection = (ref) => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    // Close sidebar after navigation
    setSidebarOpen(false);
  };

  // ✅ Always include shapRef in sectionRefs
  const sectionRefs = {
    summaryRef,
    graphRef,
    shapRef, // ✅ Always include this
    aiSummaryRef,
    pieChartRef,
  };

  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* Sidebar - Sliding on mobile, static on desktop */}
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
        {hasShapExplanation && (
          <div className="sticky top-0 z-30 bg-gradient-to-r from-green-500 to-blue-600 text-white py-2 px-4 shadow-md">
            <div className="flex items-center justify-center space-x-2">
              <span className="text-sm font-medium">
                🤖 AI-Enhanced Portfolio
              </span>
              <span className="bg-white bg-opacity-20 rounded-full px-2 py-1 text-xs">
                Explainable AI Available
              </span>
            </div>
          </div>
        )}

        {/* Floating Menu Button - Positioned below header */}
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
          {/* Attach refs to sections */}
          <section ref={summaryRef}>
            <SummaryCards portfolioData={portfolioData} />
          </section>

          <section ref={graphRef}>
            <PortfolioGraph portfolioData={portfolioData} />
          </section>

          {/* ✅ SHAP Explanation Section - Show if SHAP data exists */}
          {hasShapExplanation && (
            <section ref={shapRef}>
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4 text-white">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-xl font-bold">
                        🔍 AI Decision Explanation
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
                  <SHAPDashboard portfolioData={portfolioData} />
                </div>
              </div>
            </section>
          )}

          {/* ⚠️ TEMPORARY: Force show SHAP section for debugging */}
          {!hasShapExplanation && (
            <section ref={shapRef}>
              <div className="bg-yellow-50 border-2 border-yellow-300 rounded-xl p-6">
                <h3 className="text-lg font-bold text-yellow-800 mb-4">
                  🔍 SHAP Debug Information
                </h3>
                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Has SHAP data:</strong>{" "}
                    {hasShapExplanation ? "✅ Yes" : "❌ No"}
                  </p>
                  <p>
                    <strong>SHAP explanations exists:</strong>{" "}
                    {portfolioData?.shap_explanations ? "✅ Yes" : "❌ No"}
                  </p>
                  <p>
                    <strong>WealthWise enhanced:</strong>{" "}
                    {portfolioData?.wealthwise_enhanced ? "✅ Yes" : "❌ No"}
                  </p>
                  <p>
                    <strong>Top-level keys:</strong>{" "}
                    {portfolioData
                      ? Object.keys(portfolioData).join(", ")
                      : "None"}
                  </p>

                  {portfolioData?.shap_explanations && (
                    <div className="mt-4 p-4 bg-white rounded border">
                      <p>
                        <strong>SHAP Data Found:</strong>
                      </p>
                      <pre className="text-xs mt-2 overflow-auto max-h-40">
                        {JSON.stringify(
                          portfolioData.shap_explanations,
                          null,
                          2
                        )}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            </section>
          )}

          <section ref={aiSummaryRef}>
            <AIPortfolioSummary portfolioData={portfolioData} />
          </section>

          <section ref={pieChartRef}>
            <StockPieChart
              data={portfolioData}
              onSliceClick={handleSliceClick}
            />
          </section>

          <section ref={buttonsRef}>
            <DashboardButtons />
          </section>
        </div>
      </main>
    </div>
  );
}
