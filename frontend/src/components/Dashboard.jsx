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
import { useContext } from "react";
import { PortfolioContext } from "../context/PortfolioContext";

// -----------------------------
// Test SHAP
// -----------------------------
// Import SHAP testing utilities
// Uncomment the line below if you have the test_shap.js file saved
// import { testShapData, monitorShapRequests, quickShapCheck } from '../utils/test_shap.js';

// Inline SHAP testing functions for immediate use
const testShapData = (portfolioData, simulationId = null) => {
  console.log("🧪 Starting comprehensive SHAP data test...");
  console.log(
    "📊 Testing simulation:",
    simulationId || portfolioData?.id || "unknown"
  );

  if (!portfolioData) {
    console.log("❌ No portfolio data provided");
    return { error: "No portfolio data provided" };
  }

  const results = {
    simulationId: simulationId || portfolioData.id,
    timestamp: new Date().toISOString(),
    hasShapData: false,
    shapLocation: null,
    shapAnalysis: {},
    issues: [],
    recommendations: [],
  };

  try {
    // Check data structure
    console.log("📋 Portfolio data keys:", Object.keys(portfolioData));

    // 1. Check top-level SHAP data
    console.log("\n🔍 Checking top-level SHAP data...");
    if (portfolioData.shap_explanation) {
      console.log("✅ Found SHAP data at top level!");
      results.hasShapData = true;
      results.shapLocation = "top-level";
      results.shapAnalysis = {
        keys: Object.keys(portfolioData.shap_explanation),
      };
    } else {
      console.log("❌ No SHAP data at top level");
    }

    // 2. Check results-level SHAP data
    console.log("\n🔍 Checking results-level SHAP data...");
    if (portfolioData.results?.shap_explanation) {
      console.log("✅ Found SHAP data in results!");
      if (!results.hasShapData) {
        results.hasShapData = true;
        results.shapLocation = "results";
        results.shapAnalysis = {
          keys: Object.keys(portfolioData.results.shap_explanation),
        };
      }
    } else {
      console.log("❌ No SHAP data in results");
    }

    // 3. Check SHAP flags
    console.log("\n🏁 Checking SHAP flags...");
    const wealthwiseEnhanced = portfolioData.wealthwise_enhanced;
    const hasShapFlag = portfolioData.has_shap_explanations;
    const methodology = portfolioData.methodology;

    console.log("  wealthwise_enhanced:", wealthwiseEnhanced);
    console.log("  has_shap_explanations:", hasShapFlag);
    console.log("  methodology:", methodology);

    results.flags = { wealthwiseEnhanced, hasShapFlag, methodology };

    // 4. Check for inconsistencies
    if (hasShapFlag && !results.hasShapData) {
      const issue = "has_shap_explanations=true but no SHAP data found";
      console.log("⚠️ ISSUE:", issue);
      results.issues.push(issue);
    }

    if (wealthwiseEnhanced && !results.hasShapData) {
      const issue = "wealthwise_enhanced=true but no SHAP data found";
      console.log("⚠️ ISSUE:", issue);
      results.issues.push(issue);
    }

    // 5. Generate recommendations
    if (!results.hasShapData) {
      if (results.flags.wealthwiseEnhanced) {
        results.recommendations.push(
          "Backend is generating SHAP data but not exposing it in API response"
        );
        results.recommendations.push(
          "Check format_enhanced_simulation_response() function"
        );
      } else {
        results.recommendations.push(
          "WealthWise enhanced features are not enabled"
        );
      }
    }

    // 6. Summary
    console.log("\n🎯 === SHAP TEST SUMMARY ===");
    if (results.hasShapData) {
      console.log("✅ SHAP data IS available!");
      console.log(`📍 Location: ${results.shapLocation}`);
      const accessPath =
        results.shapLocation === "top-level"
          ? "portfolioData.shap_explanation"
          : "portfolioData.results.shap_explanation";
      console.log("🔧 Access via:", accessPath);
    } else {
      console.log("❌ SHAP data is NOT available");
      console.log("🔧 This explains why your SHAP dashboard is not visible");
    }

    if (results.issues.length > 0) {
      console.log("⚠️ Issues found:", results.issues.length);
      results.issues.forEach((issue) => console.log(`  - ${issue}`));
    }

    if (results.recommendations.length > 0) {
      console.log("💡 Recommendations:");
      results.recommendations.forEach((rec) => console.log(`  - ${rec}`));
    }
  } catch (error) {
    console.error("❌ SHAP test failed:", error);
    results.error = error.message;
  }

  // Store results for inspection
  window.lastShapTestResults = results;
  console.log("\n💾 Results stored in window.lastShapTestResults");

  return results;
};

const quickShapCheck = (portfolioData) => {
  if (!portfolioData) {
    console.log("❌ No portfolio data provided");
    return false;
  }

  const hasTopLevel = !!portfolioData.shap_explanation;
  const hasResults = !!portfolioData.results?.shap_explanation;
  const hasShap = hasTopLevel || hasResults;

  console.log("🎯 Quick SHAP Check:");
  console.log("  Top-level SHAP:", hasTopLevel);
  console.log("  Results SHAP:", hasResults);
  console.log("  WealthWise enhanced:", portfolioData.wealthwise_enhanced);
  console.log("  Has SHAP flag:", portfolioData.has_shap_explanations);

  return {
    hasShap,
    location: hasTopLevel ? "top-level" : hasResults ? "results" : null,
    accessPath: hasTopLevel
      ? "portfolioData.shap_explanation"
      : hasResults
      ? "portfolioData.results.shap_explanation"
      : null,
  };
};

const testStoredPortfolioData = () => {
  console.log("📦 Testing stored portfolio data...");

  try {
    const stored = localStorage.getItem("portfolioData");
    if (stored) {
      const data = JSON.parse(stored);
      console.log("✅ Found stored portfolio data");
      return testShapData(data, data.id);
    } else {
      console.log("❌ No stored portfolio data found");
      return { error: "No stored data" };
    }
  } catch (error) {
    console.log("❌ Failed to parse stored data:", error.message);
    return { error: error.message };
  }
};

// -----------------------------
// End Test SHAP
// -----------------------------

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

  // -----------------------------
  // Test SHAP
  // -----------------------------
  // Auto-test SHAP data when component mounts or portfolioData changes
  useEffect(() => {
    if (portfolioData) {
      console.log("\n🧪 === AUTO SHAP TEST ON DASHBOARD LOAD ===");
      const testResults = testShapData(portfolioData, portfolioData.id);

      // Store test results for debugging
      window.dashboardShapTest = testResults;

      // Also make testing functions available globally
      window.testCurrentPortfolioShap = () =>
        testShapData(portfolioData, portfolioData.id);
      window.quickShapCheck = () => quickShapCheck(portfolioData);
      window.testStoredPortfolioData = testStoredPortfolioData;

      console.log("\n💡 Available test functions:");
      console.log("  - window.testCurrentPortfolioShap()");
      console.log("  - window.quickShapCheck()");
      console.log("  - window.testStoredPortfolioData()");
    }
  }, [portfolioData]);

  // Keyboard shortcut for quick testing (development only)
  useEffect(() => {
    const handleKeyPress = (event) => {
      // Ctrl+Shift+S to run SHAP test
      if (event.ctrlKey && event.shiftKey && event.key === "S") {
        event.preventDefault();
        console.log("\n🧪 === KEYBOARD SHORTCUT SHAP TEST ===");
        if (portfolioData) {
          testShapData(portfolioData, portfolioData.id);
        } else {
          testStoredPortfolioData();
        }
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [portfolioData]);
  // -----------------------------
  // End Test SHAP
  // -----------------------------

  if (!portfolioData)
    return (
      <div className="p-4 md:p-8 text-red-500">
        Failed to load simulation data. Please try again.
        {/* -----------------------------
            Test SHAP
            ----------------------------- */}
        <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded">
          <h3 className="font-bold text-yellow-800">
            🧪 SHAP Testing Available
          </h3>
          <p className="text-sm text-yellow-700 mt-2">
            Even without portfolio data, you can test stored data:
          </p>
          <button
            onClick={testStoredPortfolioData}
            className="mt-2 px-3 py-1 bg-yellow-500 text-white rounded text-sm hover:bg-yellow-600"
          >
            Test Stored Portfolio Data
          </button>
        </div>
        {/* -----------------------------
            End Test SHAP
            ----------------------------- */}
      </div>
    );

  // ✅ FIXED: Check for SHAP data in the correct location
  const hasShapExplanation = Boolean(portfolioData?.shap_explanation);

  // -----------------------------
  // Test SHAP
  // -----------------------------
  // 🔍 Enhanced debug log with test results
  console.log("🔍 SHAP Debug:", {
    hasShapExplanation,
    shapData: portfolioData?.shap_explanations,
    wealthwiseEnhanced: portfolioData?.wealthwise_enhanced,
  });

  // Quick test on every render (for debugging)
  const shapCheckResult = quickShapCheck(portfolioData);
  // -----------------------------
  // End Test SHAP
  // -----------------------------

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

        {/* -----------------------------
            Test SHAP
            ----------------------------- */}
        {/* Development SHAP Testing Panel - Only show in development */}
        {process.env.NODE_ENV === "development" && (
          <div className="sticky top-0 z-20 bg-yellow-100 border-b border-yellow-300 p-2">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-yellow-800">
                🧪 SHAP Testing Panel (Dev Mode)
              </span>
              <div className="flex space-x-2">
                <button
                  onClick={() => testShapData(portfolioData, portfolioData.id)}
                  className="px-2 py-1 bg-yellow-500 text-white rounded text-xs hover:bg-yellow-600"
                >
                  Test Current Data
                </button>
                <button
                  onClick={() => quickShapCheck(portfolioData)}
                  className="px-2 py-1 bg-blue-500 text-white rounded text-xs hover:bg-blue-600"
                >
                  Quick Check
                </button>
                <button
                  onClick={testStoredPortfolioData}
                  className="px-2 py-1 bg-green-500 text-white rounded text-xs hover:bg-green-600"
                >
                  Test Stored Data
                </button>
              </div>
            </div>
            <div className="text-xs text-yellow-700 mt-1">
              Quick Test Result:{" "}
              {shapCheckResult?.hasShap
                ? `✅ SHAP found at ${shapCheckResult.location}`
                : "❌ No SHAP data"}{" "}
              | Press Ctrl+Shift+S for keyboard test
            </div>
          </div>
        )}
        {/* -----------------------------
            End Test SHAP
            ----------------------------- */}

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

          {/* ⚠️ ENHANCED: Debug section with testing capabilities */}
          {!hasShapExplanation && (
            <section ref={shapRef}>
              <div className="bg-yellow-50 border-2 border-yellow-300 rounded-xl p-6">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="text-lg font-bold text-yellow-800">
                    🔍 SHAP Debug Information
                  </h3>
                  {/* -----------------------------
                      Test SHAP
                      ----------------------------- */}
                  <div className="flex space-x-2">
                    <button
                      onClick={() =>
                        testShapData(portfolioData, portfolioData.id)
                      }
                      className="px-3 py-1 bg-yellow-500 text-white rounded text-sm hover:bg-yellow-600"
                    >
                      🧪 Run Full Test
                    </button>
                    <button
                      onClick={() => quickShapCheck(portfolioData)}
                      className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
                    >
                      ⚡ Quick Check
                    </button>
                  </div>
                  {/* -----------------------------
                      End Test SHAP
                      ----------------------------- */}
                </div>

                <div className="space-y-2 text-sm">
                  <p>
                    <strong>Has SHAP data:</strong>{" "}
                    {hasShapExplanation ? "✅ Yes" : "❌ No"}
                  </p>
                  <p>
                    <strong>SHAP explanations exists:</strong>{" "}
                    {portfolioData?.shap_explanation ? "✅ Yes" : "❌ No"}
                  </p>
                  <p>
                    <strong>WealthWise enhanced:</strong>{" "}
                    {portfolioData?.wealthwise_enhanced ? "✅ Yes" : "❌ No"}
                  </p>
                  {/* -----------------------------
                      Test SHAP
                      ----------------------------- */}
                  <p>
                    <strong>Quick Check Result:</strong>{" "}
                    {shapCheckResult?.hasShap
                      ? `✅ Found at ${shapCheckResult.location}`
                      : "❌ Not found"}
                  </p>
                  {shapCheckResult?.accessPath && (
                    <p>
                      <strong>Access Path:</strong>{" "}
                      <code className="bg-gray-100 px-1 rounded">
                        {shapCheckResult.accessPath}
                      </code>
                    </p>
                  )}
                  {/* -----------------------------
                      End Test SHAP
                      ----------------------------- */}
                  <p>
                    <strong>Top-level keys:</strong>{" "}
                    {portfolioData
                      ? Object.keys(portfolioData).join(", ")
                      : "None"}
                  </p>

                  {portfolioData?.shap_explanation && (
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

                  {/* -----------------------------
                      Test SHAP
                      ----------------------------- */}
                  {/* Show last test results if available */}
                  {typeof window !== "undefined" &&
                    window.lastShapTestResults && (
                      <details className="mt-4 p-3 bg-white rounded border">
                        <summary className="cursor-pointer font-medium text-gray-700">
                          🧪 Last Test Results
                        </summary>
                        <div className="mt-2 space-y-1 text-xs">
                          <p>
                            <strong>Has SHAP Data:</strong>{" "}
                            {window.lastShapTestResults.hasShapData
                              ? "✅"
                              : "❌"}
                          </p>
                          <p>
                            <strong>Location:</strong>{" "}
                            {window.lastShapTestResults.shapLocation || "None"}
                          </p>
                          <p>
                            <strong>Issues:</strong>{" "}
                            {window.lastShapTestResults.issues.length}
                          </p>
                          {window.lastShapTestResults.recommendations.length >
                            0 && (
                            <div>
                              <strong>Recommendations:</strong>
                              <ul className="ml-4 mt-1">
                                {window.lastShapTestResults.recommendations.map(
                                  (rec, i) => (
                                    <li key={i} className="text-xs">
                                      • {rec}
                                    </li>
                                  )
                                )}
                              </ul>
                            </div>
                          )}
                        </div>
                      </details>
                    )}
                  {/* -----------------------------
                      End Test SHAP
                      ----------------------------- */}
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
