import { useRef, useState } from "react";
import Sidebar from "./DashboardComponents/Sidebar";
import Header from "./DashboardComponents/Header";
import SummaryCards from "./DashboardComponents/SummaryCards";
import PortfolioGraph from "./DashboardComponents/PortfolioGraph";
import StockPieChart from "./DashboardComponents/StockPieChart";
import AIPortfolioSummary from "./DashboardComponents/AIPortfolioSummary";
import DashboardButtons from "./DashboardComponents/DashboardButtons";
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
  const pieChartRef = useRef(null);
  const buttonsRef = useRef(null);

  if (!portfolioData)
    return (
      <div className="p-4 md:p-8 text-red-500">
        Failed to load simulation data. Please try again.
      </div>
    );

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

  const sectionRefs = {
    summaryRef,
    graphRef,
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
