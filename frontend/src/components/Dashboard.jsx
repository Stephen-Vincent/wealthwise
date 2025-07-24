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
    // Close sidebar on mobile after navigation
    setSidebarOpen(false);
  };

  const sectionRefs = {
    summaryRef,
    graphRef,
    aiSummaryRef,
    pieChartRef,
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-screen">
      {/* Mobile Header with Hamburger Menu */}
      <div className="lg:hidden bg-white shadow-sm border-b px-4 py-3 flex items-center justify-between">
        <h1 className="text-lg font-semibold text-gray-900">Dashboard</h1>
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100"
          aria-label="Toggle menu"
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
      </div>

      {/* Sidebar - Desktop: Fixed, Mobile: Overlay */}
      <div
        className={`
        lg:relative lg:block
        ${sidebarOpen ? "block" : "hidden"}
        lg:w-1/6 w-full
        lg:min-h-screen
        ${
          sidebarOpen
            ? "fixed inset-0 z-50 lg:relative lg:inset-auto lg:z-auto"
            : ""
        }
      `}
      >
        {/* Mobile overlay background */}
        {sidebarOpen && (
          <div
            className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-40"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Sidebar content */}
        <div
          className={`
          lg:relative lg:h-full
          ${sidebarOpen ? "relative z-50 h-full" : ""}
        `}
        >
          <Sidebar
            scrollToSection={scrollToSection}
            sectionRefs={sectionRefs}
            onClose={() => setSidebarOpen(false)}
          />
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 lg:w-5/6 w-full">
        {/* Header - Hidden on mobile (shown in mobile header above) */}
        <div className="hidden lg:block">
          <Header portfolioData={portfolioData} />
        </div>

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
