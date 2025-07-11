import { useRef } from "react"; // <-- Import useRef

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

  // Add refs for each dashboard section
  const summaryRef = useRef(null);
  const graphRef = useRef(null);
  const aiSummaryRef = useRef(null);
  const pieChartRef = useRef(null);
  const buttonsRef = useRef(null);

  if (!portfolioData)
    return (
      <div className="p-8 text-red-500">
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
  };

  const sectionRefs = {
    summaryRef,
    graphRef,
    aiSummaryRef,
    pieChartRef,
  };

  return (
    <div className="flex">
      <Sidebar scrollToSection={scrollToSection} sectionRefs={sectionRefs} />
      <main className="p-8 w-5/6 ">
        <Header portfolioData={portfolioData} />

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
          <StockPieChart data={portfolioData} onSliceClick={handleSliceClick} />
        </section>
        <section ref={buttonsRef}>
          <DashboardButtons />
        </section>
      </main>
    </div>
  );
}
