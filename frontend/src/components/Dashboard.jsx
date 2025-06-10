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

  if (!portfolioData)
    return (
      <div className="p-8 text-red-500">
        Failed to load simulation data. Please try again.
      </div>
    );

  const handleSliceClick = (label) => {
    navigate(`/stock/${label}`);
  };

  return (
    <div className="flex">
      <Sidebar />
      <main className="p-8 w-5/6 ">
        <Header portfolioData={portfolioData} />
        <SummaryCards portfolioData={portfolioData} />
        <PortfolioGraph portfolioData={portfolioData} />
        <StockPieChart data={portfolioData} onSliceClick={handleSliceClick} />
        <AIPortfolioSummary portfolioData={portfolioData} />
        <DashboardButtons />
      </main>
    </div>
  );
}
