import Sidebar from "./DashboardComponents/Sidebar";
import Header from "./DashboardComponents/Header";
import SummaryCards from "./DashboardComponents/SummaryCards";
import PortfolioGraph from "./DashboardComponents/PortfolioGraph";
import StockPieChart from "./DashboardComponents/StockPieChart";
import AIPortfolioSummary from "./DashboardComponents/AIPortfolioSummary";
import DashboardButtons from "./DashboardComponents/DashboardButtons";
import { useNavigate } from "react-router-dom";
import PortfolioContext from "../context/PortfolioContext";
import { useContext } from "react";

export default function Dashboard() {
  const navigate = useNavigate();

  const handleSliceClick = (label) => {
    navigate(`/stock/${label}`);
  };

  const portfolioData = useContext(PortfolioContext);

  return (
    <div className="flex">
      <Sidebar />
      <main className="p-8 w-5/6 ">
        <Header />
        <SummaryCards />
        <PortfolioGraph />

        <StockPieChart
          riskLevel={portfolioData.risk}
          onSliceClick={handleSliceClick}
        />

        <AIPortfolioSummary portfolioData={portfolioData} />
        <DashboardButtons />
      </main>
    </div>
  );
}
