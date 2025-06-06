import Sidebar from "./DashboardComponents/Sidebar";
import Header from "./DashboardComponents/Header";
import SummaryCards from "./DashboardComponents/SummaryCards";
import PortfolioGraph from "./DashboardComponents/PortfolioGraph";
import StockPieChart from "./DashboardComponents/StockPieChart";
import AIPortfolioSummary from "./DashboardComponents/AIPortfolioSummary";
import DashboardButtons from "./DashboardComponents/DashboardButtons";
import { useNavigate, useParams } from "react-router-dom";
import { useState, useEffect, useContext } from "react";
import axios from "axios";
import PortfolioContext from "../context/PortfolioContext";

export default function Dashboard() {
  const navigate = useNavigate();
  const { id } = useParams();
  const [simulationData, setSimulationData] = useState(null);

  useEffect(() => {
    const userId = localStorage.getItem("userId");
    if (!userId) {
      console.error("No userId found in localStorage. Redirecting to login.");
      navigate("/login");
    } else {
      console.log("Loaded userId:", userId);
      console.log("Simulation ID from route:", id);
      // If you have any simulation fetching logic, it can go here.
      axios
        .get(`http://localhost:8000/simulations/${id}`)
        .then((response) => {
          setSimulationData(response.data);
          console.log("Fetched simulation data:", response.data);

          // Save simulation data to backend
          axios
            .post("http://localhost:8000/simulations", {
              ...response.data,
              user_id: userId,
            })
            .then(() => {
              console.log("Simulation data saved successfully.");
            })
            .catch((err) => {
              console.error("Error saving simulation data:", err);
            });
        })
        .catch((error) => {
          console.error("Error fetching simulation data:", error);
        });
    }
  }, [id]);

  const handleSliceClick = (label) => {
    navigate(`/stock/${label}`);
  };

  const handleLogout = () => {
    localStorage.removeItem("userId");
    localStorage.removeItem("user_name");
    navigate("/login");
  };

  const portfolioData = useContext(PortfolioContext);

  return (
    <div className="flex">
      <Sidebar />
      <main className="p-8 w-5/6 ">
        <button
          onClick={handleLogout}
          className="mb-4 bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded"
        >
          Logout
        </button>
        <Header portfolioData={simulationData} />
        <SummaryCards portfolioData={simulationData} />
        <PortfolioGraph portfolioData={simulationData} />

        <StockPieChart data={simulationData} onSliceClick={handleSliceClick} />
        {console.log(
          "📊 simulationData passed to StockPieChart:",
          simulationData
        )}

        <AIPortfolioSummary portfolioData={simulationData} />
        <DashboardButtons />
      </main>
    </div>
  );
}
