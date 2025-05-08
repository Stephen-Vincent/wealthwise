import { useContext } from "react";
import PortfolioContext from "../../context/PortfolioContext";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend
);

const options = {
  responsive: true,
  plugins: {
    legend: {
      display: false,
    },
  },
  animation: {
    duration: 1000,
    easing: "easeInOutQuad",
  },
};

export default function PortfolioGraph() {
  const { portfolioData } = useContext(PortfolioContext);

  const timeline = portfolioData?.timeline || [];
  const chartData = {
    labels: timeline.map((point) => point.date),
    datasets: [
      {
        label: "Portfolio Value",
        data: timeline.map((point) => point.value),
        fill: false,
        borderColor: "#00A8FF",
        tension: 0.4,
      },
    ],
  };

  return (
    <div className="mb-8">
      <h3 className="text-lg font-semibold mb-2">Your Portfolio</h3>
      <p className="text-blue-600 text-lg font-bold mb-4">
        £
        {portfolioData?.final_balance?.toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        }) || "0.00"}
      </p>
      <div className="bg-gray-100 rounded p-4">
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}
