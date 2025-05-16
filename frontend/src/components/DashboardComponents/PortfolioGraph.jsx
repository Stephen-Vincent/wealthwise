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

function getRandomColor() {
  return `hsl(${Math.floor(Math.random() * 360)}, 70%, 50%)`;
}

export default function PortfolioGraph() {
  const { portfolioData } = useContext(PortfolioContext);
  const risk = portfolioData?.risk;

  const portfolio = portfolioData?.portfolio || {};
  const timeline = portfolioData?.timeline || [];

  const stockKeys = Object.keys(portfolio);

  // Build chart datasets for each stock symbol
  const datasets = stockKeys.map((symbol) => {
    const stockTimeline = portfolio[symbol]?.timeline || [];

    return {
      label: symbol,
      data: timeline.map((point) => {
        const match = stockTimeline.find((entry) => entry.date === point.date);
        return match ? match.value : null;
      }),
      fill: false,
      borderColor: getRandomColor(), // Assign a random color to each stock line
      tension: 0.4, // Smooth the line chart curves
    };
  });

  const chartData = {
    labels: timeline.map((point) => point.date),
    datasets,
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
