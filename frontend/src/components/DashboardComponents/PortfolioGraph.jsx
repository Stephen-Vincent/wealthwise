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
  console.log("📊 Raw portfolioData:", portfolioData);
  const risk = portfolioData?.risk;

  const portfolio = portfolioData?.portfolio || {};
  const timeline = portfolioData?.timeline || [];

  const timeframe = portfolioData?.timeframe || "";
  const isShortTerm =
    timeframe.includes("< 1 year") || timeframe.includes("1 year");

  const filteredTimeline = isShortTerm
    ? timeline.slice(-252) // Approx. 1 trading year
    : timeline;

  const chartLabels = filteredTimeline.map((entry) => entry.date);

  const stockKeys = Object.keys(portfolio);
  console.log("📊 Available stock symbols:", stockKeys);
  console.log("📅 Timeline sample:", timeline.slice(0, 5));

  // Prepare portfolio total value over time
  const totalValueDataset = {
    label: "Total Portfolio Value",
    data: filteredTimeline.map((entry) => entry.value),
    borderColor: "blue",
    backgroundColor: "blue",
    tension: 0.4,
    fill: false,
    borderWidth: 1, // thinner line
    pointRadius: 0,
    pointHoverRadius: 0,
  };

  // Highlight monthly contributions on the 1st of each month
  const contributionDataset = {
    label: "Monthly Contribution",
    data: filteredTimeline.map((entry) =>
      entry.date.endsWith("-01") ? entry.value : null
    ),
    borderColor: "orange",
    backgroundColor: "orange",
    pointRadius: 0,
    pointHoverRadius: 0,
    showLine: false,
  };

  const chartData = {
    labels: chartLabels,
    datasets: [totalValueDataset, contributionDataset],
  };
  console.log("📈 Chart labels:", chartData.labels);

  // Calculate investment details
  const initialInvestment = portfolioData?.initial_investment || 0;
  const monthlyContribution = portfolioData?.monthly_contribution || 0;
  const months = filteredTimeline.length;
  const totalInvested = initialInvestment + monthlyContribution * months;
  const finalBalance = portfolioData?.final_balance || 0;
  const profitOrLoss = finalBalance - totalInvested;

  return (
    <div className="mb-8">
      <h3 className="text-lg font-semibold mb-2">Your Portfolio</h3>
      <div className="text-sm text-gray-700 mb-1">
        Initial Investment: £
        {(portfolioData?.initial_investment || 0).toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}
        <br />
        Monthly Contribution: £
        {(portfolioData?.monthly_contribution || 0).toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}
      </div>
      <p className="text-blue-600 text-lg font-bold mb-4">
        £
        {portfolioData?.final_balance?.toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        }) || "0.00"}
      </p>
      <p className="text-gray-700 text-sm mb-2">
        Total Invested: £
        {totalInvested.toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}
        <br />
        {profitOrLoss >= 0 ? "Total Gain" : "Total Loss"}: £
        {Math.abs(profitOrLoss).toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}
      </p>
      <div className="bg-gray-100 rounded p-4">
        {timeline.length === 0 ? (
          <p className="text-red-500">
            No portfolio data available to display.
          </p>
        ) : (
          <Line data={chartData} options={options} />
        )}
      </div>
    </div>
  );
}
