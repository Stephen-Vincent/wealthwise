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
    annotation: {
      animations: {
        duration: 1200,
        easing: "easeInOutSine",
      },
    },
  },
  animation: {
    duration: 1000,
    easing: "easeInOutQuad",
  },
};

export default function PortfolioGraph() {
  const { portfolioData } = useContext(PortfolioContext);
  console.log("📊 Raw portfolioData:", portfolioData);
  const targetValue = portfolioData?.target_value;

  const portfolio = portfolioData?.portfolio || {};
  const timeline = portfolioData?.timeline || [];

  const timeframe = portfolioData?.timeframe || "";
  const isShortTerm =
    timeframe.includes("< 1 year") || timeframe.includes("1 year");

  const filteredTimeline = isShortTerm
    ? timeline.slice(-252) // Approx. 1 trading year
    : timeline;

  function getMonthlyDataPoints(timeline) {
    const monthlyEntries = {};

    for (const entry of timeline) {
      const monthKey = entry.date.slice(0, 7); // "YYYY-MM"
      if (!monthlyEntries[monthKey]) {
        monthlyEntries[monthKey] = entry;
      }

      if (
        entry.date.endsWith("-01") &&
        typeof entry.value === "number" &&
        typeof entry.contribution === "number"
      ) {
        monthlyEntries[monthKey] = entry; // Prefer 1st-of-month entries with contributions
      }
    }

    return Object.values(monthlyEntries);
  }

  const cleanedTimeline = getMonthlyDataPoints(filteredTimeline);

  // Find when the target was first reached and log it (by value >= targetValue)
  const firstTargetHit = cleanedTimeline.find(
    (entry) => entry.value >= targetValue
  );
  console.log(
    "🎯 First target hit check based on value >= targetValue:",
    firstTargetHit
  );

  const chartLabels = cleanedTimeline.map((entry) => {
    const date = new Date(entry.date);
    return date.toLocaleDateString("en-GB", {
      month: "short",
      year: "2-digit",
    });
  });

  const stockKeys = Object.keys(portfolio);

  // Prepare portfolio total value over time
  const totalValueDataset = {
    label: "Portfolio Value",
    data: cleanedTimeline.map((entry) => entry.value),
    borderColor: "blue",
    backgroundColor: "blue",
    tension: 0.4,
    fill: false,
    borderWidth: 1,
    pointRadius: 0,
    pointHoverRadius: 0,
  };

  // Highlight monthly contributions on the 1st of each month
  const contributionDataset = {
    label: "Monthly Contribution",
    data: cleanedTimeline.map((entry) =>
      entry.is_contribution ? entry.value : null
    ),
    borderColor: "blue",
    backgroundColor: "blue",
    pointRadius: 4,
    pointHoverRadius: 6,
    showLine: false,
  };

  // Find the index of the first entry where the target is reached
  const firstTargetIndex = cleanedTimeline.findIndex(
    (entry) => entry.value >= targetValue
  );
  const targetReachedDataset = {
    label: "Target Reached",
    data: cleanedTimeline.map((entry, index) =>
      index === firstTargetIndex ? entry.value : null
    ),
    pointBackgroundColor: cleanedTimeline.map((entry, index) =>
      index === firstTargetIndex ? "rgba(0, 200, 0, 0.4)" : "transparent"
    ),
    pointBorderColor: cleanedTimeline.map((entry, index) =>
      index === firstTargetIndex ? "rgba(0, 200, 0, 1)" : "transparent"
    ),
    pointRadius: cleanedTimeline.map((entry, index) =>
      index === firstTargetIndex ? 10 : 0
    ),
    pointHoverRadius: cleanedTimeline.map((entry, index) =>
      index === firstTargetIndex ? 12 : 0
    ),
    pointStyle: "circle",
    showLine: false,
  };

  const initialInvestment = portfolioData?.initial_investment || 0;
  const monthlyContribution = portfolioData?.monthly_contribution || 0;
  let runningTotal = initialInvestment;
  const cumulativeInvestedDataset = {
    label: "Total Contributions",
    data: cleanedTimeline.map((entry) => {
      if (entry.is_contribution) {
        runningTotal += monthlyContribution;
      }
      return runningTotal;
    }),
    borderColor: "gray",
    backgroundColor: "gray",
    borderDash: [5, 5],
    fill: false,
    tension: 0.3,
    borderWidth: 1.5,
    pointRadius: cleanedTimeline.map((entry) =>
      entry.is_contribution ? 3 : 0
    ),
    pointHoverRadius: cleanedTimeline.map((entry) =>
      entry.is_contribution ? 5 : 0
    ),
  };

  const chartData = {
    labels: chartLabels,
    datasets: [
      totalValueDataset,
      cumulativeInvestedDataset,
      contributionDataset,
      targetReachedDataset,
    ],
  };

  // Calculate investment details
  const totalInvested = cleanedTimeline.reduce((sum, entry) => {
    return sum + (entry.is_contribution ? monthlyContribution : 0);
  }, initialInvestment);
  const finalBalance =
    cleanedTimeline.length > 0
      ? cleanedTimeline[cleanedTimeline.length - 1].value
      : 0;
  const profitOrLoss = finalBalance - totalInvested;

  return (
    <div className="mb-8">
      <h3 className="text-lg font-semibold mb-2">Your Portfolio</h3>
      <div className="text-sm text-gray-700 mb-1">
        Initial Investment: £
        {portfolioData?.starting_balance?.toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        }) || "0.00"}
        <br />
        Monthly Contribution: £
        {portfolioData?.monthly_contribution?.toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        }) || "0.00"}
      </div>
      <p className="text-blue-600 text-lg font-bold mb-4">
        £
        {finalBalance.toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}
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
