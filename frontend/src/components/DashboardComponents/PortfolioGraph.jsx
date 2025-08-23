/**
 * PortfolioGraph.jsx
 * -----------------
 * Visualizes the user's portfolio performance over time.
 * - Switch between monthly, quarterly, and yearly views.
 * - Toggle overlays for target value and gain/loss area.
 * - Shows key summary metrics and a responsive Chart.js line chart.
 * - Data comes from PortfolioContext.
 */

// React and context imports
import { useContext, useState, useMemo } from "react";
import PortfolioContext from "../../context/PortfolioContext";

// Chart.js imports and registration
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

// Utility functions for calculations and formatting
import {
  calculateReturns,
  formatCurrency,
  formatPercentage,
} from "../../utils/portfolioCalculations";

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Main component for portfolio performance graph
const PortfolioGraph = () => {
  // Get portfolio data from context
  const { portfolioData } = useContext(PortfolioContext);
  // State for switching view modes and toggling overlays
  const [viewMode, setViewMode] = useState("monthly");
  const [showTarget, setShowTarget] = useState(true);
  const [showGainLoss, setShowGainLoss] = useState(true);

  const processedData = useMemo(() => {
    // Extract timeline arrays from portfolio data
    const portfolio = portfolioData?.timeline ?? [];
    const contributions =
      portfolioData?.results?.timeline?.contributions ??
      portfolioData?.timeline?.contributions ??
      portfolioData?.contribution_data ??
      [];

    // Debug logging
    console.log("Portfolio timeline:", portfolio.slice(0, 3));
    console.log("Contributions timeline:", contributions.slice(0, 3));
    console.log("Lump sum:", portfolioData?.lump_sum);

    // Handle case where there is no data
    if (portfolio.length === 0) {
      return {
        labels: [],
        portfolioValues: [],
        contributionValues: [],
        gainLossValues: [],
        targetLine: [],
        isEmpty: true,
      };
    }

    // ADD STARTING POINT: Include initial investment
    const lumpSum = portfolioData?.lump_sum || 0;

    // Create a proper starting date (before the first portfolio entry)
    const firstPortfolioDate = new Date(portfolio[0]?.date);
    const startDate = new Date(firstPortfolioDate);
    startDate.setMonth(startDate.getMonth() - 1); // One month before

    const startingPortfolio = {
      date: startDate.toISOString(),
      value: lumpSum,
    };
    const startingContributions = {
      date: startDate.toISOString(),
      value: lumpSum,
    };

    // Prepend starting values to the data
    const portfolioWithStart = [startingPortfolio, ...portfolio];
    const contributionsWithStart = [startingContributions, ...contributions];

    // Group data function
    const groupData = (data, groupBy) => {
      const grouped = {};
      data.forEach((entry) => {
        const date = new Date(entry.date);
        let key;
        switch (groupBy) {
          case "quarterly":
            key = `${date.getFullYear()}-Q${
              Math.floor(date.getMonth() / 3) + 1
            }`;
            break;
          case "yearly":
            key = `${date.getFullYear()}`;
            break;
          default: // monthly
            key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(
              2,
              "0"
            )}`;
        }
        if (
          !grouped[key] ||
          new Date(entry.date) > new Date(grouped[key].date)
        ) {
          grouped[key] = entry;
        }
      });
      return Object.values(grouped).sort(
        (a, b) => new Date(a.date) - new Date(b.date)
      );
    };

    // Use data with starting values
    const groupedPortfolio = groupData(portfolioWithStart, viewMode);
    const groupedContributions = groupData(contributionsWithStart, viewMode);

    // Generate labels
    const labels = groupedPortfolio.map((entry) => {
      const date = new Date(entry.date);
      switch (viewMode) {
        case "quarterly":
          return `Q${
            Math.floor(date.getMonth() / 3) + 1
          } ${date.getFullYear()}`;
        case "yearly":
          return date.getFullYear().toString();
        default:
          return date.toLocaleDateString("en-GB", {
            year: "numeric",
            month: "short",
          });
      }
    });

    const portfolioValues = groupedPortfolio.map((entry) => entry.value);

    // FIXED: Calculate cumulative contributions properly
    const contributionValues = groupedPortfolio.map((portfolioEntry, index) => {
      // For the starting point, return lump sum
      if (index === 0) {
        return lumpSum;
      }

      // Calculate cumulative contributions up to this point
      const monthlyContribution = portfolioData?.monthly || 0;
      const monthsElapsed = index; // Approximation based on timeline position
      return lumpSum + monthlyContribution * monthsElapsed;
    });

    // Calculate gain/loss
    const gainLossValues = portfolioValues.map(
      (portfolio, index) => portfolio - (contributionValues[index] || 0)
    );

    // Build target line
    const targetValue = portfolioData?.target_value;
    const targetLine = targetValue
      ? new Array(labels.length).fill(targetValue)
      : [];

    return {
      labels,
      portfolioValues,
      contributionValues,
      gainLossValues,
      targetLine,
      isEmpty: false,
    };
  }, [portfolioData, viewMode]);

  console.log(
    "DEBUG portfolioData.timeline type:",
    typeof portfolioData?.timeline,
    portfolioData?.timeline
  );

  if (processedData.isEmpty) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-4 md:p-8 text-center">
        <div className="text-gray-400 text-4xl md:text-6xl mb-4">📈</div>
        <h3 className="text-base md:text-lg font-semibold text-gray-700 mb-2">
          No Portfolio Data Available
        </h3>
        <p className="text-sm md:text-base text-gray-500">
          Complete your onboarding to see your portfolio growth chart.
        </p>
      </div>
    );
  }

  // Memoized calculation of summary metrics for display
  const metrics = useMemo(() => {
    return calculateReturns(portfolioData);
  }, [portfolioData]);

  // Build the Chart.js datasets array based on toggles and processed data
  const datasets = [
    {
      label: "Portfolio Value",
      data: processedData.portfolioValues,
      fill: showGainLoss ? "+1" : false,
      borderColor: "#10B981",
      backgroundColor: showGainLoss ? "rgba(16, 185, 129, 0.1)" : "transparent",
      borderWidth: 3,
      pointBackgroundColor: "#10B981",
      pointBorderColor: "#ffffff",
      pointBorderWidth: 2,
      pointRadius: 4,
      pointHoverRadius: 6,
      tension: 0.4,
    },
    {
      label: "Total Contributions",
      data: processedData.contributionValues,
      fill: false,
      borderColor: "#6B7280",
      backgroundColor: "rgba(107, 114, 128, 0.1)",
      borderWidth: 2,
      borderDash: [5, 5],
      pointBackgroundColor: "#6B7280",
      pointBorderColor: "#ffffff",
      pointBorderWidth: 1,
      pointRadius: 3,
      tension: 0.2,
    },
  ];

  // add target line overlay
  if (showTarget && processedData.targetLine.length > 0) {
    datasets.push({
      label: "Target Amount",
      data: processedData.targetLine,
      fill: false,
      borderColor: "#F59E0B",
      backgroundColor: "transparent",
      borderWidth: 2,
      borderDash: [10, 5],
      pointRadius: 0,
      pointHoverRadius: 0,
      tension: 0,
    });
  }

  // add gain/loss area overlay (for area under the curve effect)
  if (showGainLoss) {
    datasets.push({
      label: "Gain/Loss Area",
      data: processedData.contributionValues,
      fill: true,
      borderColor: "transparent",
      backgroundColor: "rgba(107, 114, 128, 0.05)",
      pointRadius: 0,
      pointHoverRadius: 0,
      tension: 0.2,
      order: 10,
    });
  }

  // Prepare data object for Chart.js
  const data = {
    labels: processedData.labels,
    datasets,
  };

  // Calculate y-axis min/max for better chart scaling
  const allValues = [
    ...processedData.portfolioValues,
    ...processedData.contributionValues,
    ...(showTarget ? processedData.targetLine : []),
  ].filter((v) => v != null && v > 0);

  const minY = Math.min(...allValues) * 0.95;
  const maxY = Math.max(...allValues) * 1.05;

  // Chart.js options for styling, tooltips, axes, and legend - MOBILE OPTIMIZED
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index",
      intersect: false,
    },
    plugins: {
      legend: {
        display: true,
        position: "top",
        labels: {
          usePointStyle: true,
          padding: window.innerWidth < 768 ? 10 : 20,
          font: {
            family: "system-ui, -apple-system, sans-serif",
            size: window.innerWidth < 768 ? 10 : 12,
          },
          filter: (legendItem) => {
            // Hide the gain/loss area from legend
            return legendItem.text !== "Gain/Loss Area";
          },
        },
      },
      tooltip: {
        backgroundColor: "rgba(0, 0, 0, 0.8)",
        titleColor: "#ffffff",
        bodyColor: "#ffffff",
        borderColor: "#374151",
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        titleFont: {
          size: window.innerWidth < 768 ? 12 : 14,
        },
        bodyFont: {
          size: window.innerWidth < 768 ? 11 : 13,
        },
        callbacks: {
          title: (context) => `Period: ${context[0].label}`,
          label: (context) => {
            const value = context.parsed.y;
            const label = context.dataset.label;
            // Custom tooltip for portfolio value to show gain/loss
            if (label === "Portfolio Value") {
              const contributionValue =
                processedData.contributionValues[context.dataIndex];
              const gainLoss = value - contributionValue;
              const gainLossPercent =
                contributionValue > 0
                  ? (gainLoss / contributionValue) * 100
                  : 0;
              return [
                `${label}: ${formatCurrency(value)}`,
                `Gain/Loss: ${gainLoss >= 0 ? "+" : ""}${formatCurrency(
                  gainLoss
                )} (${gainLossPercent >= 0 ? "+" : ""}${gainLossPercent.toFixed(
                  1
                )}%)`,
              ];
            }
            return `${label}: ${formatCurrency(value)}`;
          },
        },
      },
    },
    scales: {
      x: {
        title: {
          display: window.innerWidth >= 768, // Hide title on mobile
          text: "Time Period",
          font: {
            family: "system-ui, -apple-system, sans-serif",
            size: window.innerWidth < 768 ? 12 : 14,
            weight: "bold",
          },
        },
        grid: {
          color: "rgba(0, 0, 0, 0.05)",
        },
        ticks: {
          font: {
            family: "system-ui, -apple-system, sans-serif",
            size: window.innerWidth < 768 ? 10 : 12,
          },
          maxTicksLimit: window.innerWidth < 768 ? 6 : 12, // Fewer ticks on mobile
        },
      },
      y: {
        title: {
          display: window.innerWidth >= 768, // Hide title on mobile
          text: "Value (£)",
          font: {
            family: "system-ui, -apple-system, sans-serif",
            size: window.innerWidth < 768 ? 12 : 14,
            weight: "bold",
          },
        },
        min: minY,
        max: maxY,
        grid: {
          color: "rgba(0, 0, 0, 0.05)",
        },
        ticks: {
          callback: (value) => {
            // Shorter currency format on mobile
            if (window.innerWidth < 768) {
              return value >= 1000
                ? `£${(value / 1000).toFixed(0)}k`
                : `£${value.toFixed(0)}`;
            }
            return formatCurrency(value);
          },
          font: {
            family: "system-ui, -apple-system, sans-serif",
            size: window.innerWidth < 768 ? 10 : 12,
          },
          maxTicksLimit: window.innerWidth < 768 ? 6 : 8, // Fewer ticks on mobile
        },
      },
    },
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-3 md:p-6 mb-4 md:mb-6">
      {/* Header with controls - MOBILE RESPONSIVE */}
      <div className="flex flex-col space-y-3 md:space-y-4 mb-4 md:mb-6">
        {/* Title and metrics */}
        <div>
          <h3 className="text-lg md:text-xl font-bold text-gray-800 mb-2">
            Portfolio Growth
          </h3>
          {metrics.currentValue && (
            <div className="flex flex-wrap items-center gap-2 md:gap-4 text-xs md:text-sm">
              <span className="text-gray-600">
                Current:{" "}
                <span className="font-semibold text-green-600">
                  {formatCurrency(metrics.currentValue)}
                </span>
              </span>
              <span className="text-gray-600">
                Return:{" "}
                <span
                  className={`font-semibold ${
                    metrics.totalReturnPercent >= 0
                      ? "text-green-600"
                      : "text-red-600"
                  }`}
                >
                  {formatPercentage(metrics.totalReturnPercent)}
                </span>
              </span>
              <span className="text-gray-600 hidden md:inline">
                {metrics.dataPoints} data points
              </span>
            </div>
          )}
        </div>

        {/* Controls - Stacked on mobile */}
        <div className="flex flex-col space-y-3 md:flex-row md:items-center md:justify-between md:space-y-0">
          {/* View Mode Selector buttons */}
          <div className="flex bg-gray-100 rounded-lg p-1 w-full md:w-auto">
            {["monthly", "quarterly", "yearly"].map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`flex-1 md:flex-none px-2 md:px-3 py-1 text-xs md:text-sm font-medium rounded-md transition-colors ${
                  viewMode === mode
                    ? "bg-white text-blue-600 shadow-sm"
                    : "text-gray-600 hover:text-gray-800"
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>

          {/* Overlay toggles - Stacked on mobile */}
          <div className="flex flex-col space-y-2 md:flex-row md:items-center md:space-y-0 md:space-x-3">
            {portfolioData?.target_value && (
              <label className="flex items-center space-x-2 text-xs md:text-sm">
                <input
                  type="checkbox"
                  checked={showTarget}
                  onChange={(e) => setShowTarget(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-gray-700">Target Line</span>
              </label>
            )}

            <label className="flex items-center space-x-2 text-xs md:text-sm">
              <input
                type="checkbox"
                checked={showGainLoss}
                onChange={(e) => setShowGainLoss(e.target.checked)}
                className="rounded border-gray-300 text-green-600 focus:ring-green-500"
              />
              <span className="text-gray-700">Gain/Loss Area</span>
            </label>
          </div>
        </div>
      </div>

      {/* Performance summary cards - MOBILE RESPONSIVE GRID */}
      {metrics.currentValue && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 md:gap-4 mb-4 md:mb-6">
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-2 md:p-4 rounded-lg border border-green-200">
            <div className="text-xs md:text-sm text-green-600 font-medium">
              Portfolio Value
            </div>
            <div className="text-sm md:text-lg font-bold text-green-700">
              {window.innerWidth < 768
                ? `£${(metrics.currentValue / 1000).toFixed(0)}k`
                : formatCurrency(metrics.currentValue)}
            </div>
          </div>

          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-2 md:p-4 rounded-lg border border-blue-200">
            <div className="text-xs md:text-sm text-blue-600 font-medium">
              Contributions
            </div>
            <div className="text-sm md:text-lg font-bold text-blue-700">
              {window.innerWidth < 768
                ? `£${(metrics.currentContributions / 1000).toFixed(0)}k`
                : formatCurrency(metrics.currentContributions)}
            </div>
          </div>

          <div
            className={`p-2 md:p-4 rounded-lg border ${
              metrics.totalGainLoss >= 0
                ? "bg-gradient-to-r from-green-50 to-emerald-50 border-green-200"
                : "bg-gradient-to-r from-red-50 to-rose-50 border-red-200"
            }`}
          >
            <div
              className={`text-xs md:text-sm font-medium ${
                metrics.totalGainLoss >= 0 ? "text-green-600" : "text-red-600"
              }`}
            >
              Total {metrics.totalGainLoss >= 0 ? "Gain" : "Loss"}
            </div>
            <div
              className={`text-sm md:text-lg font-bold ${
                metrics.totalGainLoss >= 0 ? "text-green-700" : "text-red-700"
              }`}
            >
              {metrics.totalGainLoss >= 0 ? "+" : ""}
              {window.innerWidth < 768
                ? `£${(Math.abs(metrics.totalGainLoss) / 1000).toFixed(1)}k`
                : formatCurrency(metrics.totalGainLoss)}
            </div>
          </div>

          <div
            className={`p-2 md:p-4 rounded-lg border ${
              metrics.totalReturnPercent >= 0
                ? "bg-gradient-to-r from-green-50 to-emerald-50 border-green-200"
                : "bg-gradient-to-r from-red-50 to-rose-50 border-red-200"
            }`}
          >
            <div
              className={`text-xs md:text-sm font-medium ${
                metrics.totalReturnPercent >= 0
                  ? "text-green-600"
                  : "text-red-600"
              }`}
            >
              Total Return
            </div>
            <div
              className={`text-sm md:text-lg font-bold ${
                metrics.totalReturnPercent >= 0
                  ? "text-green-700"
                  : "text-red-700"
              }`}
            >
              {formatPercentage(metrics.totalReturnPercent)}
            </div>
          </div>
        </div>
      )}

      {/* Main Chart.js line chart - MOBILE RESPONSIVE HEIGHT */}
      <div className="h-64 md:h-96 w-full">
        <Line data={data} options={options} />
      </div>

      {/* Chart legend/help area - MOBILE RESPONSIVE */}
      <div className="mt-3 md:mt-4 text-xs text-gray-500 bg-gray-50 p-2 md:p-3 rounded-lg">
        <div className="flex flex-col space-y-1 md:flex-row md:flex-wrap md:items-center md:justify-center md:space-y-0 md:space-x-6">
          <div className="flex items-center space-x-1">
            <div className="w-3 h-0.5 bg-green-500"></div>
            <span>Portfolio Value (includes gains/losses)</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-0.5 bg-gray-500 border-dashed border-t"></div>
            <span>Total Contributions (money you put in)</span>
          </div>
          {showTarget && portfolioData?.target_value && (
            <div className="flex items-center space-x-1">
              <div className="w-3 h-0.5 bg-yellow-500 border-dashed border-t"></div>
              <span>Target Amount</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PortfolioGraph;
