import { useContext, useState, useMemo } from "react";
import PortfolioContext from "../../context/PortfolioContext";
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

const PortfolioGraph = () => {
  const { portfolioData } = useContext(PortfolioContext);
  const [viewMode, setViewMode] = useState("monthly"); // monthly, quarterly, yearly
  const [showTarget, setShowTarget] = useState(true);
  const [showGainLoss, setShowGainLoss] = useState(true);

  // Calculate percentage change
  const calculatePercentChange = (current, previous) => {
    if (!previous || previous === 0) return 0;
    return ((current - previous) / previous) * 100;
  };

  const processedData = useMemo(() => {
    const portfolio = portfolioData?.results?.timeline?.portfolio ?? [];
    const contributions = portfolioData?.results?.timeline?.contributions ?? [];

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

    // Group data by time period based on view mode
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

    const groupedPortfolio = groupData(portfolio, viewMode);
    const groupedContributions = groupData(contributions, viewMode);

    // Create labels based on view mode
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

    // Match contributions to portfolio timeline
    const contributionValues = groupedPortfolio.map((portfolioEntry) => {
      const matchingContribution = groupedContributions.find(
        (contrib) =>
          Math.abs(new Date(contrib.date) - new Date(portfolioEntry.date)) <
          30 * 24 * 60 * 60 * 1000
      );
      return matchingContribution?.value || 0;
    });

    // Calculate gain/loss (portfolio value - contributions)
    const gainLossValues = portfolioValues.map(
      (portfolio, index) => portfolio - (contributionValues[index] || 0)
    );

    // Target line (if target is set)
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

  if (processedData.isEmpty) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-8 text-center">
        <div className="text-gray-400 text-6xl mb-4">📈</div>
        <h3 className="text-lg font-semibold text-gray-700 mb-2">
          No Portfolio Data Available
        </h3>
        <p className="text-gray-500">
          Complete your onboarding to see your portfolio growth chart.
        </p>
      </div>
    );
  }

  // Calculate key metrics for display
  const metrics = useMemo(() => {
    return calculateReturns(portfolioData);
  }, [portfolioData]);

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

  // Add target line if enabled and target exists
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

  // Add gain/loss area if enabled
  if (showGainLoss) {
    datasets.push({
      label: "Gain/Loss Area",
      data: processedData.contributionValues,
      fill: false,
      borderColor: "transparent",
      backgroundColor: "rgba(107, 114, 128, 0.05)",
      pointRadius: 0,
      pointHoverRadius: 0,
      tension: 0.2,
      order: 10,
    });
  }

  const data = {
    labels: processedData.labels,
    datasets,
  };

  const allValues = [
    ...processedData.portfolioValues,
    ...processedData.contributionValues,
    ...(showTarget ? processedData.targetLine : []),
  ].filter((v) => v != null && v > 0);

  const minY = Math.min(...allValues) * 0.95;
  const maxY = Math.max(...allValues) * 1.05;

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
          padding: 20,
          font: {
            family: "system-ui, -apple-system, sans-serif",
            size: 12,
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
        callbacks: {
          title: (context) => `Period: ${context[0].label}`,
          label: (context) => {
            const value = context.parsed.y;
            const label = context.dataset.label;

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
          display: true,
          text: "Time Period",
          font: {
            family: "system-ui, -apple-system, sans-serif",
            size: 14,
            weight: "bold",
          },
        },
        grid: {
          color: "rgba(0, 0, 0, 0.05)",
        },
        ticks: {
          font: {
            family: "system-ui, -apple-system, sans-serif",
          },
        },
      },
      y: {
        title: {
          display: true,
          text: "Value (£)",
          font: {
            family: "system-ui, -apple-system, sans-serif",
            size: 14,
            weight: "bold",
          },
        },
        min: minY,
        max: maxY,
        grid: {
          color: "rgba(0, 0, 0, 0.05)",
        },
        ticks: {
          callback: (value) => formatCurrency(value),
          font: {
            family: "system-ui, -apple-system, sans-serif",
          },
        },
      },
    },
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
      {/* Header with controls */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 space-y-4 sm:space-y-0">
        <div>
          <h3 className="text-xl font-bold text-gray-800 mb-2">
            Portfolio Growth
          </h3>
          {metrics.currentValue && (
            <div className="flex items-center space-x-4 text-sm">
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
              <span className="text-gray-600">
                {metrics.dataPoints} data points
              </span>
            </div>
          )}
        </div>

        <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-2 sm:space-y-0 sm:space-x-4">
          {/* View Mode Selector */}
          <div className="flex bg-gray-100 rounded-lg p-1">
            {["monthly", "quarterly", "yearly"].map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                  viewMode === mode
                    ? "bg-white text-blue-600 shadow-sm"
                    : "text-gray-600 hover:text-gray-800"
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>

          {/* Toggle Controls */}
          <div className="flex items-center space-x-3">
            {portfolioData?.target_value && (
              <label className="flex items-center space-x-2 text-sm">
                <input
                  type="checkbox"
                  checked={showTarget}
                  onChange={(e) => setShowTarget(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-gray-700">Target Line</span>
              </label>
            )}

            <label className="flex items-center space-x-2 text-sm">
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

      {/* Performance Summary */}
      {metrics.currentValue && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200">
            <div className="text-sm text-green-600 font-medium">
              Portfolio Value
            </div>
            <div className="text-lg font-bold text-green-700">
              {formatCurrency(metrics.currentValue)}
            </div>
          </div>

          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
            <div className="text-sm text-blue-600 font-medium">
              Contributions
            </div>
            <div className="text-lg font-bold text-blue-700">
              {formatCurrency(metrics.currentContributions)}
            </div>
          </div>

          <div
            className={`p-4 rounded-lg border ${
              metrics.totalGainLoss >= 0
                ? "bg-gradient-to-r from-green-50 to-emerald-50 border-green-200"
                : "bg-gradient-to-r from-red-50 to-rose-50 border-red-200"
            }`}
          >
            <div
              className={`text-sm font-medium ${
                metrics.totalGainLoss >= 0 ? "text-green-600" : "text-red-600"
              }`}
            >
              Total {metrics.totalGainLoss >= 0 ? "Gain" : "Loss"}
            </div>
            <div
              className={`text-lg font-bold ${
                metrics.totalGainLoss >= 0 ? "text-green-700" : "text-red-700"
              }`}
            >
              {metrics.totalGainLoss >= 0 ? "+" : ""}
              {formatCurrency(metrics.totalGainLoss)}
            </div>
          </div>

          <div
            className={`p-4 rounded-lg border ${
              metrics.totalReturnPercent >= 0
                ? "bg-gradient-to-r from-green-50 to-emerald-50 border-green-200"
                : "bg-gradient-to-r from-red-50 to-rose-50 border-red-200"
            }`}
          >
            <div
              className={`text-sm font-medium ${
                metrics.totalReturnPercent >= 0
                  ? "text-green-600"
                  : "text-red-600"
              }`}
            >
              Total Return
            </div>
            <div
              className={`text-lg font-bold ${
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

      {/* Chart */}
      <div className="h-96 w-full">
        <Line data={data} options={options} />
      </div>

      {/* Chart Legend/Help */}
      <div className="mt-4 text-xs text-gray-500 bg-gray-50 p-3 rounded-lg">
        <div className="flex flex-wrap items-center justify-center space-x-6">
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
