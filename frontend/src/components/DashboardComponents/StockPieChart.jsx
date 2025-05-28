import React, { useEffect, useState, useContext } from "react";
import PropTypes from "prop-types";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Pie } from "react-chartjs-2";
import PortfolioContext from "../../context/PortfolioContext";

// Register chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

export default function StockPieChart() {
  const { portfolioData } = useContext(PortfolioContext);
  const breakdown = portfolioData?.breakdown;

  const [stockNameMap, setStockNameMap] = useState({});

  useEffect(() => {
    fetch("http://localhost:8000/stock-name-map")
      .then((res) => res.json())
      .then((data) => setStockNameMap(data))
      .catch((err) => console.error("Failed to fetch stock name map:", err));
  }, []);

  if (!breakdown || typeof breakdown !== "object") {
    return (
      <p className="text-gray-500 text-center">
        No portfolio breakdown available.
      </p>
    );
  }

  const tickers = Object.keys(breakdown);
  const rawValues = Object.values(breakdown);
  const total = rawValues.reduce((sum, val) => sum + val, 0);

  const labels = tickers.map((ticker, i) => {
    const name = stockNameMap[ticker]?.name || ticker;
    const percent = ((rawValues[i] / total) * 100).toFixed(2);
    return `${name} - ${percent}%`;
  });

  if (total === 0) {
    return (
      <p className="text-gray-500 text-center">
        Portfolio breakdown has zero total value.
      </p>
    );
  }

  const data = {
    labels,
    datasets: [
      {
        label: "Portfolio Allocation (%)",
        data: rawValues.map((val) => ((val / total) * 100).toFixed(2)),
        backgroundColor: [
          "#4F46E5",
          "#10B981",
          "#F59E0B",
          "#3B82F6",
          "#EF4444",
          "#8B5CF6",
        ],
        borderWidth: 3,
        borderColor: "#fff",
        hoverOffset: 10,
      },
    ],
  };

  const options = {
    responsive: true,
    layout: {
      padding: 20,
    },
    plugins: {
      legend: {
        position: "right",
        labels: {
          generateLabels: function (chart) {
            const data = chart.data;
            if (!data.labels || !data.datasets.length) return [];

            return data.labels.map((label, i) => {
              const backgroundColor = data.datasets[0].backgroundColor[i];
              return {
                text: label,
                fillStyle: backgroundColor,
                strokeStyle: backgroundColor,
                index: i,
              };
            });
          },
        },
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            const ticker = Object.keys(breakdown)[context.dataIndex];
            const companyName = stockNameMap[ticker]?.name || ticker;
            const value = context.parsed;
            return `${companyName} - ${value}%`;
          },
        },
      },
      shadow: {
        enabled: true,
        color: "rgba(0, 0, 0, 0.3)",
        blur: 10,
        offsetX: 0,
        offsetY: 4,
      },
    },
  };

  return (
    <div className="mb-6">
      <h3 className="text-lg font-semibold text-center mb-4">
        Capital Allocation by Stock
      </h3>
      <div
        className="flex items-center justify-center mx-auto shadow-xl rounded-xl p-6 backdrop-blur-md"
        style={{ backgroundColor: "rgba(255, 255, 255, 0.7)" }}
      >
        <Pie data={data} options={options} width={200} height={200} />
      </div>
    </div>
  );
}
