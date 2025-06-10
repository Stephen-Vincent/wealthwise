import React, { useEffect, useState, useContext } from "react";
import PropTypes from "prop-types";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Pie } from "react-chartjs-2";
import PortfolioContext from "../../context/PortfolioContext";

// Register chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

export default function StockPieChart() {
  const { portfolioData } = useContext(PortfolioContext);
  const stocksPicked = portfolioData?.results?.stocks_picked || [];
  console.log("📊 StockPieChart stocks picked:", stocksPicked);

  if (!stocksPicked.length) {
    return (
      <p className="text-gray-500 text-center">
        No portfolio stock allocation available.
      </p>
    );
  }

  const tickers = stocksPicked.map((stock) => stock.symbol);
  const rawValues = stocksPicked.map((stock) => stock.allocation);
  const total = rawValues.reduce((sum, val) => sum + val, 0);

  const labels = tickers.map((ticker, i) => {
    const percent = (rawValues[i] * 100).toFixed(2);
    return `${ticker} - ${percent}%`;
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
        data: rawValues.map((val) => (val * 100).toFixed(2)),
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
            const ticker = tickers[context.dataIndex];
            return `${ticker} - ${context.parsed}%`;
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
        <Pie data={data} options={options} width={120} height={120} />
      </div>
    </div>
  );
}
