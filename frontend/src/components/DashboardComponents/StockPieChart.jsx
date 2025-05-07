import React, { useRef } from "react";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Pie } from "react-chartjs-2";

// Register components
ChartJS.register(ArcElement, Tooltip, Legend);

const shadowPlugin = {
  id: "outerShadow",
  beforeDraw: (chart) => {
    const ctx = chart.ctx;
    ctx.save();
    chart.data.datasets.forEach((dataset, i) => {
      const meta = chart.getDatasetMeta(i);
      meta.data.forEach((element) => {
        ctx.shadowColor = "rgba(0, 0, 0, 0.2)";
        ctx.shadowBlur = 15;
        ctx.shadowOffsetX = 4;
        ctx.shadowOffsetY = 4;
        ctx.fill();
      });
    });
    ctx.restore();
  },
};

const data = {
  labels: ["Apple", "Google", "Meta", "Amazon"],
  datasets: [
    {
      label: "Portfolio Allocation",
      data: [30, 25, 25, 20],
      backgroundColor: [
        "rgba(0, 168, 255, 0.85)",
        "rgba(59, 130, 246, 0.85)",
        "rgba(124, 58, 237, 0.85)",
        "rgba(251, 191, 36, 0.85)",
      ],
      borderColor: "#ffffff",
      borderWidth: 2,
    },
  ],
};

const options = {
  responsive: true,
  plugins: {
    legend: {
      position: "right",
      labels: {
        usePointStyle: true,
        pointStyle: "circle",
      },
    },
  },
  animation: {
    duration: 1000,
    easing: "easeOutBounce",
  },
};

export default function StockPieChart({ onSliceClick }) {
  const chartRef = useRef(null);

  const handleClick = (event) => {
    const chart = chartRef.current;
    if (!chart) return;
    const points = chart.getElementsAtEventForMode(
      event,
      "nearest",
      { intersect: true },
      true
    );
    if (points.length > 0) {
      const index = points[0].index;
      const label = data.labels[index];
      onSliceClick?.(label);
    }
  };

  return (
    <div className="mb-8">
      <h3 className="text-lg font-semibold mb-2">My Stocks</h3>
      <p className="text-sm mb-4 text-gray-700">
        Tap on any slice of the pie to dive deeper into each stock’s story and
        see how it performed over time.
      </p>
      <div className="flex flex-col md:flex-row md:items-center md:space-x-8 justify-center">
        <div className="mx-auto w-full max-w-md">
          <Pie
            ref={chartRef}
            data={data}
            options={options}
            plugins={[shadowPlugin]}
            onClick={handleClick}
          />
        </div>
      </div>
    </div>
  );
}
