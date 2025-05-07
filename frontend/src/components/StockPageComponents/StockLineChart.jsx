import React, { useState } from "react";
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

const mockData = {
  labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
  datasets: [
    {
      label: "Holdings Value",
      data: [500, 600, 700, 900, 850, 1000],
      fill: false,
      borderColor: "#00A8FF",
      tension: 0.4,
    },
  ],
};

const options = {
  responsive: true,
  plugins: {
    legend: {
      display: false,
    },
  },
  animation: {
    duration: 1000,
    easing: "easeInOutQuart",
  },
};

export default function StockLineChart() {
  const [timeframe, setTimeframe] = useState("All Time");

  return (
    <section className="px-6 py-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">Your Holdings</h2>
        <select
          className="border border-gray-300 rounded-md px-3 py-1 text-sm"
          value={timeframe}
          onChange={(e) => setTimeframe(e.target.value)}
        >
          <option>All Time</option>
          <option>1Y</option>
          <option>6M</option>
          <option>1M</option>
        </select>
      </div>
      <div className="bg-gray-100 rounded p-4">
        <Line data={mockData} options={options} />
      </div>
    </section>
  );
}
