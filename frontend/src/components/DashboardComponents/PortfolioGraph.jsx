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

const data = {
  labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
  datasets: [
    {
      label: "Portfolio Value",
      data: [1500, 1800, 2200, 2600, 3000, 5000],
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
    easing: "easeInOutQuad",
  },
};

export default function PortfolioGraph() {
  return (
    <div className="mb-8">
      <h3 className="text-lg font-semibold mb-2">Your Portfolio</h3>
      <p className="text-blue-600 text-lg font-bold mb-4">£5,000.00</p>
      <div className="bg-gray-100 rounded p-4">
        <Line data={data} options={options} />
      </div>
    </div>
  );
}
