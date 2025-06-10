import { useContext } from "react";
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
} from "chart.js";

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
  Legend
);

const PortfolioGraph = () => {
  const { portfolioData } = useContext(PortfolioContext);

  const portfolio = portfolioData?.results?.timeline?.portfolio ?? [];
  const contributions = portfolioData?.results?.timeline?.contributions ?? [];

  if (portfolio.length === 0 || contributions.length === 0) {
    return <div>Loading graph...</div>;
  }

  // Aggregate portfolio by latest entry in each month
  const monthlyData = {};
  portfolio.forEach((entry) => {
    const date = new Date(entry.date);
    const key = `${date.getFullYear()}-${date.getMonth()}`;
    monthlyData[key] = entry; // overwrite with latest entry in the month
  });

  const sortedMonthlyEntries = Object.values(monthlyData).sort(
    (a, b) => new Date(a.date) - new Date(b.date)
  );

  const labels = sortedMonthlyEntries.map((entry) =>
    new Date(entry.date).toLocaleDateString("en-GB", {
      year: "numeric",
      month: "short",
    })
  );

  const portfolioValues = sortedMonthlyEntries.map((entry) => entry.value);

  const lumpSum = portfolioData?.lump_sum ?? 0;
  const monthly = portfolioData?.monthly ?? 0;
  const firstDate = new Date(sortedMonthlyEntries[0]?.date);
  const contributionValuesByMonth = labels.map((_, index) => {
    return lumpSum + monthly * index;
  });

  const data = {
    labels,
    datasets: [
      {
        label: "Portfolio Value",
        data: portfolioValues,
        fill: false,
        borderColor: "rgb(75, 192, 192)",
        tension: 0.1,
      },
      {
        label: "Contributions",
        data: contributionValuesByMonth,
        fill: false,
        borderColor: "grey",
        borderDash: [5, 5],
        tension: 0.1,
      },
    ],
  };

  // Calculate min and max y values dynamically based on data
  const allYValues = [
    ...data.datasets[0].data,
    ...data.datasets[1].data,
  ].filter((v) => v !== null);

  const minY = Math.min(...allYValues);
  const maxY = Math.max(...allYValues);

  const options = {
    responsive: true,
    plugins: {
      legend: {
        display: true,
      },
      title: {
        display: false,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Date",
        },
      },
      y: {
        title: {
          display: true,
          text: "Value £",
        },
        min: minY,
        max: maxY,
      },
    },
  };

  return (
    <div className="graph-container">
      <Line data={data} options={options} />
    </div>
  );
};

export default PortfolioGraph;
