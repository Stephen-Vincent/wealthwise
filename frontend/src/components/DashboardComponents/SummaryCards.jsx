import { useContext } from "react";
import PortfolioContext from "../../context/PortfolioContext";

export default function SummaryCards() {
  const { portfolioData } = useContext(PortfolioContext);

  const returnPercent =
    portfolioData?.total_start && portfolioData?.total_end
      ? (
          ((portfolioData.total_end - portfolioData.total_start) /
            portfolioData.total_start) *
          100
        ).toFixed(2)
      : "N/A";

  return (
    <div className="mb-8">
      <h3 className="text-xl font-bold mb-4">My Summary</h3>
      <p className="mb-4 text-gray-700">
        <strong>My Goal:</strong> {portfolioData?.goal || "Not set"}
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Risk Score", value: portfolioData?.risk || "N/A" },
          {
            label: "Starting Balance",
            value: `£${portfolioData?.total_start?.toLocaleString() || 0}`,
          },
          {
            label: "Current Balance",
            value: `£${portfolioData?.final_balance?.toLocaleString() || 0}`,
          },
          { label: "Return %", value: `${returnPercent}%` },
        ].map((card, index) => (
          <div
            key={index}
            className="p-4 bg-white shadow rounded-lg text-center border border-gray-200"
          >
            <div className="text-sm text-gray-500">{card.label}</div>
            <div className="text-xl font-bold text-gray-900">{card.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
