import { useContext } from "react";
import PortfolioContext from "../../context/PortfolioContext";

export default function SummaryCards() {
  const { portfolioData } = useContext(PortfolioContext);
  return (
    <div className="mb-8">
      <h3 className="text-xl font-bold mb-4">My Summary</h3>
      <p className="mb-4 text-gray-700">
        <strong>My Goal:</strong> {portfolioData?.goal || "Not set"}
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Risk Score", value: "65" },
          {
            label: "Starting Balance",
            value: `£${portfolioData?.lump_sum || 0}`,
          },
          { label: "Current Balance", value: "£4,500" },
          { label: "Return %", value: "200%" },
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
