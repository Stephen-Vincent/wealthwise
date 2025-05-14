import { useContext } from "react";
import PortfolioContext from "../../context/PortfolioContext";

export default function AIPortfolioSummary() {
  const { portfolioData, isLoading, error } = useContext(PortfolioContext);

  if (isLoading) return <p>Loading portfolio summary...</p>;
  if (error) return <p>Error loading portfolio data: {error}</p>;
  if (!portfolioData) return <p>No portfolio data available yet.</p>;

  return (
    <div className="mb-8">
      <h3 className="text-xl font-bold mb-4">AI Portfolio Summary</h3>
      <div className="space-y-4 text-sm text-gray-800">
        <div>
          <h4 className="font-bold">Portfolio Overview:</h4>
          <p>{portfolioData.overview}</p>
        </div>
        <div>
          <h4 className="font-bold">Key Events Impacting Your Portfolio:</h4>
          <ul className="list-disc list-inside space-y-1">
            {portfolioData.keyEvents?.map((event, idx) => (
              <li key={idx}>{event}</li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="font-bold">Risk and Resilience:</h4>
          <p>{portfolioData.riskInsight}</p>
        </div>
        <div>
          <h4 className="font-bold">Final Insight:</h4>
          <p>{portfolioData.finalInsight}</p>
        </div>
      </div>
    </div>
  );
}
