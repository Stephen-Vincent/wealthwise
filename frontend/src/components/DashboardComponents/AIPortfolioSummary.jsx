import { usePortfolio } from "../../context/PortfolioContext";

export default function AIPortfolioSummary() {
  const { portfolioData, isLoading, error } = usePortfolio();
  console.log("🧠 AI Summary Received:", portfolioData);

  if (isLoading) return <p>Loading portfolio summary...</p>;
  if (error) return <p>Error loading portfolio data: {error}</p>;
  if (!portfolioData || !portfolioData.ai_summary)
    return <p>No AI summary available yet.</p>;

  return (
    <div className="mb-8">
      <h3 className="text-xl font-bold mb-4">AI Portfolio Summary</h3>
      <div className="whitespace-pre-line text-sm text-gray-800">
        {portfolioData.ai_summary}
      </div>
    </div>
  );
}
