import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { usePortfolio } from "../../context/PortfolioContext";

export default function AIPortfolioSummary() {
  const { portfolioData, isLoading, error } = usePortfolio();

  if (isLoading) return <p>Loading portfolio summary...</p>;
  if (error) return <p>Error loading portfolio data: {error}</p>;

  const aiAnalysis = portfolioData?.ai_analysis;

  if (!portfolioData || !aiAnalysis) {
    return <p>No AI analysis available yet.</p>;
  }

  const formatMarkdown = (text) => {
    if (!text) return "";
    return typeof text === "string" ? text.replace(/\\n/g, "\n") : String(text);
  };

  return (
    <div className="mb-8 mx-auto max-w-6xl">
      <h2 className="text-3xl font-semibold mb-8 text-gray-900 text-center">
        AI Portfolio Analysis
      </h2>

      {/* Main Summary */}
      {aiAnalysis.summary && (
        <div className="bg-white p-8 rounded-lg shadow-md border border-gray-200 prose prose-lg prose-indigo max-w-full mx-auto mb-8">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {formatMarkdown(aiAnalysis.summary)}
          </ReactMarkdown>
        </div>
      )}
    </div>
  );
}
