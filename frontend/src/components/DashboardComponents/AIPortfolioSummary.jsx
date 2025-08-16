/**
 * AIPortfolioSummary.jsx
 * ----------------------
 * Displays an AI-generated summary of the user's portfolio.
 * - Fetches summary data from PortfolioContext
 * - Renders summary as markdown with GitHub-flavored markdown support
 * - Handles loading, error, and missing data states
 */

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { usePortfolio } from "../../context/PortfolioContext";

export default function AIPortfolioSummary() {
  const { portfolioData, isLoading, error } = usePortfolio();

  if (isLoading) return <p>Loading portfolio summary...</p>;
  if (error) return <p>Error loading portfolio data: {error}</p>;
  if (!portfolioData || !portfolioData.ai_summary)
    return <p>No AI summary available yet.</p>;

  let rawSummary = portfolioData.ai_summary;

  if (Array.isArray(rawSummary)) {
    rawSummary = rawSummary.join("\n\n");
  } else if (typeof rawSummary !== "string") {
    rawSummary = String(rawSummary);
  }

  const decodedSummary = rawSummary.replace(/\\n/g, "\n");

  return (
    <div className="mb-8 mx-auto">
      <h2 className="text-3xl font-semibold mb-8 text-gray-900 text-center">
        AI Portfolio Summary
      </h2>
      <div
        className="
          bg-white 
          p-8 
          rounded-lg 
          shadow-md
          border border-gray-200
          prose
          prose-lg
          prose-indigo
          max-w-full
          mx-auto
        "
      >
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {decodedSummary}
        </ReactMarkdown>
      </div>
    </div>
  );
}
