/**
 * AIPortfolioSummary Component
 *
 * Renders an AI-generated summary of the user's portfolio.
 * - Reads `portfolioData` from PortfolioContext
 * - Shows loading/error states
 * - Safely normalizes `ai_analysis.summary` to a string for react-markdown
 */

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { usePortfolio } from "../../context/PortfolioContext";

export default function AIPortfolioSummary() {
  const { portfolioData, isLoading, error } = usePortfolio();

  if (isLoading) return <p>Loading portfolio summary...</p>;
  if (error) return <p>Error loading portfolio data: {String(error)}</p>;

  const aiAnalysis = portfolioData?.ai_analysis;

  if (!portfolioData || !aiAnalysis) {
    return <p>No AI analysis available yet.</p>;
  }

  const normalizeToMarkdown = (value) => {
    if (value == null) return "";
    if (typeof value === "string") {
      // unescape literal "\n" into real newlines
      return value.replace(/\\n/g, "\n");
    }
    if (Array.isArray(value)) {
      // join array parts as paragraphs
      return value.map(normalizeToMarkdown).join("\n\n");
    }
    // objects/numbers/bools → pretty JSON as a fenced block
    try {
      return "```\n" + JSON.stringify(value, null, 2) + "\n```";
    } catch {
      return String(value);
    }
  };

  const summaryText = normalizeToMarkdown(aiAnalysis.summary);

  if (!summaryText.trim()) {
    return <p>No AI analysis available yet.</p>;
  }

  return (
    <div className="mb-8 mx-auto max-w-6xl">
      <h2 className="text-3xl font-semibold mb-8 text-gray-900 text-center">
        AI Portfolio Analysis
      </h2>

      <div className="bg-white p-8 rounded-lg shadow-md border border-gray-200 prose prose-lg prose-indigo max-w-full mx-auto mb-8">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{summaryText}</ReactMarkdown>
      </div>
    </div>
  );
}
