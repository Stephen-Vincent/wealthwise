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
  // Debug: inspect the full AI analysis payload when not in production
  if (process.env.NODE_ENV !== "production") {
    // eslint-disable-next-line no-console
    console.log("[AIPortfolioSummary] ai_analysis:", aiAnalysis);
  }

  if (!portfolioData || !aiAnalysis) {
    return (
      <p>
        No AI analysis found for this simulation. Try opening a newer run or
        regenerate analysis.
      </p>
    );
  }

  const normalizeToMarkdown = (value) => {
    if (value == null) return "";

    // If backend wrapped the summary in a known key
    if (typeof value === "object" && !Array.isArray(value)) {
      const possible =
        value.markdown || value.text || value.content || value.summary;
      if (possible != null) return normalizeToMarkdown(possible);
    }

    if (typeof value === "string") {
      // Unescape literal "\n" into real newlines; trim trailing spaces
      return value.replace(/\\n/g, "\n").trim();
    }

    if (Array.isArray(value)) {
      // Join array items; if items are objects, stringify them prettily
      return value
        .map((item) => {
          if (item == null) return "";
          if (typeof item === "string") return item;
          try {
            return "```\n" + JSON.stringify(item, null, 2) + "\n```";
          } catch {
            return String(item);
          }
        })
        .filter(Boolean)
        .join("\n\n");
    }

    // Fallback: objects/numbers/booleans → pretty JSON as a fenced block
    try {
      return "```\n" + JSON.stringify(value, null, 2) + "\n```";
    } catch {
      return String(value);
    }
  };

  const summaryText = normalizeToMarkdown(aiAnalysis.summary);

  if (!summaryText.trim()) {
    return (
      <p>
        AI analysis is present but empty. Please regenerate or check server
        logs.
      </p>
    );
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
