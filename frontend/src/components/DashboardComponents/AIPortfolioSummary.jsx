/**
 * AIPortfolioSummary Component
 *
 * Renders an AI-generated summary of the user's portfolio.
 * - Reads `portfolioData` from PortfolioContext
 * - Tries multiple shapes: top-level ai_summary, ai_analysis.summary,
 *   results.ai_analysis.summary, results.education.summary, etc.
 * - Safely normalizes any value to a markdown string for react-markdown
 * - Adds helpful debug logs in non‑production environments
 */

import React, { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { usePortfolio } from "../../context/PortfolioContext";

// ---------- helpers ----------
const normalizeToMarkdown = (value) => {
  if (value == null) return "";

  // If backend wrapped the summary in a known key
  if (typeof value === "object" && !Array.isArray(value)) {
    const possible =
      value.markdown ||
      value.text ||
      value.content ||
      value.summary ||
      value.ai_summary ||
      value.aiSummary;
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

// Pull a summary from any known location/shape
const extractAiSummary = (pd) => {
  if (!pd) return "";

  // single sources
  if (pd.ai_summary) return pd.ai_summary;
  if (pd.aiAnalysis) return pd.aiAnalysis; // sometimes normalized by context
  if (pd.ai_analysis?.summary) return pd.ai_analysis.summary;

  // common nested shapes
  if (pd.results?.ai_summary) return pd.results.ai_summary;
  if (pd.results?.ai_analysis?.summary) return pd.results.ai_analysis.summary;
  if (pd.results?.education?.summary) return pd.results.education.summary;

  // Some APIs put it under results.ai_analysis.content / .text / .markdown
  const ra = pd.results?.ai_analysis;
  if (ra?.content) return ra.content;
  if (ra?.text) return ra.text;
  if (ra?.markdown) return ra.markdown;

  // final fallback to anything string-like on ai_analysis
  if (typeof ra === "string") return ra;

  return "";
};

export default function AIPortfolioSummary() {
  const { portfolioData, isLoading, error } = usePortfolio();

  if (isLoading) return <p>Loading portfolio summary...</p>;
  if (error) return <p>Error loading portfolio data: {String(error)}</p>;
  if (!portfolioData) {
    return <p>No portfolio selected yet.</p>;
  }

  // Gather a raw summary from any known shape and normalize to markdown
  const summaryText = useMemo(() => {
    const raw = extractAiSummary(portfolioData);
    return normalizeToMarkdown(raw);
  }, [portfolioData]);

  // helpful debug in non‑production
  if (process.env.NODE_ENV !== "production") {
    // eslint-disable-next-line no-console
    console.log("[AIPortfolioSummary] keys:", Object.keys(portfolioData || {}));
    // eslint-disable-next-line no-console
    console.log(
      "[AIPortfolioSummary] ai_summary(top-level):",
      portfolioData?.ai_summary ? "present" : "missing"
    );
    // eslint-disable-next-line no-console
    console.log(
      "[AIPortfolioSummary] ai_analysis(summary @ root):",
      portfolioData?.ai_analysis?.summary ? "present" : "missing"
    );
    // eslint-disable-next-line no-console
    console.log(
      "[AIPortfolioSummary] results.ai_analysis.summary:",
      portfolioData?.results?.ai_analysis?.summary ? "present" : "missing"
    );
    // eslint-disable-next-line no-console
    console.log(
      "[AIPortfolioSummary] results.education.summary:",
      portfolioData?.results?.education?.summary ? "present" : "missing"
    );
    // eslint-disable-next-line no-console
    console.log(
      "[AIPortfolioSummary] summaryText length:",
      summaryText?.length || 0
    );
  }

  if (!summaryText || !summaryText.trim()) {
    return (
      <div className="mb-8 mx-auto max-w-6xl">
        <h2 className="text-3xl font-semibold mb-8 text-gray-900 text-center">
          AI Portfolio Analysis
        </h2>
        <div className="bg-yellow-50 p-6 rounded-lg border border-yellow-200 text-yellow-800">
          <p className="font-semibold mb-1">
            No AI analysis summary found for this simulation.
          </p>
          <p className="text-sm">
            Try opening a newer run or regenerate analysis. (If you&apos;re in
            development, check the console for shape hints.)
          </p>
        </div>
      </div>
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
