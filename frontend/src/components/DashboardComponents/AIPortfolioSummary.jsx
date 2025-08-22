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

      {/* Analysis Insights Grid */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {/* Performance Insights */}
        {aiAnalysis.performance_insights?.performance_analysis && (
          <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
            <h3 className="text-xl font-semibold text-blue-900 mb-4">
              Performance Analysis
            </h3>
            <div className="prose prose-blue max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {formatMarkdown(
                  aiAnalysis.performance_insights.performance_analysis
                )}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {/* Risk Insights */}
        {aiAnalysis.risk_insights?.risk_analysis && (
          <div className="bg-amber-50 p-6 rounded-lg border border-amber-200">
            <h3 className="text-xl font-semibold text-amber-900 mb-4">
              Risk Analysis
            </h3>
            <div className="prose prose-amber max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {formatMarkdown(aiAnalysis.risk_insights.risk_analysis)}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      {/* News Sentiment Summary */}
      {aiAnalysis.news_sentiment &&
        Object.keys(aiAnalysis.news_sentiment).length > 0 && (
          <div className="bg-green-50 p-6 rounded-lg border border-green-200 mb-8">
            <h3 className="text-xl font-semibold text-green-900 mb-4">
              Market Sentiment Analysis
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {Object.entries(aiAnalysis.news_sentiment).map(
                ([symbol, sentiment]) => (
                  <div key={symbol} className="text-center">
                    <div className="font-bold text-green-800">{symbol}</div>
                    <div className="text-sm text-green-700">
                      {sentiment?.sentiment_category || "Neutral"}
                    </div>
                    <div className="text-xs text-green-600">
                      {sentiment?.total_articles || 0} articles
                    </div>
                  </div>
                )
              )}
            </div>
          </div>
        )}

      {/* Metadata Footer */}
      {aiAnalysis.metadata && (
        <div className="bg-gray-100 p-4 rounded-lg text-sm text-gray-600">
          <div className="flex flex-wrap justify-between items-center">
            <span>
              Analysis Version: {aiAnalysis.metadata.analysis_version}
            </span>
            <span>Stocks Analyzed: {aiAnalysis.metadata.stocks_analyzed}</span>
            <span>
              Features:{" "}
              {aiAnalysis.metadata.includes_news_sentiment ? "News " : ""}
              {aiAnalysis.metadata.includes_market_context
                ? "Market Context"
                : ""}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
