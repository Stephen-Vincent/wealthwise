import { useContext } from "react";
import PortfolioContext from "../../context/PortfolioContext";

export default function SummaryCards() {
  const formatCurrency = (amount) =>
    amount != null ? `£${parseFloat(amount).toFixed(2)}` : "£0";

  // Access portfolio data from context
  const { portfolioData } = useContext(PortfolioContext);
  const results = portfolioData?.results || {};

  // Round today's date down to the 1st of the current month
  const now = new Date();

  // Derive timeline data and find the latest entry before next month
  const timeline = portfolioData?.results?.timeline?.portfolio || [];
  const sortedTimeline = [...timeline].sort(
    (a, b) => new Date(a.date) - new Date(b.date)
  );
  const nextMonthStart = new Date(now.getFullYear(), now.getMonth() + 1, 1);
  const filteredTimeline = sortedTimeline.filter(
    (entry) => new Date(entry.date) < nextMonthStart
  );

  // Calculate final balance from results or fallback to portfolioData
  const finalBalance = results?.end_value ?? portfolioData?.final_balance ?? 0;

  // Calculate total invested based on lump sum and monthly contributions over timeframe
  const lumpSum = portfolioData?.lump_sum || 0;
  const monthlyContribution = portfolioData?.monthly || 0;
  const timeframeYears = portfolioData?.timeframe || 0;
  const totalInvested = lumpSum + monthlyContribution * 12 * timeframeYears;

  // Calculate return percentage if total invested and final balance are available
  const returnPercent =
    totalInvested > 0 && finalBalance != null
      ? (
          ((parseFloat(finalBalance) - parseFloat(totalInvested)) /
            parseFloat(totalInvested)) *
          100
        ).toFixed(2)
      : "N/A";

  // Detect if the investment target has been met by checking filtered timeline entries
  let targetReachedEntry = null;
  if (
    portfolioData?.target_value &&
    Array.isArray(timeline) &&
    timeline.length > 0
  ) {
    targetReachedEntry = filteredTimeline.find(
      (entry) => entry.value >= portfolioData.target_value
    );
  }

  // Prepare target card information including status based on target achievement
  const targetCard = {
    label: "Target Investment",
    value:
      portfolioData?.target_value != null
        ? formatCurrency(portfolioData.target_value)
        : "Not set",
    status:
      portfolioData?.target_value && finalBalance
        ? targetReachedEntry
          ? {
              text: "Target Achieved",
              color: "text-green-600",
              date: targetReachedEntry.date,
              reachedValue: targetReachedEntry.value,
            }
          : { text: "Target Not Met", color: "text-red-600" }
        : null,
  };

  // Render summary cards displaying portfolio overview and status
  return (
    <div className="mb-8">
      <h3 className="text-xl font-bold mb-4">My Summary</h3>
      <p className="mb-4 text-gray-700">
        <strong>My Goal:</strong> {portfolioData?.goal || "Not set"}
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        {[
          targetCard,
          {
            label: "Risk Profile",
            value:
              portfolioData?.risk_score != null
                ? `${portfolioData.risk_label || "Unknown"} (Score: ${
                    portfolioData.risk_score
                  })`
                : portfolioData?.risk_label || "N/A",
          },
          {
            label: "Starting Balance",
            value:
              portfolioData?.lump_sum != null
                ? formatCurrency(portfolioData.lump_sum)
                : "£0",
          },
          {
            label: "Current Balance",
            value: finalBalance != null ? formatCurrency(finalBalance) : "£0",
          },
          {
            label: "Total Invested",
            value: formatCurrency(totalInvested),
          },
          {
            label: "Return %",
            value: `${returnPercent}%`,
            status: null,
          },
        ].map((card, index) => (
          <div
            key={index}
            className="p-4 bg-white shadow rounded-lg text-center border border-gray-200"
          >
            <div className="text-sm text-gray-500">{card.label}</div>
            <div className="text-xl font-bold text-gray-900">{card.value}</div>
            {card.status && (
              <div className={`mt-1 text-sm font-medium ${card.status.color}`}>
                {card.status.text}
                {card.status.text === "Target Achieved" && card.status.date && (
                  <div className="text-xs text-gray-500 mt-1">
                    Reached {formatCurrency(card.status.reachedValue)} on{" "}
                    {new Date(card.status.date).toLocaleDateString()}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
