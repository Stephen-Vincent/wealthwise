import { useContext } from "react";
import PortfolioContext from "../../context/PortfolioContext";

export default function SummaryCards() {
  const { portfolioData } = useContext(PortfolioContext);

  // Round today's date down to the 1st of the current month
  const now = new Date();

  const timeline = portfolioData?.timeline || [];
  const sortedTimeline = [...timeline].sort(
    (a, b) => new Date(a.date) - new Date(b.date)
  );
  const nextMonthStart = new Date(now.getFullYear(), now.getMonth() + 1, 1);
  const filteredTimeline = sortedTimeline.filter(
    (entry) => new Date(entry.date) < nextMonthStart
  );
  const latestEntry = filteredTimeline.at(-1);

  const finalBalance = latestEntry?.value || portfolioData?.final_balance;

  // Calculate total invested based on actual contributions in the timeline
  const initialInvestment = portfolioData?.initial_investment || 0;
  const monthlyContribution = portfolioData?.monthly_contribution || 0;
  const totalInvested = filteredTimeline.reduce((sum, entry) => {
    return sum + (entry.is_contribution ? monthlyContribution : 0);
  }, initialInvestment);

  const returnPercent =
    totalInvested > 0 && finalBalance
      ? (((finalBalance - totalInvested) / totalInvested) * 100).toFixed(2)
      : "N/A";

  // Prepare the target card

  let targetReachedEntry = null;
  if (
    portfolioData?.target_value &&
    Array.isArray(filteredTimeline) &&
    filteredTimeline.length > 0
  ) {
    targetReachedEntry = filteredTimeline.find(
      (entry) => entry.value >= portfolioData.target_value
    );
  }

  const targetCard = {
    label: "Target Investment",
    value: `£${portfolioData?.target_value?.toLocaleString() || "Not set"}`,
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
                ? `${portfolioData.risk || "Unknown"} (Score: ${
                    portfolioData.risk_score
                  })`
                : portfolioData?.risk || "N/A",
          },
          {
            label: "Starting Balance",
            value: `£${
              portfolioData?.total_start !== undefined
                ? portfolioData.total_start.toLocaleString()
                : "0"
            }`,
          },
          {
            label: "Current Balance",
            value: `£${finalBalance?.toLocaleString() || 0}`,
          },
          { label: "Return %", value: `${returnPercent}%` },
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
                    Reached £{card.status.reachedValue?.toLocaleString()} on{" "}
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
