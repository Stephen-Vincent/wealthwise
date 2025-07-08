// utils/portfolioCalculations.js
// Centralized portfolio calculation utilities

/**
 * Calculate all portfolio return metrics consistently
 * @param {Object} portfolioData - The portfolio data from context
 * @param {Object} results - The results object from portfolio data
 * @returns {Object} Calculated return metrics
 */
export const calculateReturns = (portfolioData, results) => {
  const lumpSum = portfolioData?.lump_sum || 0;
  const monthlyContribution = portfolioData?.monthly || 0;
  const timeframeYears = portfolioData?.timeframe || 0;
  const finalBalance = results?.end_value ?? portfolioData?.final_balance ?? 0;

  // Total amount invested (should be same as currentContributions)
  const totalInvested = lumpSum + monthlyContribution * 12 * timeframeYears;

  // Total gain or loss
  const totalGainLoss = finalBalance - totalInvested;

  // Total return percentage (over entire period)
  const totalReturnPercent =
    totalInvested > 0 ? (totalGainLoss / totalInvested) * 100 : 0;

  // Annualized return percentage
  const annualizedReturnPercent =
    totalInvested > 0 && timeframeYears > 0
      ? (Math.pow(finalBalance / totalInvested, 1 / timeframeYears) - 1) * 100
      : 0;

  return {
    lumpSum,
    monthlyContribution,
    timeframeYears,
    totalInvested,
    finalBalance,
    totalGainLoss,
    totalReturnPercent, // Use this for main return display
    annualizedReturnPercent, // Use this for annualized return
  };
};

/**
 * Format currency for display
 * @param {number} amount - Amount to format
 * @returns {string} Formatted currency string
 */
export const formatCurrency = (amount) =>
  amount != null
    ? `£${parseFloat(amount).toLocaleString("en-GB", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`
    : "£0.00";

/**
 * Format percentage for display
 * @param {number} value - Percentage value to format
 * @returns {string} Formatted percentage string
 */
export const formatPercentage = (value) => {
  if (value === "N/A" || value == null) return "N/A";
  const num = parseFloat(value);
  return `${num >= 0 ? "+" : ""}${num.toFixed(2)}%`;
};

/**
 * Calculate target achievement status
 * @param {Object} portfolioData - The portfolio data from context
 * @param {number} finalBalance - The final portfolio balance
 * @returns {Object|null} Target status object or null if no target
 */
export const calculateTargetStatus = (portfolioData, finalBalance) => {
  const timeline = portfolioData?.results?.timeline?.portfolio || [];
  const targetValue = portfolioData?.target_value;

  if (!targetValue || !timeline.length) return null;

  const sortedTimeline = [...timeline].sort(
    (a, b) => new Date(a.date) - new Date(b.date)
  );
  const targetReachedEntry = sortedTimeline.find(
    (entry) => entry.value >= targetValue
  );

  if (targetReachedEntry) {
    return {
      achieved: true,
      date: targetReachedEntry.date,
      value: targetReachedEntry.value,
      text: "Target Achieved! 🎉",
      color: "text-green-600",
      bgColor: "bg-green-50",
      borderColor: "border-green-200",
    };
  } else {
    const remaining = targetValue - finalBalance;
    const progress = (finalBalance / targetValue) * 100;

    return {
      achieved: false,
      remaining,
      progress,
      text: `${progress.toFixed(1)}% Complete`,
      color: progress > 75 ? "text-yellow-600" : "text-blue-600",
      bgColor: progress > 75 ? "bg-yellow-50" : "bg-blue-50",
      borderColor: progress > 75 ? "border-yellow-200" : "border-blue-200",
    };
  }
};
