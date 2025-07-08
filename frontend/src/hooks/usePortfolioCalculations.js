// hooks/usePortfolioCalculations.js
import { useMemo } from "react";
import { formatCurrency } from "../utils/portfolioCalculations";

export default function usePortfolioCalculations(portfolioData) {
  const calculations = useMemo(() => {
    const lumpSum = portfolioData?.lump_sum || 0;
    const monthlyContribution = portfolioData?.monthly || 0;
    const timeframeYears = portfolioData?.timeframe || 0;
    const results = portfolioData?.results || {};
    const finalBalance =
      results?.end_value ?? portfolioData?.final_balance ?? 0;

    // FIXED: Calculate total invested to match the graph logic
    // Graph starts with lump sum, then adds monthly contributions from month 1 onwards
    // So total invested = lump sum + (monthly * 12 * years)
    // But if we're looking at current value vs timeline, we need to check actual timeline data

    let totalInvested;

    // Try to get the actual invested amount from timeline data (most accurate)
    const contributionsTimeline = results?.timeline?.contributions;
    if (contributionsTimeline && contributionsTimeline.length > 0) {
      // Use the final contribution value from timeline
      const sortedContributions = [...contributionsTimeline].sort(
        (a, b) => new Date(a.date) - new Date(b.date)
      );
      totalInvested =
        sortedContributions[sortedContributions.length - 1]?.value || 0;
    } else {
      // Fallback: Calculate based on parameters
      // This should match the graph's calculation method
      totalInvested = lumpSum + monthlyContribution * 12 * timeframeYears;
    }

    // Calculate gains/losses
    const totalGainLoss = finalBalance - totalInvested;

    // Calculate return percentages
    const totalReturnPercent =
      totalInvested > 0 ? (totalGainLoss / totalInvested) * 100 : 0;

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
      totalReturnPercent,
      annualizedReturnPercent,
    };
  }, [portfolioData]);

  // Calculate target achievement status
  const targetStatus = useMemo(() => {
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
      const remaining = targetValue - calculations.finalBalance;
      const progress = (calculations.finalBalance / targetValue) * 100;

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
  }, [portfolioData, calculations.finalBalance]);

  return { calculations, targetStatus };
}
