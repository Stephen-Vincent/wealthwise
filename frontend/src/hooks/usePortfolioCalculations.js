// hooks/usePortfolioCalculations.js
import { useMemo } from "react";
import { formatCurrency } from "../utils/portfolioCalculations";

export default function usePortfolioCalculations(portfolioData) {
  const calculations = useMemo(() => {
    console.log("🔍 DEBUG: usePortfolioCalculations input:", portfolioData);

    if (!portfolioData) {
      console.log("❌ No portfolioData provided");
      return {
        lumpSum: 0,
        monthlyContribution: 0,
        timeframeYears: 0,
        totalInvested: 0,
        finalBalance: 0,
        totalGainLoss: 0,
        totalReturnPercent: 0,
        annualizedReturnPercent: 0,
      };
    }

    const lumpSum = portfolioData?.lump_sum || 0;
    const monthlyContribution = portfolioData?.monthly || 0;
    const timeframeYears = portfolioData?.timeframe || 0;
    const results = portfolioData?.results || {};

    console.log("📊 Basic parameters:", {
      lumpSum,
      monthlyContribution,
      timeframeYears,
      "results keys": Object.keys(results),
      "portfolioData keys": Object.keys(portfolioData),
    });

    // Look for finalBalance in multiple possible locations
    let finalBalance = 0;

    // Check various possible locations for the final balance
    const possibleBalances = [
      results?.end_value,
      portfolioData?.final_balance,
      portfolioData?.current_value,
      portfolioData?.end_value,
      results?.final_value,
      portfolioData?.totalValue,
      results?.total_value,
    ];

    // Try to get from timeline if available
    const timeline = results?.timeline;
    if (timeline) {
      console.log("📈 Timeline data available:", {
        "timeline keys": Object.keys(timeline),
        "portfolio timeline": timeline.portfolio
          ? timeline.portfolio.length
          : "none",
        "contributions timeline": timeline.contributions
          ? timeline.contributions.length
          : "none",
      });

      // Try to get final portfolio value from timeline
      if (timeline.portfolio && timeline.portfolio.length > 0) {
        const lastPortfolioEntry =
          timeline.portfolio[timeline.portfolio.length - 1];
        console.log("📊 Last portfolio entry:", lastPortfolioEntry);

        if (
          typeof lastPortfolioEntry === "object" &&
          lastPortfolioEntry.value
        ) {
          finalBalance = lastPortfolioEntry.value;
          console.log(
            "✅ Found finalBalance from portfolio timeline:",
            finalBalance
          );
        } else if (typeof lastPortfolioEntry === "number") {
          finalBalance = lastPortfolioEntry;
          console.log(
            "✅ Found finalBalance from portfolio timeline (number):",
            finalBalance
          );
        }
      }
    }

    // If no balance found from timeline, try the direct properties
    if (finalBalance === 0) {
      for (const balance of possibleBalances) {
        if (balance !== undefined && balance !== null && balance !== 0) {
          finalBalance = balance;
          console.log(
            "✅ Found finalBalance from direct property:",
            finalBalance
          );
          break;
        }
      }
    }

    console.log("💰 All possible balance values:", possibleBalances);
    console.log("🎯 Final selected balance:", finalBalance);

    // Calculate total invested
    let totalInvested;

    // Try to get the actual invested amount from timeline data (most accurate)
    const contributionsTimeline = results?.timeline?.contributions;
    if (contributionsTimeline && contributionsTimeline.length > 0) {
      console.log(
        "📈 Contributions timeline available:",
        contributionsTimeline.length,
        "entries"
      );

      // Use the final contribution value from timeline
      const sortedContributions = [...contributionsTimeline].sort(
        (a, b) => new Date(a.date) - new Date(b.date)
      );
      const lastContribution =
        sortedContributions[sortedContributions.length - 1];
      totalInvested = lastContribution?.value || 0;

      console.log(
        "✅ Total invested from timeline:",
        totalInvested,
        "Last entry:",
        lastContribution
      );
    } else {
      // Fallback: Calculate based on parameters
      totalInvested = lumpSum + monthlyContribution * 12 * timeframeYears;
      console.log("⚠️ Total invested calculated (fallback):", totalInvested);
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

    const calculatedResults = {
      lumpSum,
      monthlyContribution,
      timeframeYears,
      totalInvested,
      finalBalance,
      totalGainLoss,
      totalReturnPercent,
      annualizedReturnPercent,
    };

    console.log("🧮 Final calculations:", calculatedResults);

    return calculatedResults;
  }, [portfolioData]);

  // Calculate target achievement status
  const targetStatus = useMemo(() => {
    const timeline = portfolioData?.results?.timeline?.portfolio || [];
    const targetValue = portfolioData?.target_value;

    console.log("🎯 Target status calculation:", {
      targetValue,
      timelineLength: timeline.length,
      finalBalance: calculations.finalBalance,
    });

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
