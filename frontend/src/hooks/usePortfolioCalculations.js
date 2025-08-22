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

    // FIXED: Prioritize backend-calculated values first
    let finalBalance = 0;

    // 1. First, check the new top-level final_balance from backend
    if (portfolioData?.final_balance) {
      finalBalance = portfolioData.final_balance;
      console.log("✅ Using backend final_balance:", finalBalance);
    }
    // 2. Check performance_metrics.ending_value
    else if (portfolioData?.performance_metrics?.ending_value) {
      finalBalance = portfolioData.performance_metrics.ending_value;
      console.log("✅ Using performance_metrics.ending_value:", finalBalance);
    }
    // 3. Check portfolio_metrics.ending_value in results
    else if (results?.portfolio_metrics?.ending_value) {
      finalBalance = results.portfolio_metrics.ending_value;
      console.log(
        "✅ Using results.portfolio_metrics.ending_value:",
        finalBalance
      );
    }
    // 4. Fallback to timeline if no backend values
    else {
      const timeline = portfolioData?.timeline || results?.timeline;

      if (timeline && Array.isArray(timeline) && timeline.length > 0) {
        const lastEntry = timeline[timeline.length - 1];
        finalBalance = lastEntry?.value || 0;
        console.log("✅ Using timeline fallback:", finalBalance);
      }
      // 5. Check nested timeline structure
      else if (timeline?.portfolio && timeline.portfolio.length > 0) {
        const lastPortfolioEntry =
          timeline.portfolio[timeline.portfolio.length - 1];
        finalBalance = lastPortfolioEntry?.value || 0;
        console.log("✅ Using nested timeline fallback:", finalBalance);
      }
      // 6. Last resort - check old field names
      else {
        const possibleBalances = [
          results?.end_value,
          portfolioData?.current_value,
          portfolioData?.end_value,
          results?.final_value,
          portfolioData?.totalValue,
          results?.total_value,
        ];

        for (const balance of possibleBalances) {
          if (balance !== undefined && balance !== null && balance !== 0) {
            finalBalance = balance;
            console.log("✅ Using fallback property:", finalBalance);
            break;
          }
        }
      }
    }

    console.log("🎯 Final selected balance:", finalBalance);

    // Calculate total invested - prioritize backend data
    let totalInvested = 0;

    // 1. Check if backend provides total_contributed
    if (portfolioData?.performance_metrics?.total_contributed) {
      totalInvested = portfolioData.performance_metrics.total_contributed;
      console.log("✅ Using backend total_contributed:", totalInvested);
    } else if (results?.portfolio_metrics?.total_contributed) {
      totalInvested = results.portfolio_metrics.total_contributed;
      console.log("✅ Using results total_contributed:", totalInvested);
    }
    // 2. Try to get from timeline contributions
    else {
      const contributionsTimeline =
        portfolioData?.timeline || results?.timeline?.contributions;

      if (contributionsTimeline && contributionsTimeline.length > 0) {
        const sortedContributions = [...contributionsTimeline].sort(
          (a, b) => new Date(a.date) - new Date(b.date)
        );
        const lastContribution =
          sortedContributions[sortedContributions.length - 1];
        totalInvested =
          lastContribution?.total_contributed || lastContribution?.value || 0;
        console.log("✅ Total invested from timeline:", totalInvested);
      }
      // 3. Fallback: Calculate based on parameters
      else {
        totalInvested = lumpSum + monthlyContribution * 12 * timeframeYears;
        console.log("⚠️ Total invested calculated (fallback):", totalInvested);
      }
    }

    // Calculate gains/losses
    const totalGainLoss = finalBalance - totalInvested;

    // Get return percentages from backend first, calculate as fallback
    let totalReturnPercent = 0;
    let annualizedReturnPercent = 0;

    // Use backend calculations if available
    if (portfolioData?.performance_metrics?.total_return) {
      totalReturnPercent = portfolioData.performance_metrics.total_return;
      console.log("✅ Using backend total_return:", totalReturnPercent);
    } else if (results?.portfolio_metrics?.total_return) {
      totalReturnPercent = results.portfolio_metrics.total_return;
      console.log("✅ Using results total_return:", totalReturnPercent);
    } else if (totalInvested > 0) {
      totalReturnPercent = (totalGainLoss / totalInvested) * 100;
      console.log("⚠️ Calculated total_return fallback:", totalReturnPercent);
    }

    if (portfolioData?.performance_metrics?.annualized_return) {
      annualizedReturnPercent =
        portfolioData.performance_metrics.annualized_return;
      console.log(
        "✅ Using backend annualized_return:",
        annualizedReturnPercent
      );
    } else if (results?.portfolio_metrics?.annualized_return) {
      annualizedReturnPercent = results.portfolio_metrics.annualized_return;
      console.log(
        "✅ Using results annualized_return:",
        annualizedReturnPercent
      );
    } else if (totalInvested > 0 && timeframeYears > 0) {
      annualizedReturnPercent =
        (Math.pow(finalBalance / totalInvested, 1 / timeframeYears) - 1) * 100;
      console.log(
        "⚠️ Calculated annualized_return fallback:",
        annualizedReturnPercent
      );
    }

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
    // Use the backend target_achieved flag first
    if (portfolioData?.target_achieved !== undefined) {
      const targetValue = portfolioData?.target_value || 0;

      if (portfolioData.target_achieved) {
        return {
          achieved: true,
          date: portfolioData?.created_at,
          value: calculations.finalBalance,
          text: "Target Achieved! 🎉",
          color: "text-green-600",
          bgColor: "bg-green-50",
          borderColor: "border-green-200",
        };
      } else {
        const remaining = targetValue - calculations.finalBalance;
        const progress =
          targetValue > 0 ? (calculations.finalBalance / targetValue) * 100 : 0;

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
    }

    // Fallback: Calculate from timeline data
    const timeline =
      portfolioData?.timeline ||
      portfolioData?.results?.timeline?.portfolio ||
      [];
    const targetValue = portfolioData?.target_value;

    console.log("🎯 Target status calculation:", {
      targetValue,
      timelineLength: Array.isArray(timeline) ? timeline.length : 0,
      finalBalance: calculations.finalBalance,
      target_achieved: portfolioData?.target_achieved,
    });

    if (!targetValue || !Array.isArray(timeline) || !timeline.length)
      return null;

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
