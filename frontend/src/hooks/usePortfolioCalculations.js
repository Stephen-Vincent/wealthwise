// hooks/usePortfolioCalculations.js
import { useMemo } from "react";
import { formatCurrency } from "../utils/portfolioCalculations";

// Normalize any timeline-ish input into an array of { date, value, ... }
// Supports shapes:
//  - Array of entries: [{date, value, ...}]
//  - Legacy object with keys: { "2024-01": 1234, ... }
//  - Legacy nested: { portfolio: [...], contributions: [...] }
function normalizeTimeline(input) {
  if (Array.isArray(input)) return input;

  if (input && typeof input === "object") {
    // Nested legacy shape
    if (Array.isArray(input.portfolio)) return input.portfolio;

    // Mapping object -> array
    const keys = Object.keys(input);
    if (keys.length > 0 && !("length" in input)) {
      return keys.map((date) => {
        const raw = input[date];
        if (raw && typeof raw === "object") {
          // already an entry object, ensure date exists
          return { date, ...raw };
        }
        const value = typeof raw === "number" ? raw : Number(raw) || 0;
        return { date, value };
      });
    }
  }

  return [];
}

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

    // ---------- Normalize possible timeline shapes to arrays ----------
    const topLevelTimeline = normalizeTimeline(portfolioData?.timeline);
    const legacyPortfolioTimeline = normalizeTimeline(results?.timeline);
    const legacyContribTimeline = normalizeTimeline(
      results?.timeline?.contributions
    );
    const pmTimeline = normalizeTimeline(
      results?.portfolio_metrics?.timeline_data
    );

    // ---------- Final Balance (prefer backend-calculated metrics) ----------
    let finalBalance = 0;

    if (portfolioData?.final_balance) {
      finalBalance = portfolioData.final_balance;
      console.log("✅ Using backend final_balance:", finalBalance);
    } else if (portfolioData?.performance_metrics?.ending_value) {
      finalBalance = portfolioData.performance_metrics.ending_value;
      console.log("✅ Using performance_metrics.ending_value:", finalBalance);
    } else if (results?.portfolio_metrics?.ending_value) {
      finalBalance = results.portfolio_metrics.ending_value;
      console.log(
        "✅ Using results.portfolio_metrics.ending_value:",
        finalBalance
      );
    } else {
      // fallback to any available timeline arrays in preference order
      const candidateTimelines = [
        pmTimeline,
        legacyPortfolioTimeline,
        topLevelTimeline,
      ].filter((arr) => Array.isArray(arr) && arr.length > 0);

      if (candidateTimelines.length > 0) {
        const tl = candidateTimelines[0];
        const lastEntry = tl[tl.length - 1];
        finalBalance = lastEntry?.value ?? lastEntry?.ending_value ?? 0;
        console.log("✅ Using timeline fallback:", finalBalance);
      } else if (
        results?.timeline?.portfolio &&
        Array.isArray(results.timeline.portfolio) &&
        results.timeline.portfolio.length > 0
      ) {
        const lastPortfolioEntry =
          results.timeline.portfolio[results.timeline.portfolio.length - 1];
        finalBalance = lastPortfolioEntry?.value || 0;
        console.log("✅ Using nested timeline fallback:", finalBalance);
      } else {
        // last resort fallbacks
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

    // ---------- Total Invested ----------
    let totalInvested = 0;

    if (portfolioData?.performance_metrics?.total_contributed) {
      totalInvested = portfolioData.performance_metrics.total_contributed;
      console.log("✅ Using backend total_contributed:", totalInvested);
    } else if (results?.portfolio_metrics?.total_contributed) {
      totalInvested = results.portfolio_metrics.total_contributed;
      console.log("✅ Using results total_contributed:", totalInvested);
    } else {
      // Try to infer from contributions timeline (prefer explicit contribs)
      const candidateContribs = [
        legacyContribTimeline, // results.timeline.contributions
        pmTimeline, // if timeline_data includes total_contributed values
        topLevelTimeline,
      ].find((arr) => Array.isArray(arr) && arr.length > 0);

      if (candidateContribs) {
        const sortedContributions = [...candidateContribs].sort(
          (a, b) => new Date(a.date) - new Date(b.date)
        );
        const lastContribution =
          sortedContributions[sortedContributions.length - 1];
        totalInvested =
          lastContribution?.total_contributed ?? lastContribution?.value ?? 0;
        console.log("✅ Total invested from timeline:", totalInvested);
      } else {
        totalInvested = lumpSum + monthlyContribution * 12 * timeframeYears;
        console.log("⚠️ Total invested calculated (fallback):", totalInvested);
      }
    }

    // ---------- Returns ----------
    const totalGainLoss = finalBalance - totalInvested;

    let totalReturnPercent = 0;
    let annualizedReturnPercent = 0;

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

  // ---------- Target achievement status ----------
  const targetStatus = useMemo(() => {
    // Use backend flag if present
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

    // Fallback: derive from any available timeline array
    const timeline =
      normalizeTimeline(
        portfolioData?.results?.portfolio_metrics?.timeline_data
      ) ||
      normalizeTimeline(portfolioData?.results?.timeline) ||
      normalizeTimeline(portfolioData?.timeline);

    const targetValue = portfolioData?.target_value;

    console.log("🎯 Target status calculation:", {
      targetValue,
      timelineLength: Array.isArray(timeline) ? timeline.length : 0,
      finalBalance: calculations.finalBalance,
      target_achieved: portfolioData?.target_achieved,
    });

    if (!targetValue || !Array.isArray(timeline) || timeline.length === 0)
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
