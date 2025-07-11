/**
 * SummaryCards Component
 *
 * This component displays a summary of the user's investment portfolio,
 * including key metrics such as investment goal, target amount, risk profile,
 * total invested, current value, and total return. It uses context to access
 * portfolio data and custom hooks for portfolio calculations.
 *
 * Features:
 * - Dynamically styled cards based on portfolio data and risk profile.
 * - Visual indicators for target achievement and progress.
 * - Enhanced risk profile description with color-coded styling.
 * - Responsive grid layout for different screen sizes.
 */

import { useContext } from "react";
import PortfolioContext from "../../context/PortfolioContext";
import usePortfolioCalculations from "../../hooks/usePortfolioCalculations";
import {
  formatCurrency,
  formatPercentage,
} from "../../utils/portfolioCalculations";

// Icons for different metrics displayed in the summary cards
const ICONS = {
  target: "🎯",
  risk: "📊",
  balance: "💰",
  invested: "💳",
  returns: "📈",
  timeframe: "⏰",
  goal: "🎯",
};

// Risk level configurations defining colors, border styles, and descriptions
// Used to visually distinguish risk profiles and provide user-friendly info
const RISK_CONFIGS = {
  "Ultra Conservative": {
    color: "text-blue-600",
    bgColor: "bg-blue-50",
    borderColor: "border-blue-200",
    description: "Very low risk, stable returns",
  },
  Conservative: {
    color: "text-green-600",
    bgColor: "bg-green-50",
    borderColor: "border-green-200",
    description: "Low risk, steady growth",
  },
  "Moderate Conservative": {
    color: "text-teal-600",
    bgColor: "bg-teal-50",
    borderColor: "border-teal-200",
    description: "Balanced approach, modest risk",
  },
  Moderate: {
    color: "text-yellow-600",
    bgColor: "bg-yellow-50",
    borderColor: "border-yellow-200",
    description: "Balanced risk and growth",
  },
  "Moderate Aggressive": {
    color: "text-orange-600",
    bgColor: "bg-orange-50",
    borderColor: "border-orange-200",
    description: "Growth focused, higher risk",
  },
  Aggressive: {
    color: "text-red-600",
    bgColor: "bg-red-50",
    borderColor: "border-red-200",
    description: "High risk, high growth potential",
  },
  "Ultra Aggressive": {
    color: "text-purple-600",
    bgColor: "bg-purple-50",
    borderColor: "border-purple-200",
    description: "Very high risk, maximum growth",
  },

  // ADD THESE MISSING MAPPINGS:
  High: {
    color: "text-red-600",
    bgColor: "bg-red-50",
    borderColor: "border-red-200",
    description: "High risk, high growth potential",
  },
  Medium: {
    color: "text-yellow-600",
    bgColor: "bg-yellow-50",
    borderColor: "border-yellow-200",
    description: "Balanced risk and growth",
  },
  Low: {
    color: "text-green-600",
    bgColor: "bg-green-50",
    borderColor: "border-green-200",
    description: "Low risk, steady growth",
  },
  "Very High": {
    color: "text-purple-600",
    bgColor: "bg-purple-50",
    borderColor: "border-purple-200",
    description: "Very high risk, maximum growth",
  },
  "Very Low": {
    color: "text-blue-600",
    bgColor: "bg-blue-50",
    borderColor: "border-blue-200",
    description: "Very low risk, stable returns",
  },
};

export default function SummaryCards() {
  // Access portfolio data from context
  const { portfolioData } = useContext(PortfolioContext);

  // Use custom hook for portfolio calculations and target status
  const { calculations, targetStatus } =
    usePortfolioCalculations(portfolioData);

  // Determine risk configuration based on portfolio risk label,
  // fallback to default gray styling if no matching label found
  const riskConfig = RISK_CONFIGS[portfolioData?.risk_label] || {
    color: "text-gray-600",
    bgColor: "bg-gray-50",
    borderColor: "border-gray-200",
    description: "Risk assessment unavailable",
  };

  // Define summary cards data with styling, icons, labels, values, and subtitles
  // Includes conditional logic for displaying investment details and statuses
  const cards = [
    {
      id: "goal",
      icon: ICONS.goal,
      label: "Investment Goal",
      value: portfolioData?.goal
        ? portfolioData.goal.charAt(0).toUpperCase() +
          portfolioData.goal.slice(1)
        : "Not set",
      subtitle: portfolioData?.timeframe
        ? `${portfolioData.timeframe} year${
            portfolioData.timeframe !== 1 ? "s" : ""
          } timeline`
        : null,
      color: "text-indigo-600",
      bgColor: "bg-indigo-50",
      borderColor: "border-indigo-200",
    },
    {
      id: "target",
      icon: ICONS.target,
      label: "Target Amount",
      value: portfolioData?.target_value
        ? formatCurrency(portfolioData.target_value)
        : "Not set",
      subtitle: targetStatus?.text,
      color: targetStatus?.color || "text-gray-600",
      bgColor: targetStatus?.bgColor || "bg-gray-50",
      borderColor: targetStatus?.borderColor || "border-gray-200",
      status: targetStatus,
    },
    {
      id: "risk",
      icon: ICONS.risk,
      label: "Risk Profile",
      value: portfolioData?.risk_label || "Not assessed",
      subtitle: portfolioData?.risk_score
        ? `Score: ${portfolioData.risk_score}/100`
        : riskConfig.description,
      color: riskConfig.color,
      bgColor: riskConfig.bgColor,
      borderColor: riskConfig.borderColor,
    },
    {
      id: "invested",
      icon: ICONS.invested,
      label: "Total Invested",
      value: formatCurrency(calculations.totalInvested),
      subtitle:
        calculations.monthlyContribution > 0
          ? // Show lump sum + monthly contribution if monthlyContribution > 0
            `£${calculations.lumpSum.toLocaleString()} lump sum + £${calculations.monthlyContribution.toLocaleString()}/month`
          : calculations.lumpSum > 0
          ? // Show one-time investment if only lump sum > 0
            "One-time investment"
          : // Otherwise, no investment recorded
            "No investment recorded",
      color: "text-blue-600",
      bgColor: "bg-blue-50",
      borderColor: "border-blue-200",
    },
    {
      id: "balance",
      icon: ICONS.balance,
      label: "Current Value",
      value: formatCurrency(calculations.finalBalance),
      subtitle:
        calculations.totalGainLoss !== 0
          ? calculations.totalGainLoss > 0
            ? // Positive gain with plus sign
              `+${formatCurrency(Math.abs(calculations.totalGainLoss))} gain`
            : // Negative loss with minus sign
              `-${formatCurrency(Math.abs(calculations.totalGainLoss))} loss`
          : // No change if gain/loss is zero
            "No change",
      // Color coding based on gain/loss positive, negative, or neutral
      color:
        calculations.totalGainLoss > 0
          ? "text-green-600"
          : calculations.totalGainLoss < 0
          ? "text-red-600"
          : "text-gray-600",
      bgColor:
        calculations.totalGainLoss > 0
          ? "bg-green-50"
          : calculations.totalGainLoss < 0
          ? "bg-red-50"
          : "bg-gray-50",
      borderColor:
        calculations.totalGainLoss > 0
          ? "border-green-200"
          : calculations.totalGainLoss < 0
          ? "border-red-200"
          : "border-gray-200",
    },
    {
      id: "returns",
      icon: ICONS.returns,
      label: "Total Return",
      value: formatPercentage(calculations.totalReturnPercent),
      subtitle:
        calculations.timeframeYears > 0
          ? // Show annualized return if timeframe is positive
            `${formatPercentage(
              calculations.annualizedReturnPercent
            )} annualized`
          : // Otherwise, indicate performance data unavailable
            "Performance data unavailable",
      // Color coding based on return positive, negative, or neutral
      color:
        calculations.totalReturnPercent > 0
          ? "text-green-600"
          : calculations.totalReturnPercent < 0
          ? "text-red-600"
          : "text-gray-600",
      bgColor:
        calculations.totalReturnPercent > 0
          ? "bg-green-50"
          : calculations.totalReturnPercent < 0
          ? "bg-red-50"
          : "bg-gray-50",
      borderColor:
        calculations.totalReturnPercent > 0
          ? "border-green-200"
          : calculations.totalReturnPercent < 0
          ? "border-red-200"
          : "border-gray-200",
    },
  ];

  return (
    <div className="mb-8">
      {/* Header with title and optional portfolio name */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-2xl font-bold text-gray-800">Portfolio Summary</h3>
        {portfolioData?.name && (
          <div className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
            {portfolioData.name}
          </div>
        )}
      </div>

      {/* Grid layout for summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {cards.map((card) => (
          <div
            key={card.id}
            className={`relative p-6 bg-white shadow-lg rounded-xl border-l-4 ${card.borderColor} hover:shadow-xl transition-all duration-200 group`}
          >
            {/* Icon and Label */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <span className="text-2xl">{card.icon}</span>
                <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                  {card.label}
                </h4>
              </div>
              {/* Show checkmark if target achieved */}
              {card.id === "target" && card.status?.achieved && (
                <div className="text-green-500 animate-pulse">✅</div>
              )}
            </div>

            {/* Main Value */}
            <div
              className={`text-2xl font-bold ${card.color} mb-2 group-hover:scale-105 transition-transform duration-200`}
            >
              {card.value}
            </div>

            {/* Subtitle/Additional Info */}
            {card.subtitle && (
              <div className="text-sm text-gray-600">{card.subtitle}</div>
            )}

            {/* Special Status for Target Achievement */}
            {card.id === "target" && card.status && card.status.achieved && (
              <div className="mt-3 text-xs text-gray-500 bg-green-50 p-2 rounded">
                🎉 Reached {formatCurrency(card.status.value)} on{" "}
                {new Date(card.status.date).toLocaleDateString("en-GB")}
              </div>
            )}

            {/* Progress bar for target not yet achieved */}
            {card.id === "target" &&
              card.status &&
              !card.status.achieved &&
              card.status.progress && (
                <div className="mt-3">
                  <div className="flex justify-between text-xs text-gray-600 mb-1">
                    <span>Progress</span>
                    {/* Progress percentage with one decimal place */}
                    <span>{card.status.progress.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full transition-all duration-500"
                      style={{
                        // Limit progress bar width to max 100%
                        width: `${Math.min(card.status.progress, 100)}%`,
                      }}
                    />
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {formatCurrency(card.status.remaining)} remaining
                  </div>
                </div>
              )}

            {/* Subtle background decoration */}
            <div
              className={`absolute top-0 right-0 w-20 h-20 ${card.bgColor} rounded-full opacity-20 -mr-10 -mt-10`}
            />
          </div>
        ))}
      </div>

      {/* Enhanced Risk Profile Description Section */}
      {portfolioData?.risk_label && (
        <div
          className={`mt-6 p-4 rounded-lg border ${riskConfig.borderColor} ${riskConfig.bgColor}`}
        >
          <div className="flex items-start space-x-3">
            <span className="text-2xl">{ICONS.risk}</span>
            <div>
              <h4 className={`font-semibold ${riskConfig.color}`}>
                {portfolioData.risk_label} Portfolio
              </h4>
              <p className="text-sm text-gray-600 mt-1">
                {riskConfig.description}
              </p>
              {/* Optional detailed risk description */}
              {portfolioData?.risk_description && (
                <p className="text-sm text-gray-600 mt-1">
                  {portfolioData.risk_description}
                </p>
              )}
              {/* Optional allocation guidance with lightbulb icon */}
              {portfolioData?.allocation_guidance && (
                <p className="text-sm text-gray-600 mt-2 font-medium">
                  💡 Strategy: {portfolioData.allocation_guidance}
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
