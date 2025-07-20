import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { usePortfolio } from "../context/PortfolioContext";

const messages = [
  "🧠 Calculating your risk score...",
  "📊 Putting together your portfolio...",
  "🔍 Gathering market insights...",
  "✅ Finalizing your personalized dashboard...",
];

export default function LoadingScreen() {
  const navigate = useNavigate();
  const { portfolioData } = usePortfolio();
  const userId = localStorage.getItem("userId");
  const rawSimulationId = localStorage.getItem("simulationId");
  const simulationId =
    rawSimulationId && rawSimulationId !== "null" ? rawSimulationId : null;
  const isCreatingPortfolio =
    localStorage.getItem("isCreatingPortfolio") === "true";

  const [index, setIndex] = useState(0);
  const [fade, setFade] = useState(true);
  const [minTimeElapsed, setMinTimeElapsed] = useState(false);
  const [progress, setProgress] = useState(0);

  // Calculate progress based on various factors
  useEffect(() => {
    const updateProgress = () => {
      let currentProgress = 0;

      // Base progress from message index (0-60%)
      currentProgress += (index / (messages.length - 1)) * 60;

      // Add progress if API call is complete (60-80%)
      if (!isCreatingPortfolio && portfolioData) {
        currentProgress = Math.max(currentProgress, 80);
      }

      // Add progress if data is ready (80-95%)
      if (portfolioData && portfolioData.id) {
        currentProgress = Math.max(currentProgress, 95);
      }

      // Complete when ready to navigate (95-100%)
      if (portfolioData && portfolioData.id && minTimeElapsed) {
        currentProgress = 100;
      }

      // If still creating portfolio, cap at 40%
      if (isCreatingPortfolio) {
        currentProgress = Math.min(currentProgress, 40);
      }

      setProgress(Math.round(currentProgress));
    };

    const interval = setInterval(updateProgress, 200);
    updateProgress(); // Run immediately

    return () => clearInterval(interval);
  }, [index, portfolioData, minTimeElapsed, isCreatingPortfolio]);

  // Handle message cycling
  useEffect(() => {
    let timeouts = [];

    messages.forEach((_, i) => {
      const t = setTimeout(() => {
        setFade(false);
        setTimeout(() => {
          setIndex(i);
          setFade(true);
        }, 300);
      }, i * 1000);
      timeouts.push(t);
    });

    return () => timeouts.forEach(clearTimeout);
  }, []);

  // Ensure minimum loading time
  useEffect(() => {
    const minLoadingTime = setTimeout(() => {
      setMinTimeElapsed(true);
    }, 4000);

    return () => clearTimeout(minLoadingTime);
  }, []);

  // Check if portfolio data is ready and navigate when conditions are met
  useEffect(() => {
    const checkDataAndNavigate = () => {
      // If we're still creating the portfolio, don't navigate yet
      if (isCreatingPortfolio) {
        console.log("🔄 Still creating portfolio, waiting...");
        return;
      }

      // Check for the actual data structure
      const isDataReady =
        portfolioData &&
        portfolioData.id &&
        (portfolioData.stocks_picked ||
          portfolioData.breakdown ||
          portfolioData.results);

      console.log("🔍 Data check:", {
        hasPortfolioData: !!portfolioData,
        hasId: !!portfolioData?.id,
        hasStocksPicked: !!portfolioData?.stocks_picked,
        hasBreakdown: !!portfolioData?.breakdown,
        hasResults: !!portfolioData?.results,
        minTimeElapsed,
        userId,
        simulationId,
        isCreatingPortfolio,
        progress,
      });

      // Only navigate if data is ready, minimum time has elapsed, and we have valid IDs
      if (
        isDataReady &&
        minTimeElapsed &&
        userId &&
        simulationId &&
        !isCreatingPortfolio &&
        progress >= 100
      ) {
        console.log("✅ All conditions met, navigating to dashboard:", {
          portfolioData: portfolioData,
          userId: userId,
          simulationId: simulationId,
        });

        // Add a small delay for smooth transition
        setTimeout(() => {
          navigate(`/dashboard/${userId}/${simulationId}`);
        }, 500);
      }
    };

    // Check every second while we're waiting
    const interval = setInterval(checkDataAndNavigate, 1000);

    // Also check immediately
    checkDataAndNavigate();

    return () => clearInterval(interval);
  }, [
    portfolioData,
    minTimeElapsed,
    userId,
    simulationId,
    navigate,
    isCreatingPortfolio,
    progress,
  ]);

  // Fallback timeout
  useEffect(() => {
    const fallbackTimeout = setTimeout(() => {
      if (!portfolioData || !portfolioData.id) {
        console.error(
          "❌ Data not ready after 15 seconds, something may be wrong"
        );
      }
    }, 15000);

    return () => clearTimeout(fallbackTimeout);
  }, [portfolioData]);

  // Progress circle component
  const ProgressCircle = ({ percentage }) => {
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;

    return (
      <div className="relative w-32 h-32 mb-8">
        <svg
          className="transform -rotate-90 w-32 h-32"
          width="128"
          height="128"
        >
          {/* Background circle */}
          <circle
            cx="64"
            cy="64"
            r={radius}
            stroke="currentColor"
            strokeWidth="8"
            fill="transparent"
            className="text-gray-200"
          />
          {/* Progress circle */}
          <circle
            cx="64"
            cy="64"
            r={radius}
            stroke="currentColor"
            strokeWidth="8"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className="text-[#00A8FF] transition-all duration-300 ease-out"
            strokeLinecap="round"
          />
        </svg>
        {/* Percentage text */}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold text-[#00A8FF]">
            {percentage}%
          </span>
        </div>
      </div>
    );
  };

  // Show appropriate message based on state
  const getCurrentMessage = () => {
    if (isCreatingPortfolio) {
      return "🤖 AI is creating your optimized portfolio...";
    }
    return messages[index];
  };

  return (
    <div className="flex flex-col items-center justify-center py-12 text-xl text-center h-screen">
      {/* Progress Circle */}
      <ProgressCircle percentage={progress} />

      {/* Loading Message */}
      <p
        className={`text-lg transition-opacity duration-500 ease-in-out mb-4 ${
          fade ? "opacity-100" : "opacity-0"
        } ${index === messages.length - 1 ? "animate-pulse" : ""}`}
      >
        {getCurrentMessage()}
      </p>

      {/* Additional status if still creating */}
      {isCreatingPortfolio && (
        <p className="text-sm text-gray-600 mt-2 animate-pulse max-w-md">
          Our AI is analyzing market conditions and optimizing your portfolio...
        </p>
      )}

      {/* Completion message */}
      {progress >= 100 && (
        <div className="mt-4 text-green-600 font-semibold animate-pulse">
          🎉 Portfolio ready! Redirecting...
        </div>
      )}
    </div>
  );
}
