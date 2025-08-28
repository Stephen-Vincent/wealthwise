/**
 * LoadingScreen.jsx
 *
 * This component serves as the loading screen for WealthWise.
 * It displays animated progress and cycling status messages while the AI creates the user's personalized portfolio.
 * The component manages navigation timing, ensuring that the user is only redirected to the dashboard after all required data
 * is loaded and a minimum display duration is reached. It provides smooth progress animations and handles fallback and emergency
 * navigation scenarios to avoid stuck states, while offering clear feedback throughout the loading process.
 */
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

  // Get fresh isCreatingPortfolio state and add debugging
  const [isCreatingPortfolio, setIsCreatingPortfolio] = useState(() => {
    const flag = localStorage.getItem("isCreatingPortfolio") === "true";
    console.log("📱 Initial isCreatingPortfolio flag:", flag);
    return flag;
  });

  const [index, setIndex] = useState(0);
  const [fade, setFade] = useState(true);
  const [minTimeElapsed, setMinTimeElapsed] = useState(false);
  const [progress, setProgress] = useState(0);
  const [readyToNavigate, setReadyToNavigate] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);

  // NEW: Add loop breakers
  const [debugCount, setDebugCount] = useState(0);
  const [hasNavigationAttempted, setHasNavigationAttempted] = useState(false);

  // Monitor and clear the isCreatingPortfolio flag
  useEffect(() => {
    const checkAndClearFlag = () => {
      const currentFlag =
        localStorage.getItem("isCreatingPortfolio") === "true";

      // Clear flag if we have portfolio data
      if (portfolioData && portfolioData.id && currentFlag) {
        console.log(
          "🔄 Clearing isCreatingPortfolio flag - portfolio data received"
        );
        localStorage.setItem("isCreatingPortfolio", "false");
        setIsCreatingPortfolio(false);
      }

      // Force clear flag after 8 seconds to prevent infinite loading
      setTimeout(() => {
        if (localStorage.getItem("isCreatingPortfolio") === "true") {
          console.log(
            "⏰ Force clearing isCreatingPortfolio flag after 8 seconds"
          );
          localStorage.setItem("isCreatingPortfolio", "false");
          setIsCreatingPortfolio(false);
        }
      }, 8000);
    };

    checkAndClearFlag();
  }, [portfolioData]);

  // FIXED: Enhanced progress calculation with limited debugging and exit conditions
  useEffect(() => {
    if (!hasStarted) {
      setHasStarted(true);
    }

    let animationFrame;
    let startTime = Date.now();
    const minLoadingDuration = 4000; // 4 seconds minimum
    let debugLogCount = 0; // LIMIT DEBUG LOGS

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const timeProgress = Math.min(elapsed / minLoadingDuration, 1);

      // Calculate target progress with detailed logging
      let targetProgress = 0;
      let progressReason = "initializing";

      // Always start with time-based progress (0-70% over 4 seconds)
      targetProgress = timeProgress * 70;
      progressReason = "time-based";

      // Add data-based progress
      if (portfolioData && portfolioData.id) {
        targetProgress = Math.max(targetProgress, 85);
        progressReason = "data-received";
      }

      // FIXED: Force boolean values - never null
      const hasRequiredData = !!(portfolioData && portfolioData.id);
      const hasRequiredIds = !!(userId && simulationId);
      const timeComplete = !!minTimeElapsed;

      if (hasRequiredData && hasRequiredIds && timeComplete) {
        targetProgress = 100;
        progressReason = "all-conditions-met";
      }

      // Limited debug logging - only first 3 logs after 5 seconds
      if (targetProgress < 100 && elapsed > 5000 && debugLogCount < 3) {
        console.log("🔍 Progress Debug:", {
          targetProgress,
          progressReason,
          hasRequiredData,
          hasRequiredIds,
          timeComplete,
          isCreatingPortfolio,
          elapsed: Math.round(elapsed / 1000) + "s",
        });
        debugLogCount++;
      }

      // Smooth progress animation using easing - ONLY MOVE FORWARD
      setProgress((prev) => {
        const newTarget = Math.max(prev, targetProgress); // Never go backwards
        const diff = newTarget - prev;

        if (diff < 0.1) {
          return newTarget;
        }

        // Smooth easing - slower near the end
        const increment = diff * (newTarget > 90 ? 0.08 : 0.03);
        return Math.min(prev + increment, 100);
      });

      // Add exit condition - stop after 15 seconds max
      if (targetProgress < 100 && elapsed < 15000) {
        animationFrame = requestAnimationFrame(animate);
      } else if (elapsed >= 15000) {
        console.log("🛑 Animation stopped - 15 second timeout");
        setProgress(100);
      }
    };

    animationFrame = requestAnimationFrame(animate);

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [
    portfolioData,
    minTimeElapsed,
    isCreatingPortfolio,
    userId,
    simulationId,
    hasStarted,
  ]);

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
      console.log("⏰ Minimum loading time elapsed");
      setMinTimeElapsed(true);
    }, 4000);

    return () => clearTimeout(minLoadingTime);
  }, []);

  // Check if ready to navigate with limited logging and exit conditions
  useEffect(() => {
    // Limit debug attempts
    if (debugCount > 5) {
      console.log("🛑 Debug limit reached, stopping navigation checks");
      return;
    }

    const checkReadyToNavigate = () => {
      // FORCE BOOLEAN VALUES - never null
      const isDataReady = !!(
        portfolioData &&
        portfolioData.id &&
        (portfolioData.stocks_picked ||
          portfolioData.breakdown ||
          portfolioData.results)
      );

      const canNavigate = !!(
        isDataReady &&
        minTimeElapsed &&
        userId &&
        simulationId
      );

      setReadyToNavigate(canNavigate);

      // CONTROLLED LOGGING - only log first few attempts
      if (debugCount < 5) {
        console.log("🔍 Navigation check:", {
          hasPortfolioData: !!portfolioData,
          hasId: !!portfolioData?.id,
          hasStocks: !!portfolioData?.stocks_picked?.length,
          hasBreakdown: !!portfolioData?.breakdown,
          hasResults: !!portfolioData?.results,
          minTimeElapsed,
          userId: !!userId,
          simulationId: !!simulationId,
          isCreatingPortfolio,
          canNavigate,
          progress: Math.round(progress),
          debugCount,
        });
        setDebugCount((prev) => prev + 1);
      }
    };

    // STOP THE INTERVAL: Only check a few times, not forever
    const interval = setInterval(() => {
      if (debugCount < 10 && !hasNavigationAttempted) {
        checkReadyToNavigate();
      } else {
        clearInterval(interval);
      }
    }, 1000);

    checkReadyToNavigate(); // Initial check

    return () => clearInterval(interval);
  }, [
    portfolioData,
    minTimeElapsed,
    userId,
    simulationId,
    isCreatingPortfolio,
    progress,
    debugCount,
    hasNavigationAttempted,
  ]);

  // FIXED: Navigate when progress reaches 100% and ready - prevent multiple attempts
  useEffect(() => {
    if (readyToNavigate && progress >= 99 && !hasNavigationAttempted) {
      // Lowered threshold to 99%
      console.log("✅ Navigation conditions met, redirecting to dashboard");
      console.log("📊 Final navigation data:", {
        portfolioId: portfolioData?.id,
        userId,
        simulationId,
        progress: Math.round(progress),
        hasShap: !!portfolioData?.shap_explanation,
      });

      setHasNavigationAttempted(true);
      // Clear the creating portfolio flag
      localStorage.setItem("isCreatingPortfolio", "false");
      setIsCreatingPortfolio(false);

      setTimeout(() => {
        navigate(`/dashboard/${userId}/${simulationId}`);
      }, 500);
    }
  }, [
    readyToNavigate,
    progress,
    navigate,
    userId,
    simulationId,
    portfolioData,
    hasNavigationAttempted,
  ]);

  // FIXED: Enhanced fallback navigation with better conditions and navigation guard
  useEffect(() => {
    const fallbackTimeout = setTimeout(() => {
      if (
        portfolioData &&
        portfolioData.id &&
        userId &&
        simulationId &&
        !hasNavigationAttempted
      ) {
        console.log("⏰ Fallback navigation triggered");
        console.log("🔍 Fallback triggered because:", {
          progress: Math.round(progress),
          readyToNavigate,
          isCreatingPortfolio,
          hasMinTimeElapsed: minTimeElapsed,
          portfolioDataKeys: Object.keys(portfolioData || {}),
        });

        setHasNavigationAttempted(true); // PREVENT MULTIPLE ATTEMPTS
        localStorage.setItem("isCreatingPortfolio", "false");
        setIsCreatingPortfolio(false);
        navigate(`/dashboard/${userId}/${simulationId}`);
      }
    }, 10000);

    return () => clearTimeout(fallbackTimeout);
  }, [
    portfolioData,
    userId,
    simulationId,
    navigate,
    progress,
    readyToNavigate,
    isCreatingPortfolio,
    minTimeElapsed,
    hasNavigationAttempted,
  ]);

  // FIXED: Emergency navigation for stuck states with navigation guard
  useEffect(() => {
    const emergencyTimeout = setTimeout(() => {
      if (
        portfolioData &&
        portfolioData.id &&
        userId &&
        simulationId &&
        progress < 99 &&
        !hasNavigationAttempted
      ) {
        console.log(
          "🚨 Emergency navigation - progress stuck at",
          Math.round(progress)
        );
        setHasNavigationAttempted(true); // PREVENT MULTIPLE ATTEMPTS
        localStorage.setItem("isCreatingPortfolio", "false");
        setIsCreatingPortfolio(false);
        navigate(`/dashboard/${userId}/${simulationId}`);
      }
    }, 15000); // 15 second emergency backup

    return () => clearTimeout(emergencyTimeout);
  }, [
    portfolioData,
    userId,
    simulationId,
    navigate,
    progress,
    hasNavigationAttempted,
  ]);

  // Progress circle component with smooth animations
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
          {/* Progress circle with smooth transition */}
          <circle
            cx="64"
            cy="64"
            r={radius}
            stroke="currentColor"
            strokeWidth="8"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className="text-[#00A8FF] transition-all duration-100 ease-out"
            strokeLinecap="round"
          />
        </svg>
        {/* Percentage text with smooth counting */}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold text-[#00A8FF] tabular-nums">
            {Math.round(percentage)}%
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

  // Get status message based on progress
  const getStatusMessage = () => {
    if (isCreatingPortfolio) {
      return "Our AI is analyzing market conditions and optimizing your portfolio...";
    }

    if (progress < 25) {
      return "Initializing analysis systems...";
    } else if (progress < 50) {
      return "Processing your risk profile and goals...";
    } else if (progress < 75) {
      return "Analyzing market data and stock performance...";
    } else if (progress < 95) {
      return "Optimizing your personalized portfolio...";
    } else if (progress < 100) {
      return "Finalizing recommendations...";
    } else {
      return "Portfolio optimization complete!";
    }
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

      {/* Dynamic status message */}
      <p className="text-sm text-gray-600 mt-2 max-w-md">
        {getStatusMessage()}
      </p>

      {/* Completion message */}
      {progress >= 99 && readyToNavigate && (
        <div className="mt-4 text-green-600 font-semibold animate-pulse">
          🎉 Portfolio ready! Redirecting...
        </div>
      )}

      {/* Debug info in development */}
      {process.env.NODE_ENV === "development" && (
        <div className="fixed bottom-4 left-4 text-xs bg-black text-white p-2 rounded max-w-xs">
          <div>Progress: {Math.round(progress)}%</div>
          <div>Ready: {readyToNavigate ? "Yes" : "No"}</div>
          <div>Creating: {isCreatingPortfolio ? "Yes" : "No"}</div>
          <div>Min Time: {minTimeElapsed ? "Yes" : "No"}</div>
          <div>Data: {portfolioData?.id ? "Yes" : "No"}</div>
          <div>Debug Count: {debugCount}</div>
          <div>Nav Attempted: {hasNavigationAttempted ? "Yes" : "No"}</div>
        </div>
      )}
    </div>
  );
}
