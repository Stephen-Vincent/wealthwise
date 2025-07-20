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
  const [readyToNavigate, setReadyToNavigate] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);

  // Simplified smooth progress animation
  useEffect(() => {
    if (!hasStarted) {
      setHasStarted(true);
    }

    let animationFrame;
    let startTime = Date.now();
    const minLoadingDuration = 4000; // 4 seconds minimum

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const timeProgress = Math.min(elapsed / minLoadingDuration, 1);

      // Calculate target progress - SIMPLIFIED LOGIC
      let targetProgress = 0;

      // Always start with time-based progress (0-80% over 4 seconds)
      targetProgress = timeProgress * 80;

      // Add data-based progress
      if (portfolioData && portfolioData.id) {
        targetProgress = Math.max(targetProgress, 90);
      }

      // Complete when all conditions are met
      if (
        portfolioData &&
        portfolioData.id &&
        minTimeElapsed &&
        userId &&
        simulationId &&
        !isCreatingPortfolio
      ) {
        targetProgress = 100;
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

      // Continue animation if not complete
      if (targetProgress < 100) {
        animationFrame = requestAnimationFrame(animate);
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
      setMinTimeElapsed(true);
    }, 4000);

    return () => clearTimeout(minLoadingTime);
  }, []);

  // Check if ready to navigate - SIMPLIFIED
  useEffect(() => {
    const checkReadyToNavigate = () => {
      // Clear creating portfolio flag if we have data
      if (portfolioData && portfolioData.id && isCreatingPortfolio) {
        localStorage.setItem("isCreatingPortfolio", "false");
      }

      const isDataReady =
        portfolioData &&
        portfolioData.id &&
        (portfolioData.stocks_picked ||
          portfolioData.breakdown ||
          portfolioData.results);

      const canNavigate =
        isDataReady && minTimeElapsed && userId && simulationId;

      setReadyToNavigate(canNavigate);

      console.log("🔍 Navigation check:", {
        hasPortfolioData: !!portfolioData,
        hasId: !!portfolioData?.id,
        minTimeElapsed,
        userId: !!userId,
        simulationId: !!simulationId,
        canNavigate,
        progress: Math.round(progress),
      });
    };

    const interval = setInterval(checkReadyToNavigate, 1000);
    checkReadyToNavigate();

    return () => clearInterval(interval);
  }, [
    portfolioData,
    minTimeElapsed,
    userId,
    simulationId,
    isCreatingPortfolio,
    progress,
  ]);

  // Navigate when progress reaches 100% and ready
  useEffect(() => {
    if (readyToNavigate && progress >= 99.5) {
      console.log("✅ Navigation conditions met, redirecting to dashboard");

      // Clear the creating portfolio flag
      localStorage.setItem("isCreatingPortfolio", "false");

      setTimeout(() => {
        navigate(`/dashboard/${userId}/${simulationId}`);
      }, 500);
    }
  }, [readyToNavigate, progress, navigate, userId, simulationId]);

  // Fallback navigation after 10 seconds if we have data
  useEffect(() => {
    const fallbackTimeout = setTimeout(() => {
      if (portfolioData && portfolioData.id && userId && simulationId) {
        console.log("⏰ Fallback navigation triggered");
        localStorage.setItem("isCreatingPortfolio", "false");
        navigate(`/dashboard/${userId}/${simulationId}`);
      }
    }, 10000);

    return () => clearTimeout(fallbackTimeout);
  }, [portfolioData, userId, simulationId, navigate]);

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
      {progress >= 99.5 && readyToNavigate && (
        <div className="mt-4 text-green-600 font-semibold animate-pulse">
          🎉 Portfolio ready! Redirecting...
        </div>
      )}
    </div>
  );
}
