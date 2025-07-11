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
  const [index, setIndex] = useState(0);
  const [fade, setFade] = useState(true);
  const [minTimeElapsed, setMinTimeElapsed] = useState(false);

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

  // Ensure minimum loading time (3 seconds after the last message)
  useEffect(() => {
    const minLoadingTime = setTimeout(() => {
      setMinTimeElapsed(true);
    }, 4000); // 4 seconds (messages cycle for ~3 seconds + 1 extra)

    return () => clearTimeout(minLoadingTime);
  }, []);

  // Check if portfolio data is ready and navigate when conditions are met
  useEffect(() => {
    const checkDataAndNavigate = () => {
      // Check for the actual data structure based on console logs
      const isDataReady =
        portfolioData &&
        portfolioData.id &&
        (portfolioData.stocks_picked || portfolioData.breakdown);

      console.log("🔍 Data check:", {
        hasPortfolioData: !!portfolioData,
        hasId: !!portfolioData?.id,
        hasStocksPicked: !!portfolioData?.stocks_picked,
        hasBreakdown: !!portfolioData?.breakdown,
        hasRiskScore: !!portfolioData?.risk_score,
        minTimeElapsed,
        userId,
        simulationId,
      });

      // Only navigate if data is ready, minimum time has elapsed, and we have valid IDs
      if (isDataReady && minTimeElapsed && userId && simulationId) {
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

    checkDataAndNavigate();
  }, [portfolioData, minTimeElapsed, userId, simulationId, navigate]);

  // Fallback: If data isn't ready after a reasonable time, show error or retry
  useEffect(() => {
    const fallbackTimeout = setTimeout(() => {
      if (!portfolioData || !portfolioData.id) {
        console.error(
          "❌ Data not ready after 15 seconds, something may be wrong"
        );
        // You could show an error message or retry logic here
        // For now, we'll just log the error
      }
    }, 15000); // 15 seconds

    return () => clearTimeout(fallbackTimeout);
  }, [portfolioData]);

  return (
    <div className="flex flex-col items-center py-12 text-xl text-center h-screen">
      <p
        className={`text-lg transition-opacity duration-500 ease-in-out mt-32 ${
          fade ? "opacity-100" : "opacity-0"
        } ${index === messages.length - 1 ? "animate-pulse" : ""}`}
      >
        {messages[index]}
      </p>
    </div>
  );
}
