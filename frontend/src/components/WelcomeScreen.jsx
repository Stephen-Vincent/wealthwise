import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

// Welcome screen component for logged-in users
export default function WelcomeScreen({
  onBack,
  userName,
  onShowOnboarding,
  onShowSimulations,
}) {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [hasSimulations, setHasSimulations] = useState(null);
  const [isVisible, setIsVisible] = useState(false);

  // Fade in effect when component mounts
  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 100);
    return () => clearTimeout(timer);
  }, []);

  // Check if user has simulations when component loads
  useEffect(() => {
    checkForSimulations();
  }, []);

  const checkForSimulations = async () => {
    const token = localStorage.getItem("access_token");
    const userId = localStorage.getItem("userId");

    console.log("🔍 Checking simulations - Token:", !!token, "UserId:", userId);

    // ADD DEBUG LOGGING
    console.log("🔍 WELCOMESCREEN DEBUG:");
    console.log("VITE_API_URL:", import.meta.env.VITE_API_URL);
    console.log("Environment mode:", import.meta.env.MODE);
    console.log("All env vars:", import.meta.env);

    const apiUrl = `${import.meta.env.VITE_API_URL}/simulations/`;
    console.log("🔍 About to fetch URL:", apiUrl);

    try {
      const res = await fetch(apiUrl, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      console.log("🔍 Simulations response status:", res.status, res.ok);

      if (res.ok) {
        const simulations = await res.json();
        console.log(
          "🔍 Simulations data:",
          simulations,
          "Length:",
          simulations.length
        );
        setHasSimulations(simulations.length > 0);
      } else {
        console.log("🔍 Simulations response not ok");
        setHasSimulations(false);
      }
    } catch (err) {
      console.error("Error checking simulations:", err);
      setHasSimulations(false);
    }
  };

  const handleViewSimulations = async () => {
    setIsLoading(true);

    // Fade out effect
    setIsVisible(false);

    // Show simulations panel instead of navigating
    setTimeout(() => {
      if (onShowSimulations) {
        onShowSimulations(); // Call the function passed from LandingScreen
      } else {
        // Fallback to navigation if onShowSimulations is not provided
        navigate("/simulations");
      }
      setIsLoading(false);
    }, 300);
  };

  const handleNewSimulation = async () => {
    setIsLoading(true);

    // Fade out effect
    setIsVisible(false);

    // Show onboarding instead of navigating
    setTimeout(() => {
      if (onShowOnboarding) {
        onShowOnboarding(); // Call the function passed from LandingScreen
      } else {
        // Fallback to navigation if onShowOnboarding is not provided
        const userId = localStorage.getItem("userId");
        navigate(`/onboarding/${userId}`);
      }
      setIsLoading(false);
    }, 300);
  };

  const handleLogout = () => {
    // Fade out effect
    setIsVisible(false);

    setTimeout(() => {
      // Clear authentication data
      localStorage.removeItem("access_token");
      localStorage.removeItem("user");
      localStorage.removeItem("userId");
      localStorage.removeItem("user_name");
      localStorage.removeItem("simulationId");

      // Return to main welcome
      onBack();
    }, 300);
  };

  return (
    <div
      className={`flex flex-col items-center justify-center w-full space-y-6 transition-all duration-500 ease-in-out ${
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
      }`}
    >
      <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-md text-center space-y-6">
        <div className="space-y-2">
          <h2 className="text-3xl font-bold text-[#333]">
            {hasSimulations
              ? `Welcome back${userName ? `, ${userName}` : ""}!`
              : `Welcome${userName ? `, ${userName}` : ""}!`}
          </h2>
        </div>

        <div className="space-y-4">
          {/* View Simulations Button - Only show if user has simulations */}
          {hasSimulations && (
            <button
              onClick={handleViewSimulations}
              disabled={isLoading}
              className="bg-[#00A8FF] text-white font-bold px-8 py-4 rounded-[15px] w-full shadow-lg hover:scale-105 active:scale-95 transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
            >
              {isLoading ? "Loading..." : "View My Simulations"}
            </button>
          )}

          {/* New Simulation Button */}
          <button
            onClick={handleNewSimulation}
            disabled={isLoading}
            className={`${
              hasSimulations
                ? "bg-white text-[#00A8FF] border-2 border-[#00A8FF]"
                : "bg-[#00A8FF] text-white"
            } font-bold px-8 py-4 rounded-[15px] w-full shadow-lg hover:scale-105 active:scale-95 transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100`}
          >
            {isLoading ? "Loading..." : "Start New Simulation"}
          </button>
        </div>

        {/* Show different messages based on simulation status */}
        <div className="pt-4 border-t border-gray-200">
          {hasSimulations === null ? (
            <p className="text-sm text-gray-500">
              Checking your simulations...
            </p>
          ) : hasSimulations ? (
            <p className="text-sm text-gray-500">
              Continue where you left off or create something new
            </p>
          ) : (
            <p className="text-sm text-gray-500">
              Start your first investment simulation
            </p>
          )}
        </div>
      </div>

      <button
        onClick={handleLogout}
        disabled={isLoading}
        className="text-[#00A8FF] underline text-sm hover:text-[#0088CC] transition-colors duration-150 disabled:opacity-50"
      >
        ← Logout
      </button>
    </div>
  );
}
