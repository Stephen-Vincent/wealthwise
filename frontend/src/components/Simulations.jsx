import { useEffect, useState } from "react";
import { usePortfolio } from "../context/PortfolioContext";

export default function Simulations({ onBack, onShowLoading }) {
  const { setPortfolioData } = usePortfolio();
  const [simulations, setSimulations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [userId, setUserId] = useState(localStorage.getItem("userId"));
  const [isVisible, setIsVisible] = useState(false);

  // Fade in effect when component mounts
  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 100);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const storedUser = JSON.parse(localStorage.getItem("user"));
    if (storedUser?.id) {
      setUserId(storedUser.id);
    } else {
      const fallbackUserId = localStorage.getItem("userId");
      if (fallbackUserId) {
        setUserId(fallbackUserId);
      }
    }
  }, []);

  // Function to load a simulation by ID, update context, and show loading panel
  const loadSimulationAndShowLoading = async (id) => {
    console.log("🧲 Loading simulation with ID:", id);
    const token = localStorage.getItem("access_token");
    try {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL}/simulations/${id}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );
      if (!res.ok) throw new Error("Failed to fetch simulation by ID");
      const simData = await res.json();
      console.log("✅ Simulation data loaded:", simData);

      // Set the portfolio data in context
      setPortfolioData(simData);

      // Store IDs for navigation
      localStorage.setItem("simulationId", id);
      localStorage.setItem("userId", userId);

      // Show loading panel instead of navigating
      if (onShowLoading) {
        onShowLoading();
      }
    } catch (err) {
      console.error("❌ Error loading simulation:", err);
    }
  };

  useEffect(() => {
    const fetchSimulations = async () => {
      if (!userId) {
        console.log("userId is null, skipping fetch.");
        setLoading(false);
        return;
      }
      const token = localStorage.getItem("access_token");
      if (!token) {
        console.error("No auth token found");
        setError("You must be logged in.");
        setLoading(false);
        return;
      }
      try {
        const res = await fetch(`${import.meta.env.VITE_API_URL}/simulations`, {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        });
        if (!res.ok) throw new Error("Failed to fetch simulations");
        const data = await res.json();
        console.log("📥 Simulations fetched:", data);
        setSimulations(data);
      } catch (err) {
        console.error("Error fetching simulations:", err);
        setError("There was an error loading your simulations.");
      } finally {
        setLoading(false);
      }
    };

    console.log("Simulations useEffect userId:", userId);
    fetchSimulations();
  }, [userId]);

  const deleteSimulation = async (id) => {
    const token = localStorage.getItem("access_token");
    if (!token) {
      console.error("No auth token found");
      setError("You must be logged in.");
      return;
    }
    try {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL}/simulations/${id}`,
        {
          method: "DELETE",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );
      if (!res.ok) throw new Error("Failed to delete simulation");
      setSimulations(simulations.filter((sim) => sim.id !== id));
    } catch (err) {
      console.error("Error deleting simulation:", err);
      setError("Failed to delete simulation.");
    }
  };

  return (
    <div
      className={`mt-24 flex flex-col items-center text-center px-8 py-6 font-sans transition-all duration-500 ease-in-out ${
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
      }`}
      style={{ height: "calc(100vh - 250px)" }}
    >
      <div className="w-full max-w-2xl h-full flex flex-col">
        <h2 className="text-2xl font-bold mb-6 text-center flex-shrink-0">
          Your Simulations
        </h2>

        {loading ? (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-center">Loading simulations...</p>
          </div>
        ) : error ? (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-center text-red-500">{error}</p>
          </div>
        ) : simulations.length === 0 ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center space-y-4">
              <p className="text-gray-600">No previous simulations found.</p>
              <p className="text-sm text-gray-500">
                Use the back button to return and start a new simulation from
                the welcome screen.
              </p>
            </div>
          </div>
        ) : (
          <>
            {/* Scrollable simulations list with max height */}
            <div
              className="flex-1 overflow-y-auto mb-6 border border-gray-200 rounded-lg bg-white/50"
              style={{ maxHeight: "calc(100vh - 350px)" }}
            >
              <div className="space-y-3 p-4">
                {simulations.map((sim, index) => (
                  <div
                    key={sim.id}
                    className={`p-4 border rounded-lg hover:bg-white/80 flex justify-between items-start transition-all duration-150 bg-white shadow-sm cursor-pointer hover:shadow-md transform transition-all ${
                      isVisible
                        ? "opacity-100 translate-y-0"
                        : "opacity-0 translate-y-4"
                    }`}
                    style={{
                      transitionDelay: isVisible ? `${index * 100}ms` : "0ms",
                      transitionDuration: "400ms",
                    }}
                    onClick={() => {
                      console.log("🔍 Simulation clicked with ID:", sim.id);
                      loadSimulationAndShowLoading(sim.id);
                    }}
                  >
                    <div className="w-full text-left">
                      <p className="font-semibold mb-3 text-lg text-gray-800">
                        🎯 Goal: {sim.goal || "N/A"}
                      </p>
                      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                        <div className="flex items-center space-x-2">
                          <span>⏳</span>
                          <span className="text-gray-700">
                            Timeline: {sim.timeframe || "N/A"}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span>✅</span>
                          <span className="text-gray-700">
                            Target:{" "}
                            <span
                              className={
                                sim.target_achieved
                                  ? "text-green-600 font-medium"
                                  : "text-red-600 font-medium"
                              }
                            >
                              {sim.target_achieved === true
                                ? "Yes"
                                : sim.target_achieved === false
                                ? "No"
                                : "N/A"}
                            </span>
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span>💰</span>
                          <span className="text-gray-700">
                            Income: {sim.income_bracket || "N/A"}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span>⚖️</span>
                          <span className="text-gray-700">
                            Risk Score: {sim.risk_score ?? "N/A"}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span>🔥</span>
                          <span className="text-gray-700">
                            Risk Level: {sim.risk_label ?? "N/A"}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span>📅</span>
                          <span className="text-gray-500">
                            {sim.created_at
                              ? new Date(sim.created_at).toLocaleDateString(
                                  "en-GB",
                                  {
                                    year: "numeric",
                                    month: "short",
                                    day: "numeric",
                                  }
                                )
                              : "Unknown"}
                          </span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteSimulation(sim.id);
                      }}
                      className="ml-4 px-3 py-2 text-sm text-white bg-red-500 rounded-lg hover:bg-red-600 transition-colors duration-150 flex-shrink-0 font-medium"
                    >
                      Delete
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {/* Back button - always visible at bottom */}
        <div className="flex-shrink-0">
          <button
            onClick={onBack}
            className="bg-gray-600 text-white px-6 py-3 rounded-lg font-bold w-full hover:bg-gray-700 transition-colors duration-150 shadow-lg"
          >
            ← Back to Welcome
          </button>
        </div>
      </div>
    </div>
  );
}
