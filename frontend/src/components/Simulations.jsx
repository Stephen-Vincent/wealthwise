/**
 * Simulations Component
 * ---------------------
 * The Simulations component displays a user's list of saved portfolio simulations,
 * providing features for searching, filtering (by goal, risk, achievement), and sorting.
 * It fetches simulation data for the current user from the backend API using an auth token,
 * and supports deleting simulations or loading one into the main portfolio context.
 *
 * Main Functionality:
 * - Fetches all simulations for the current user on mount and when userId changes.
 * - Allows users to search, filter, and sort their simulations.
 * - Displays simulation details in a scrollable list with delete and load actions.
 * - Provides a loading state, error handling, and a back button for navigation.
 *
 * State/Context Interaction:
 * - Uses useState for UI state (filters, loading, error, etc.).
 * - Uses usePortfolio context to update the global portfolio data when a simulation is loaded.
 * - Reads user and authentication data from localStorage.
 * - Calls onShowLoading prop to trigger loading panel when a simulation is loaded.
 */
import { useEffect, useState, useMemo } from "react";
import { usePortfolio } from "../context/PortfolioContext";

export default function Simulations({ onBack, onShowLoading }) {
  const { setPortfolioData } = usePortfolio();
  const [simulations, setSimulations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [userId, setUserId] = useState(localStorage.getItem("userId"));
  const [isVisible, setIsVisible] = useState(false);

  // Filter and sort states
  const [sortBy, setSortBy] = useState("date_desc"); // date_desc, date_asc, goal, risk_score, target_value
  const [filterGoal, setFilterGoal] = useState("all");
  const [filterRisk, setFilterRisk] = useState("all");
  const [filterAchieved, setFilterAchieved] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    console.log("🔍 SIMULATIONS DEBUG:");
    console.log("VITE_API_URL:", import.meta.env.VITE_API_URL);
    console.log(
      "Full constructed URL:",
      `${import.meta.env.VITE_API_URL}/simulations/`
    );
    console.log("Environment mode:", import.meta.env.MODE);
    console.log("All env vars:", import.meta.env);
  }, []);

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

  // Get unique values for filter dropdowns
  const uniqueGoals = useMemo(() => {
    const goals = [
      ...new Set(simulations.map((sim) => sim.goal).filter(Boolean)),
    ];
    return goals.sort();
  }, [simulations]);

  const uniqueRiskLevels = useMemo(() => {
    const risks = [
      ...new Set(simulations.map((sim) => sim.risk_label).filter(Boolean)),
    ];
    return risks.sort();
  }, [simulations]);

  // Filter and sort simulations
  const filteredAndSortedSimulations = useMemo(() => {
    let filtered = simulations.filter((sim) => {
      // Search filter
      if (
        searchTerm &&
        !sim.goal?.toLowerCase().includes(searchTerm.toLowerCase())
      ) {
        return false;
      }

      // Goal filter
      if (filterGoal !== "all" && sim.goal !== filterGoal) {
        return false;
      }

      // Risk filter
      if (filterRisk !== "all" && sim.risk_label !== filterRisk) {
        return false;
      }

      // Achievement filter
      if (filterAchieved !== "all") {
        if (filterAchieved === "achieved" && !sim.target_achieved) {
          return false;
        }
        if (filterAchieved === "not_achieved" && sim.target_achieved) {
          return false;
        }
      }

      return true;
    });

    // Sort simulations
    filtered.sort((a, b) => {
      switch (sortBy) {
        case "date_desc":
          return new Date(b.created_at || 0) - new Date(a.created_at || 0);
        case "date_asc":
          return new Date(a.created_at || 0) - new Date(b.created_at || 0);
        case "goal":
          return (a.goal || "").localeCompare(b.goal || "");
        case "risk_score":
          return (b.risk_score || 0) - (a.risk_score || 0);
        case "target_value":
          return (b.target_value || 0) - (a.target_value || 0);
        default:
          return 0;
      }
    });

    return filtered;
  }, [simulations, sortBy, filterGoal, filterRisk, filterAchieved, searchTerm]);

  const clearFilters = () => {
    setSortBy("date_desc");
    setFilterGoal("all");
    setFilterRisk("all");
    setFilterAchieved("all");
    setSearchTerm("");
  };

  const hasActiveFilters =
    sortBy !== "date_desc" ||
    filterGoal !== "all" ||
    filterRisk !== "all" ||
    filterAchieved !== "all" ||
    searchTerm !== "";

  return (
    <div
      className={`mt-24 flex flex-col items-center text-center px-8 py-6 font-sans transition-all duration-500 ease-in-out ${
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
      }`}
      style={{ height: "calc(100vh - 50px)" }}
    >
      <div className="w-full max-w-4xl h-full flex flex-col">
        {/* Header with count */}
        <div className="flex-shrink-0 mb-6">
          <h2 className="text-2xl font-bold mb-2 text-center">
            Your Simulations
          </h2>
          <p className="text-gray-600 text-center">
            {loading ? (
              "Loading..."
            ) : (
              <>
                {filteredAndSortedSimulations.length} of {simulations.length}{" "}
                simulation{simulations.length !== 1 ? "s" : ""}
                {hasActiveFilters && " (filtered)"}
              </>
            )}
          </p>
        </div>

        {loading ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p>Loading your simulations...</p>
            </div>
          </div>
        ) : error ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center text-red-500">
              <p className="text-lg font-semibold mb-2">Error</p>
              <p>{error}</p>
            </div>
          </div>
        ) : simulations.length === 0 ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center space-y-4">
              <div className="text-6xl mb-4">📊</div>
              <p className="text-xl font-semibold text-gray-700">
                No simulations yet
              </p>
              <p className="text-gray-600">
                Create your first simulation to see portfolio recommendations!
              </p>
              <p className="text-sm text-gray-500">
                Use the back button to return and start a new simulation from
                the welcome screen.
              </p>
            </div>
          </div>
        ) : (
          <>
            {/* Filters and Search */}
            <div className="flex-shrink-0 mb-4 bg-white/70 rounded-lg p-4 border border-gray-200">
              {/* Search Bar */}
              <div className="mb-4">
                <input
                  type="text"
                  placeholder="🔍 Search by goal..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                />
              </div>

              {/* Filter Controls */}
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3">
                {/* Sort By */}
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Sort by
                  </label>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                  >
                    <option value="date_desc">Latest First</option>
                    <option value="date_asc">Oldest First</option>
                    <option value="goal">Goal A-Z</option>
                    <option value="risk_score">Risk Score</option>
                    <option value="target_value">Target Value</option>
                  </select>
                </div>

                {/* Filter by Goal */}
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Goal
                  </label>
                  <select
                    value={filterGoal}
                    onChange={(e) => setFilterGoal(e.target.value)}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                  >
                    <option value="all">All Goals</option>
                    {uniqueGoals.map((goal) => (
                      <option key={goal} value={goal}>
                        {goal.length > 20
                          ? goal.substring(0, 20) + "..."
                          : goal}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Filter by Risk */}
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Risk Level
                  </label>
                  <select
                    value={filterRisk}
                    onChange={(e) => setFilterRisk(e.target.value)}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                  >
                    <option value="all">All Levels</option>
                    {uniqueRiskLevels.map((risk) => (
                      <option key={risk} value={risk}>
                        {risk}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Filter by Achievement */}
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Target
                  </label>
                  <select
                    value={filterAchieved}
                    onChange={(e) => setFilterAchieved(e.target.value)}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                  >
                    <option value="all">All Results</option>
                    <option value="achieved">Achieved ✅</option>
                    <option value="not_achieved">Not Achieved ❌</option>
                  </select>
                </div>

                {/* Clear Filters */}
                <div className="flex items-end">
                  <button
                    onClick={clearFilters}
                    disabled={!hasActiveFilters}
                    className={`w-full px-3 py-2 text-sm rounded-md transition-colors duration-150 ${
                      hasActiveFilters
                        ? "bg-gray-600 text-white hover:bg-gray-700"
                        : "bg-gray-200 text-gray-400 cursor-not-allowed"
                    }`}
                  >
                    Clear Filters
                  </button>
                </div>
              </div>
            </div>

            {/* Results */}
            {filteredAndSortedSimulations.length === 0 ? (
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center space-y-4">
                  <div className="text-4xl mb-4">🔍</div>
                  <p className="text-xl font-semibold text-gray-700">
                    No matching simulations
                  </p>
                  <p className="text-gray-600">
                    Try adjusting your filters or search terms.
                  </p>
                  <button
                    onClick={clearFilters}
                    className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors duration-150"
                  >
                    Clear All Filters
                  </button>
                </div>
              </div>
            ) : (
              /* Scrollable simulations list */
              <div
                className="flex-1 overflow-y-auto mb-6 border border-gray-200 rounded-lg bg-white/50"
                style={{ maxHeight: "calc(100vh - 450px)" }}
              >
                <div className="space-y-3 p-4">
                  {filteredAndSortedSimulations.map((sim, index) => (
                    <div
                      key={sim.id}
                      className={`p-4 border rounded-lg hover:bg-white/80 flex justify-between items-start transition-all duration-150 bg-white shadow-sm cursor-pointer hover:shadow-md transform transition-all ${
                        isVisible
                          ? "opacity-100 translate-y-0"
                          : "opacity-0 translate-y-4"
                      }`}
                      style={{
                        transitionDelay: isVisible ? `${index * 50}ms` : "0ms",
                        transitionDuration: "400ms",
                      }}
                      onClick={() => {
                        console.log("🔍 Simulation clicked with ID:", sim.id);
                        loadSimulationAndShowLoading(sim.id);
                      }}
                    >
                      <div className="w-full text-left">
                        <div className="flex items-start justify-between mb-3">
                          <p className="font-semibold text-lg text-gray-800 flex-1">
                            🎯 {sim.goal || "Untitled Goal"}
                          </p>
                          <div className="ml-4 flex items-center space-x-2">
                            {sim.target_achieved ? (
                              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                ✅ Achieved
                              </span>
                            ) : sim.target_achieved === false ? (
                              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                                ❌ Not Achieved
                              </span>
                            ) : (
                              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                                ❓ Unknown
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="grid grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-2 text-sm">
                          <div className="flex items-center space-x-2">
                            <span>⏳</span>
                            <span className="text-gray-700">
                              Timeline: {sim.timeframe || "N/A"}
                            </span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <span>💰</span>
                            <span className="text-gray-700">
                              Income: {sim.income_bracket || "N/A"}
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
                          {sim.target_value && (
                            <div className="flex items-center space-x-2">
                              <span>🎯</span>
                              <span className="text-gray-700">
                                Target: £{sim.target_value.toLocaleString()}
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (
                            window.confirm(
                              "Are you sure you want to delete this simulation?"
                            )
                          ) {
                            deleteSimulation(sim.id);
                          }
                        }}
                        className="ml-4 px-3 py-2 text-sm text-white bg-red-500 rounded-lg hover:bg-red-600 transition-colors duration-150 flex-shrink-0 font-medium"
                      >
                        Delete
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
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
