import { useEffect, useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import logo from "../assets/wealthwise.png"; // Adjust the path as necessary

import { PortfolioContext } from "../context/PortfolioContext"; // Add this import

export default function Simulations() {
  const navigate = useNavigate();
  const [simulations, setSimulations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [userId, setUserId] = useState(localStorage.getItem("userId"));
  const { setPortfolioData } = useContext(PortfolioContext); // Add this context hook
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
  // Function to load a simulation by ID, update context, and navigate to /loading
  const loadSimulationAndNavigate = async (id) => {
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
      // Merge all nested results into a single object, with preference for simData.results, then simData
      const fullPortfolio = {
        ...simData,
        ...(simData.results || {}),
        timeline: simData.results?.timeline || simData.timeline || [],
        portfolio: simData.results?.portfolio || simData.portfolio || [],
        stocks_picked:
          simData.results?.stocks_picked || simData.stocks_picked || [],
        return: simData.results?.return ?? simData.return ?? null,
        starting_value:
          simData.results?.starting_value ?? simData.starting_value ?? null,
        end_value: simData.results?.end_value ?? simData.end_value ?? null,
      };
      setPortfolioData(fullPortfolio);

      // Store the full user object in localStorage (user_name)
      const storedUser = JSON.parse(localStorage.getItem("user"));
      if (storedUser) {
        localStorage.setItem("user_name", storedUser.name);
      }

      // ✅ Store IDs for LoadingScreen.jsx to use
      localStorage.setItem("simulationId", id);
      localStorage.setItem("userId", userId);

      navigate("/loading");
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
        if (data.length === 0) {
          navigate("/onboarding");
          return;
        }
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

  const startNew = () => {
    navigate(`/onboarding/${userId}`);
  };

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
    <div className="flex flex-col items-center">
      <div className="flex justify-center mb-6">
        <img
          src={logo}
          alt="WealthWise logo"
          className="w-[200px] h-[200px] object-contain"
        />
      </div>
      <div className="p-8 w-full max-w-xl">
        <h2 className="text-2xl font-bold mb-4 text-center">
          Your Simulations
        </h2>

        {loading ? (
          <p className="text-center">Loading simulations...</p>
        ) : error ? (
          <p className="text-center text-red-500">{error}</p>
        ) : simulations.length === 0 ? (
          <p className="text-center">No previous simulations found.</p>
        ) : (
          <ul className="mb-6">
            {simulations.map((sim) => (
              <li
                key={sim.id}
                className="p-4 border rounded mb-2 hover:bg-gray-50 flex justify-between items-center"
              >
                <div
                  className="cursor-pointer w-full"
                  onClick={() => {
                    console.log("🔍 Simulation clicked with ID:", sim.id);
                    loadSimulationAndNavigate(sim.id);
                  }}
                >
                  {/* <p className="font-semibold">Simulation #{sim.id}</p> */}
                  <p>🎯 Goal: {sim.goal || "N/A"}</p>
                  <p>⏳ Timeline: {sim.timeframe || "N/A"}</p>
                  <p>
                    ✅ Target Achieved:{" "}
                    {sim.target_achieved === true
                      ? "Yes"
                      : sim.target_achieved === false
                      ? "No"
                      : "N/A"}
                  </p>
                  <p>💰 Income: {sim.income_bracket || "N/A"}</p>
                  <p>⚖️ Risk Score: {sim.risk_score ?? "N/A"}</p>
                  <p>🔥 Risk: {sim.risk_label ?? "N/A"}</p>
                  <p>
                    📅 Created:{" "}
                    {sim.created_at
                      ? new Date(sim.created_at).toLocaleDateString("en-GB", {
                          year: "numeric",
                          month: "short",
                          day: "numeric",
                        })
                      : "Date unknown"}
                  </p>
                </div>
                <button
                  onClick={() => deleteSimulation(sim.id)}
                  className="ml-4 px-2 py-1 text-sm text-white bg-red-500 rounded"
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}

        <button
          onClick={startNew}
          className="bg-[#00A8FF] text-white px-4 py-2 rounded font-bold w-full"
        >
          Start New Simulation
        </button>
      </div>
    </div>
  );
}
