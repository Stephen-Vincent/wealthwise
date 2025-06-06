import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import logo from "../assets/wealthwise.png"; // Adjust the path as necessary

export default function Simulations() {
  const navigate = useNavigate();
  const [simulations, setSimulations] = useState([]);
  const userId = localStorage.getItem("userId");

  useEffect(() => {
    const fetchSimulations = async () => {
      if (!userId) {
        console.log("userId is null, skipping fetch.");
        return;
      }
      try {
        const res = await fetch(
          `http://localhost:8000/users/${userId}/simulations`
        );
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
      }
    };

    console.log("Simulations useEffect userId:", userId);
    fetchSimulations();
  }, [userId]);

  const startNew = () => {
    navigate(`/onboarding/${userId}`);
  };

  const deleteSimulation = async (id) => {
    try {
      const res = await fetch(`http://localhost:8000/simulations/${id}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Failed to delete simulation");
      setSimulations(simulations.filter((sim) => sim.id !== id));
    } catch (err) {
      console.error("Error deleting simulation:", err);
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

        {simulations.length === 0 ? (
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
                  onClick={() => navigate(`/dashboard/${sim.id}`)}
                >
                  <p className="font-semibold">Simulation #{sim.id}</p>
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
                  <p>🔥 Risk: {sim.risk ?? "N/A"}</p>
                  <p>
                    📅 Created:{" "}
                    {sim.created_at
                      ? new Date(sim.created_at).toLocaleDateString()
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
