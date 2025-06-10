import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const messages = [
  "🧠 Calculating your risk score...",
  "📊 Putting together your portfolio...",
  "🔍 Gathering market insights...",
];

export default function LoadingScreen() {
  const navigate = useNavigate();
  const userId = localStorage.getItem("userId");
  const rawSimulationId = localStorage.getItem("simulationId");
  const simulationId =
    rawSimulationId && rawSimulationId !== "null" ? rawSimulationId : null;
  const [index, setIndex] = useState(0);
  const [fade, setFade] = useState(true);

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

  useEffect(() => {
    if (userId && simulationId) {
      console.log("✅ Redirecting to dashboard with:", {
        userId,
        simulationId,
      });
      localStorage.setItem("simulationCompleted", "true"); // ✅ Mark simulation as complete
      navigate(`/dashboard/${userId}/${simulationId}`);
    } else {
      console.warn("❌ Missing userId or simulationId in localStorage:", {
        userId,
        simulationId,
      });
    }
  }, []);

  return (
    <div className="flex flex-col items-center py-12 text-xl text-center h-screen">
      <p
        className={`text-lg transition-opacity duration-500 ease-in-out mt-32 ${
          fade ? "opacity-100" : "opacity-0"
        }`}
      >
        {messages[index]}
      </p>
    </div>
  );
}
