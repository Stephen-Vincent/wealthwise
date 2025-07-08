import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const messages = [
  "🧠 Calculating your risk score...",
  "📊 Putting together your portfolio...",
  "🔍 Gathering market insights...",
  "✅ Finalizing your personalized dashboard...",
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
    const interval = setInterval(() => {
      const completed = localStorage.getItem("simulationCompleted");
      const userIdCheck = localStorage.getItem("userId");
      const simulationIdCheck = localStorage.getItem("simulationId");

      if (
        completed === "true" &&
        userIdCheck &&
        simulationIdCheck &&
        simulationIdCheck !== "null"
      ) {
        clearInterval(interval);
        navigate(`/dashboard/${userIdCheck}/${simulationIdCheck}`);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

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
