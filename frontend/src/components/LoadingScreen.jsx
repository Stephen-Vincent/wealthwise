import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import logo from "../assets/wealthwise.png"; // Replace with your real logo path

const messages = [
  "🧠 Calculating your risk score...",
  "📊 Putting together your portfolio...",
  "🔍 Gathering market insights...",
];

export default function LoadingScreen() {
  const navigate = useNavigate();
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => {
        if (prev === messages.length - 1) {
          clearInterval(interval);
          setTimeout(() => navigate("/dashboard"), 1000);
        }
        return prev + 1;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col justify-center items-center h-screen text-xl text-center">
      <img
        src={logo}
        alt="WealthWise logo"
        className="w-24 h-24 animate-pulse mb-6"
      />
      <p className="transition-opacity duration-700 ease-in-out">
        {messages[index]}
      </p>
    </div>
  );
}
