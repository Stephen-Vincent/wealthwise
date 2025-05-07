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
  const [fade, setFade] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setFade(false);
      setTimeout(() => {
        setIndex((prev) => {
          const next = prev + 1;
          if (next === messages.length) {
            clearInterval(interval);
            setTimeout(() => navigate("/dashboard"), 1000);
            return prev;
          } else {
            setFade(true);
            return next;
          }
        });
      }, 500);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center py-12 text-xl text-center h-screen">
      <img
        src={logo}
        alt="WealthWise logo"
        className="w-[200px] h-[200px] object-contain mb-6"
      />
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
