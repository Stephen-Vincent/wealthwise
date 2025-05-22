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

    // Navigate after all messages are shown + 1s buffer
    const finalTimeout = setTimeout(() => {
      navigate("/dashboard");
    }, messages.length * 1000 + 1000);
    timeouts.push(finalTimeout);

    return () => timeouts.forEach(clearTimeout);
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
