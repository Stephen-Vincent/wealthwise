import { useNavigate, Outlet, useLocation } from "react-router-dom";
import axios from "axios";
import { useState } from "react";
import logo from "../assets/wealthwise.png";

export default function LandingScreen() {
  const navigate = useNavigate();
  const location = useLocation();
  const showWelcome = location.pathname === "/";

  const [simulations, setSimulations] = useState([]);

  return (
    <div className="flex flex-col items-center pt-12 pb-12 min-h-screen font-sans">
      <div className="flex flex-col items-center justify-center min-h-[300px] w-full max-w-xl text-center gap-6">
        <div className="flex justify-center">
          <img
            src={logo}
            alt="WealthWise logo"
            className="w-[200px] h-[200px] object-contain"
          />
        </div>

        {showWelcome ? (
          <div className="flex flex-col items-center justify-center space-y-4 mt-20 fade-in">
            <h1 className="text-2xl font-bold text-[#333]">
              Welcome to WealthWise
            </h1>
            <p className="text-[#666] text-center">
              Your AI-powered investing simulator
            </p>

            <div className="flex gap-4 mt-6">
              <button
                onClick={() => navigate("/signup")}
                className="bg-[#00A8FF] text-white font-bold px-6 py-3 rounded-[15px] border-none cursor-pointer"
              >
                Sign Up
              </button>
              <button
                onClick={() => navigate("/login")}
                className="bg-white text-[#00A8FF] font-bold px-6 py-3 rounded-[15px] border-2 border-[#00A8FF] cursor-pointer"
              >
                Log In
              </button>
            </div>
          </div>
        ) : (
          <div className="fade-in w-full max-w-sm mt-6">
            <Outlet />
          </div>
        )}
      </div>

      {/* Optionally display simulations */}
      {simulations.length > 0 && (
        <div className="mt-6 max-w-xl w-full text-left bg-gray-50 p-4 rounded shadow">
          <h3 className="font-semibold mb-2">Simulations:</h3>
          <ul className="list-disc list-inside">
            {simulations.map((sim) => (
              <li key={sim.id}>
                {sim.name || "Unnamed simulation"} — Target: {sim.target_value}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
