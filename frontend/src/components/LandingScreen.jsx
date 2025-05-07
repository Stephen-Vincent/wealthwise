import { useNavigate } from "react-router-dom";
import logo from "../assets/wealthwise.png";

export default function LandingScreen() {
  const navigate = useNavigate();

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

        <div className="flex flex-col items-center justify-center space-y-2 mt-32">
          <h1 className="text-xl font-bold text-[#333]">
            Welcome to WealthWise
          </h1>
          <p className="text-[#666]">Your AI-powered investing simulator</p>

          <button
            onClick={() => navigate("/onboarding")}
            className="bg-[#00A8FF] text-white font-bold px-6 py-3 rounded-[15px] border-none cursor-pointer"
          >
            Get Started
          </button>
        </div>
      </div>
    </div>
  );
}
