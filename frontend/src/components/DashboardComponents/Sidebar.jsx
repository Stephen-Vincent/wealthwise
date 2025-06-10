import logo from "../../assets/wealthwise.png";
import { useNavigate } from "react-router-dom";

export default function Sidebar() {
  const navigate = useNavigate();
  const userId = localStorage.getItem("userId");

  const handleLogout = () => {
    localStorage.removeItem("userId");
    localStorage.removeItem("user_name");
    navigate("/login");
  };

  const startNew = () => {
    localStorage.removeItem("portfolioData");
    localStorage.removeItem("simulationId");
    localStorage.removeItem("simulationData");
    localStorage.removeItem("token");
    navigate(`/onboarding/${userId}`, { replace: true });
  };

  return (
    <aside className="w-1/6 p-4  shadow-md min-h-screen flex flex-col items-center">
      <div className="mb-6 mt-4">
        <img
          src={logo}
          alt="WealthWise Logo"
          className="w-24 h-24 mx-auto mb-2"
        />
        <div className="flex flex-col items-center mt-20">
          <button
            onClick={handleLogout}
            className="mb-4 bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded"
          >
            Logout
          </button>
          <button
            onClick={startNew}
            className="bg-[#00A8FF] text-white px-4 py-2 rounded font-bold w-full"
          >
            Start New Simulation
          </button>
          <button
            onClick={() => navigate("/simulations")}
            className="mt-4 bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded font-bold w-full"
          >
            View Simulations
          </button>
        </div>
      </div>
    </aside>
  );
}
