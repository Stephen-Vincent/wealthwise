import Simulations from "./components/Simulations";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingScreen from "./components/LandingScreen";
import OnboardingForm from "./components/OnboardingForm";
import LoadingScreen from "./components/LoadingScreen";
import Dashboard from "./components/Dashboard";
import StockPage from "./components/StockPage";
import Signup from "./components/Signup";
import Login from "./components/Login";
import { PortfolioProvider } from "./context/PortfolioContext"; // ✅ import the provider

export default function App() {
  return (
    <Router>
      <PortfolioProvider>
        <div className="min-h-screen w-full bg-gradient-to-b from-white to-[#a3cde0] font-sans">
          <Routes>
            <Route path="/" element={<LandingScreen />}>
              <Route index element={<Login />} />
              <Route path="signup" element={<Signup />} />
              <Route path="login" element={<Login />} />
            </Route>

            {/* Move loading to top level so it can be accessed from anywhere */}
            <Route path="/loading" element={<LoadingScreen />} />

            <Route
              path="/dashboard/:userId/:simulationId"
              element={<Dashboard />}
            />
            <Route path="/stock/:stock" element={<StockPage />} />
            <Route path="/simulations" element={<Simulations />} />
          </Routes>
        </div>
      </PortfolioProvider>
    </Router>
  );
}
