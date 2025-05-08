import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingScreen from "./components/LandingScreen";
import OnboardingForm from "./components/OnboardingForm";
import LoadingScreen from "./components/LoadingScreen";
import Dashboard from "./components/Dashboard";
import StockPage from "./components/StockPage";
import { PortfolioProvider } from "./context/PortfolioContext"; // ✅ import the provider

export default function App() {
  return (
    <PortfolioProvider>
      {" "}
      {/* ✅ wrap everything in the provider */}
      <div className="min-h-screen w-full bg-gradient-to-b from-white to-[#a3cde0] font-sans">
        <Router>
          <Routes>
            <Route path="/" element={<LandingScreen />} />
            <Route path="/onboarding" element={<OnboardingForm />} />
            <Route path="/loading" element={<LoadingScreen />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/stock/:stock" element={<StockPage />} />
          </Routes>
        </Router>
      </div>
    </PortfolioProvider>
  );
}
