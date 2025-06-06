import Simulations from "./components/Simulations";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingScreen from "./components/LandingScreen";
import OnboardingForm from "./components/OnboardingForm";
import LoadingScreen from "./components/LoadingScreen";
import Dashboard from "./components/Dashboard";
import StockPage from "./components/StockPage";
import Signup from "./components/SignUp";
import Login from "./components/Login";
import { PortfolioProvider } from "./context/PortfolioContext"; // ✅ import the provider

export default function App() {
  return (
    <PortfolioProvider>
      <div className="min-h-screen w-full bg-gradient-to-b from-white to-[#a3cde0] font-sans">
        <Router>
          <Routes>
            <Route path="/" element={<LandingScreen />}>
              <Route index element={<Login />} />
              <Route path="signup" element={<Signup />} />
              <Route path="login" element={<Login />} />
              <Route path="onboarding" element={<OnboardingForm />} />
              <Route path="onboarding/:id" element={<OnboardingForm />} />
              <Route path="loading" element={<LoadingScreen />} />
            </Route>

            <Route path="/dashboard/:id" element={<Dashboard />} />
            <Route path="/stock/:stock" element={<StockPage />} />
            <Route path="/simulations" element={<Simulations />} />
          </Routes>
        </Router>
      </div>
    </PortfolioProvider>
  );
}
