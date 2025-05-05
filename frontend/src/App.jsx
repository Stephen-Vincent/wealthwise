import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingScreen from "./components/LandingScreen";
import OnboardingForm from "./components/OnboardingForm";
import LoadingScreen from "./components/LoadingScreen";
import Dashboard from "./components/Dashboard";

export default function App() {
  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-white to-secondary font-poppins">
      <Router>
        <Routes>
          <Route path="/" element={<LandingScreen />} />
          <Route path="/onboarding" element={<OnboardingForm />} />
          <Route path="/loading" element={<LoadingScreen />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </Router>
    </div>
  );
}
