import Simulations from "./components/Simulations";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingScreen from "./components/LandingScreen";
import LoadingScreen from "./components/LoadingScreen";
import Dashboard from "./components/Dashboard";
import Signup from "./components/Signup";
import Login from "./components/Login";
import { PortfolioProvider } from "./context/PortfolioContext";
import { useEffect } from "react";

export default function App() {
  // Set body background to match for seamless scrolling
  // useEffect(() => {
  //   // Apply background to body for full-page coverage
  //   document.body.style.background =
  //     "linear-gradient(to bottom, white, #a3cde0)";
  //   document.body.style.minHeight = "100vh";
  //   document.documentElement.style.minHeight = "100vh";

  //   // Cleanup on unmount
  //   return () => {
  //     document.body.style.background = "";
  //     document.body.style.minHeight = "";
  //     document.documentElement.style.minHeight = "";
  //   };
  // }, []);

  return (
    <Router>
      <PortfolioProvider>
        {/* 
          Use min-h-screen to ensure minimum viewport height
          Background is now applied to body for seamless scrolling
        */}
        <div className="min-h-screen w-full font-sans mt-6">
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

            <Route path="/simulations" element={<Simulations />} />
          </Routes>
        </div>
      </PortfolioProvider>
    </Router>
  );
}
