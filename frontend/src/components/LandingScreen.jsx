import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import logo from "../assets/wealthwise.png";
import Login from "./Login";
import Signup from "./Signup";
import WelcomeScreen from "./WelcomeScreen";
import OnboardingForm from "./OnboardingForm";
import LoadingScreen from "./LoadingScreen";
import Simulations from "./Simulations";

export default function LandingScreen() {
  const location = useLocation();
  const navigate = useNavigate();
  const [panel, setPanel] = useState(null); // 'login', 'signup', 'welcome', 'onboarding', 'loading', 'simulations', or null
  const [logoMovedUp, setLogoMovedUp] = useState(false);
  const [showWelcomeContent, setShowWelcomeContent] = useState(true);
  const [animatingPanel, setAnimatingPanel] = useState(false);
  const [panelVisible, setPanelVisible] = useState(false);
  const [nextPanel, setNextPanel] = useState(null);
  const [userName, setUserName] = useState(null);

  // Check if user is already logged in on component mount
  useEffect(() => {
    const checkUserLogin = () => {
      const token = localStorage.getItem("access_token");
      const user = JSON.parse(localStorage.getItem("user"));

      if (token && user) {
        // User is logged in, show welcome screen automatically
        setUserName(user.name || null);
        setShowWelcomeContent(false);
        setLogoMovedUp(true);

        // Check if we need to show a specific panel based on navigation state
        const targetPanel = location.state?.showPanel;
        if (targetPanel === "simulations") {
          setPanel("simulations");
        } else {
          setPanel("welcome");
        }
        setPanelVisible(true);
      }
    };

    checkUserLogin();
  }, [location.state]);

  // Handles showing any panel with animation
  const handleShowPanel = (which) => {
    if (animatingPanel) return;
    setAnimatingPanel(true);
    setShowWelcomeContent(false);
    setLogoMovedUp(true);
    setNextPanel(which);

    setTimeout(() => {
      setPanel(nextPanel);
      setPanelVisible(true);
      setAnimatingPanel(false);
    }, 700);
  };

  // Fix: If nextPanel is set and panel isn't, open it after animation
  if (nextPanel && !panel && !animatingPanel) {
    setTimeout(() => {
      setPanel(nextPanel);
      setPanelVisible(true);
      setNextPanel(null);
    }, 0);
  }

  // When closing the panel, logo returns down and welcome fades in
  const handleBackToWelcome = () => {
    setPanelVisible(false);
    setLogoMovedUp(false);
    setUserName(null);
    setTimeout(() => {
      setPanel(null);
      setShowWelcomeContent(true);
    }, 300);
  };

  // Function to show welcome screen after login (called from Login component)
  const handleShowWelcomeScreen = (user) => {
    setUserName(user?.name || null);
    setPanelVisible(false);
    setTimeout(() => handleShowPanel("welcome"), 300);
  };

  // Function to handle forgot password navigation
  const handleShowForgotPassword = () => {
    navigate("/forgot-password");
  };

  // Function to show onboarding (called from WelcomeScreen)
  const handleShowOnboarding = () => {
    setPanelVisible(false);
    setTimeout(() => handleShowPanel("onboarding"), 300);
  };

  // Function to go back to welcome from onboarding
  const handleBackToWelcomeFromOnboarding = () => {
    setPanelVisible(false);
    setTimeout(() => handleShowPanel("welcome"), 300);
  };

  // Function to show loading (called from OnboardingForm)
  const handleShowLoading = () => {
    setPanelVisible(false);
    setTimeout(() => handleShowPanel("loading"), 300);
  };

  // Function to show simulations (called from WelcomeScreen)
  const handleShowSimulations = () => {
    setPanelVisible(false);
    setTimeout(() => handleShowPanel("simulations"), 300);
  };

  // Function to go back to welcome from simulations
  const handleBackToWelcomeFromSimulations = () => {
    setPanelVisible(false);
    setTimeout(() => handleShowPanel("welcome"), 300);
  };

  const logoSizeClasses = "w-[120px] h-[120px]";

  return (
    <div className="min-h-screen flex items-center justify-center flex-col font-sans">
      <div className="flex flex-col items-center text-center gap-6 max-w-4xl w-full relative px-4">
        {/* Fixed positioned logo container to prevent layout shifts */}
        <div className="relative w-full flex justify-center">
          <div
            className={`transition-all duration-700 ease-in-out ${
              logoMovedUp
                ? "transform -translate-y-[50px]"
                : "transform translate-y-0"
            }`}
            style={{ willChange: "transform" }}
          >
            <img
              src={logo}
              alt="WealthWise logo"
              className={`object-contain ${logoSizeClasses}`}
            />
          </div>
        </div>

        {/* Content container with fixed height to prevent jumps */}
        <div className="w-full min-h-[400px] flex flex-col items-center justify-start">
          {/* Welcome content - only show if no panel is active and user is not logged in */}
          {!panel && (
            <div
              className={`flex flex-col items-center justify-center space-y-4 transition-all duration-500 ease-in-out absolute top-48 w-full ${
                showWelcomeContent
                  ? "opacity-100 translate-y-0 pointer-events-auto"
                  : "opacity-0 translate-y-6 pointer-events-none"
              }`}
            >
              <h1 className="text-2xl font-bold text-[#333]">
                Welcome to WealthWise
              </h1>
              <p className="text-[#666] text-center">
                Your AI-powered investing simulator
              </p>
              <div className="flex gap-4 mt-6">
                <button
                  onClick={() => handleShowPanel("signup")}
                  disabled={!showWelcomeContent || animatingPanel}
                  className="bg-[#00A8FF] text-white font-bold px-6 py-3 rounded-[15px] border-none cursor-pointer shadow hover:scale-105 active:scale-95 active:shadow-inner transition-all duration-150"
                >
                  Sign Up
                </button>
                <button
                  onClick={() => handleShowPanel("login")}
                  disabled={!showWelcomeContent || animatingPanel}
                  className="bg-white text-[#00A8FF] font-bold px-6 py-3 rounded-[15px] border-2 border-[#00A8FF] cursor-pointer shadow hover:scale-105 active:scale-95 active:shadow-inner transition-all duration-150"
                >
                  Log In
                </button>
              </div>
            </div>
          )}

          {/* Login panel with smooth entrance */}
          {panel === "login" && (
            <div
              className={`w-full absolute top-24 transition-all duration-500 ease-in-out ${
                panelVisible
                  ? "opacity-100 translate-y-0 pointer-events-auto"
                  : "opacity-0 translate-y-6 pointer-events-none"
              }`}
            >
              <Login
                onBack={handleBackToWelcome}
                onShowSignup={() => {
                  setPanelVisible(false);
                  setTimeout(() => handleShowPanel("signup"), 300);
                }}
                onShowWelcomeScreen={handleShowWelcomeScreen}
                onShowForgotPassword={handleShowForgotPassword}
              />
            </div>
          )}

          {/* Signup panel with smooth entrance */}
          {panel === "signup" && (
            <div
              className={`w-full absolute top-24 transition-all duration-500 ease-in-out ${
                panelVisible
                  ? "opacity-100 translate-y-0 pointer-events-auto"
                  : "opacity-0 translate-y-6 pointer-events-none"
              }`}
            >
              <Signup
                onBack={handleBackToWelcome}
                onShowLogin={() => {
                  setPanelVisible(false);
                  setTimeout(() => handleShowPanel("login"), 300);
                }}
              />
            </div>
          )}

          {/* Welcome screen for logged-in users */}
          {panel === "welcome" && (
            <div
              className={`w-full absolute top-24 transition-all duration-500 ease-in-out ${
                panelVisible
                  ? "opacity-100 translate-y-0 pointer-events-auto"
                  : "opacity-0 translate-y-6 pointer-events-none"
              }`}
            >
              <WelcomeScreen
                onBack={handleBackToWelcome}
                userName={userName}
                onShowOnboarding={handleShowOnboarding}
                onShowSimulations={handleShowSimulations}
              />
            </div>
          )}

          {/* Onboarding panel with smooth entrance */}
          {panel === "onboarding" && (
            <div
              className={`w-full absolute top-0 transition-all duration-500 ease-in-out ${
                panelVisible
                  ? "opacity-100 translate-y-0 pointer-events-auto"
                  : "opacity-0 translate-y-6 pointer-events-none"
              }`}
            >
              <OnboardingForm
                onBack={handleBackToWelcomeFromOnboarding}
                onShowLoading={handleShowLoading}
              />
            </div>
          )}

          {/* Loading panel with smooth entrance */}
          {panel === "loading" && (
            <div
              className={`w-full absolute top-0 transition-all duration-500 ease-in-out ${
                panelVisible
                  ? "opacity-100 translate-y-0 pointer-events-auto"
                  : "opacity-0 translate-y-6 pointer-events-none"
              }`}
            >
              <LoadingScreen />
            </div>
          )}

          {/* Simulations panel with smooth entrance */}
          {panel === "simulations" && (
            <div
              className={`w-full absolute top-0 transition-all duration-500 ease-in-out ${
                panelVisible
                  ? "opacity-100 translate-y-0 pointer-events-auto"
                  : "opacity-0 translate-y-6 pointer-events-none"
              }`}
            >
              <Simulations
                onBack={handleBackToWelcomeFromSimulations}
                onShowLoading={handleShowLoading}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
