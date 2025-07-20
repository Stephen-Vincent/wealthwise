import React, { useState, useEffect } from "react";
import { createPortal } from "react-dom";
import { ArrowLeft, ArrowRight, AlertCircle } from "lucide-react";
import ProgressDots from "../components/CustomComponents/ProgressDots";
import { usePortfolio } from "../context/PortfolioContext";

// Main onboarding form component
const OnboardingForm = ({ onBack, onShowLoading }) => {
  const [step, setStep] = useState(0);
  const [fade, setFade] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [showError, setShowError] = useState(false);

  // Local state to capture all onboarding form fields
  const [formData, setFormData] = useState({
    years_of_experience: "",
    loss_tolerance: "",
    panic_behavior: "",
    financial_behavior: "",
    engagement_level: "",
    investment_goal: "",
    target_amount: "",
    lump_sum_investment: "",
    monthly_investment: "",
    timeframe: "",
    income: "",
    consent: false,
  });
  const [showGreeting, setShowGreeting] = useState(true);
  const [userName, setUserName] = useState("");

  const { setPortfolioData } = usePortfolio();

  // On mount: fetch user info and show greeting with fade-in
  useEffect(() => {
    const fadeInTimer = setTimeout(() => {
      setIsVisible(true);
    }, 100);

    // Get real user data from localStorage or API
    try {
      const user = JSON.parse(localStorage.getItem("user") || "{}");
      const storedName = user?.name;

      if (storedName) {
        setUserName(storedName);
      } else {
        setUserName("John Doe");
      }
    } catch (error) {
      console.error("Error parsing user data:", error);
      setUserName("User");
    }

    const greetingTimer = setTimeout(() => {
      setShowGreeting(false);
    }, 2000);

    return () => {
      clearTimeout(fadeInTimer);
      clearTimeout(greetingTimer);
    };
  }, []);

  // Auto-hide error messages after 4 seconds
  useEffect(() => {
    if (showError) {
      const timer = setTimeout(() => {
        setShowError(false);
        setErrorMessage("");
      }, 4000);
      return () => clearTimeout(timer);
    }
  }, [showError]);

  // Function to show error message with animation
  const showErrorMessage = (message) => {
    setErrorMessage(message);
    setShowError(true);
  };

  // Complete list of questions/steps for onboarding
  const questions = [
    {
      key: "years_of_experience",
      label: "How many years of investing experience do you have?",
      type: "input",
      inputType: "number",
      placeholder: "e.g., 5",
      min: 0,
      max: 50,
      helpText: "Enter 0 if you're a complete beginner",
    },
    {
      key: "loss_tolerance",
      label: "How would you react if your investment dropped by 20% in a week?",
      type: "buttons",
      options: [
        {
          value: "sell_immediately",
          label: "Sell immediately to prevent further losses",
        },
        { value: "wait_and_see", label: "Wait and see what happens" },
        { value: "buy_more", label: "Buy more while prices are low" },
      ],
    },
    {
      key: "panic_behavior",
      label: "Have you ever sold investments during a market crash?",
      type: "buttons",
      options: [
        { value: "yes_always", label: "Yes, I always sell when markets crash" },
        { value: "yes_sometimes", label: "Yes, but only sometimes" },
        { value: "no_never", label: "No, I hold through market downturns" },
        {
          value: "no_experience",
          label: "I haven't experienced a major crash",
        },
      ],
    },
    {
      key: "financial_behavior",
      label: "What would you do with an unexpected £1,000 bonus?",
      type: "buttons",
      options: [
        { value: "invest_all", label: "Invest all of it" },
        { value: "save_half", label: "Save half, invest half" },
        { value: "save_all", label: "Save all of it" },
        { value: "spend_it", label: "Spend it on something I want" },
      ],
    },
    {
      key: "engagement_level",
      label: "How often do you review your investments?",
      type: "buttons",
      options: [
        { value: "daily", label: "Daily" },
        { value: "weekly", label: "Weekly" },
        { value: "monthly", label: "Monthly" },
        { value: "quarterly", label: "Quarterly" },
        { value: "rarely", label: "Rarely or never" },
      ],
    },
    {
      key: "investment_goal",
      label: "What is your main goal for investing?",
      type: "buttons",
      options: [
        { value: "buy a house", label: "Buy a house" },
        { value: "vacation", label: "Vacation" },
        { value: "emergency fund", label: "Emergency fund" },
        { value: "retirement", label: "Retirement" },
        { value: "save for a car", label: "Save for a car" },
        { value: "wealth building", label: "General wealth building" },
      ],
    },
    {
      key: "target_amount",
      label: "What is your target investment value?",
      type: "input",
      inputType: "number",
      placeholder: "20000",
      currency: true,
    },
    {
      key: "lump_sum_investment",
      label: "How much would you like to invest?",
      type: "dual_input",
      inputs: [
        {
          key: "lump_sum_investment",
          placeholder: "5000",
          type: "number",
          currency: true,
          label: "Lump Sum",
        },
        {
          key: "monthly_investment",
          placeholder: "500",
          type: "number",
          currency: true,
          label: "Monthly",
        },
      ],
    },
    {
      key: "timeframe",
      label: "What is your ideal time frame to reach your goal?",
      type: "buttons",
      options: [
        { value: "Under 1 year", label: "Under 1 year" },
        { value: "1–5 years", label: "1–5 years" },
        { value: "5–10 years", label: "5–10 years" },
      ],
    },
    {
      key: "income",
      label: "Which income bracket best represents your household?",
      type: "buttons",
      options: [
        { value: "low", label: "< £25,000" },
        { value: "medium", label: "£25,000 - £50,000" },
        { value: "high", label: "> £50,000" },
      ],
    },
  ];

  // Toast Notification Component - Using Portal for true screen positioning
  const ToastNotification = ({ message, visible }) => {
    if (typeof document === "undefined") return null; // SSR safety

    return createPortal(
      <div
        className={`fixed top-6 right-6 z-[9999] transition-all duration-500 ease-in-out transform ${
          visible
            ? "opacity-100 translate-x-0 scale-100"
            : "opacity-0 translate-x-full scale-95 pointer-events-none"
        }`}
        style={{ position: "fixed", zIndex: 9999 }} // Inline styles for extra specificity
      >
        <div className="bg-white border-l-4 border-red-400 rounded-lg shadow-lg p-4 max-w-sm min-w-[300px]">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <AlertCircle className="h-5 w-5 text-red-400 mt-0.5" />
            </div>
            <div className="ml-3 flex-1">
              <p className="text-sm font-medium text-gray-900">
                Validation Error
              </p>
              <p className="text-sm text-gray-600 mt-1">{message}</p>
            </div>
            <div className="ml-4 flex-shrink-0">
              <button
                onClick={() => {
                  setShowError(false);
                  setErrorMessage("");
                }}
                className="inline-flex text-gray-400 hover:text-gray-600 focus:outline-none transition-colors duration-200"
              >
                <span className="sr-only">Close</span>
                <svg
                  className="h-4 w-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          </div>

          {/* Progress bar for auto-dismiss */}
          <div className="mt-3 bg-gray-200 rounded-full h-1 overflow-hidden">
            <div
              className="h-full bg-red-400 rounded-full transition-all duration-[4000ms] ease-linear"
              style={{
                width: visible ? "0%" : "100%",
                transitionDelay: visible ? "0ms" : "0ms",
              }}
            />
          </div>
        </div>
      </div>,
      document.body // Portal to document body for true screen positioning
    );
  };

  // Render different question types with enhanced styling
  const renderQuestion = (question) => {
    const baseInputClasses =
      "w-full h-[70px] border-2 border-gray-300 rounded-[15px] px-4 text-lg font-bold transition-all duration-300 hover:border-[#00A8FF] focus:border-[#00A8FF] focus:ring-2 focus:ring-[#00A8FF]/20 focus:outline-none bg-white";

    switch (question.type) {
      case "input":
        return (
          <div
            className="flex flex-col justify-center items-center animate-fade-in-up"
            style={{ animationDelay: "0.2s" }}
          >
            <div className="relative max-w-[700px] w-full">
              {question.currency && (
                <span className="absolute left-4 top-1/2 transform -translate-y-1/2 text-lg font-bold text-gray-600 pointer-events-none z-10">
                  £
                </span>
              )}
              <input
                type={question.inputType || "text"}
                min={question.min}
                max={question.max}
                className={`${baseInputClasses} ${
                  question.currency ? "pl-8" : ""
                }`}
                placeholder={question.placeholder}
                value={formData[question.key]}
                onChange={(e) =>
                  setFormData({ ...formData, [question.key]: e.target.value })
                }
                required
              />
            </div>
            {question.helpText && (
              <p
                className="text-sm text-gray-500 mt-3 animate-fade-in-up"
                style={{ animationDelay: "0.4s" }}
              >
                {question.helpText}
              </p>
            )}
          </div>
        );

      case "dual_input":
        return (
          <div
            className="flex justify-center items-center animate-fade-in-up"
            style={{ animationDelay: "0.2s" }}
          >
            <div className="flex flex-col sm:flex-row gap-6 w-full max-w-3xl">
              {question.inputs.map((input, index) => (
                <div key={input.key} className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-3 text-left">
                    {input.label}
                  </label>
                  <div className="relative">
                    {input.currency && (
                      <span className="absolute left-4 top-1/2 transform -translate-y-1/2 text-lg font-bold text-gray-600 pointer-events-none z-10">
                        £
                      </span>
                    )}
                    <input
                      type={input.type}
                      step="0.01"
                      className={`w-full h-[70px] border-2 border-gray-300 rounded-[15px] text-lg font-bold transition-all duration-300 hover:border-[#00A8FF] focus:border-[#00A8FF] focus:ring-2 focus:ring-[#00A8FF]/20 focus:outline-none bg-white animate-fade-in-up ${
                        input.currency ? "pl-8 pr-4" : "px-4"
                      }`}
                      style={{ animationDelay: `${0.3 + index * 0.1}s` }}
                      placeholder={input.placeholder}
                      value={formData[input.key]}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          [input.key]: e.target.value,
                        })
                      }
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      case "buttons":
        return (
          <div className="flex justify-center items-center gap-4 w-full max-w-4xl">
            <div className="flex flex-col gap-3 w-full">
              {question.options.map((option, index) => (
                <button
                  key={option.value}
                  type="button"
                  className={`font-bold px-6 py-4 rounded-[15px] transition-all duration-300 text-left transform hover:scale-105 hover:shadow-lg animate-fade-in-up ${
                    formData[question.key] === option.value
                      ? "bg-white text-[#00A8FF] border-2 border-[#00A8FF] shadow-md scale-105"
                      : "bg-[#00A8FF] text-white hover:brightness-110"
                  }`}
                  style={{ animationDelay: `${0.2 + index * 0.1}s` }}
                  onClick={() =>
                    setFormData({ ...formData, [question.key]: option.value })
                  }
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  // Advance to next onboarding step, with validation
  const nextStep = () => {
    const currentKey = questions[step]?.key;
    const currentValue = formData[currentKey];

    // Special validation for years of experience (number input)
    if (currentKey === "years_of_experience") {
      const experience = parseInt(formData.years_of_experience);
      if (isNaN(experience) || experience < 0 || experience > 50) {
        showErrorMessage("Please enter a valid number of years (0-50).");
        return;
      }
    }

    // Special validation for lump_sum_investment/monthly_investment step
    if (
      currentKey === "lump_sum_investment" &&
      !formData.lump_sum_investment &&
      !formData.monthly_investment
    ) {
      showErrorMessage("Please enter a lump sum or a monthly contribution.");
      return;
    }

    // Special validation for target amount
    if (currentKey === "target_amount") {
      const target = parseFloat(formData.target_amount);
      if (isNaN(target) || target <= 0) {
        showErrorMessage("Please enter a valid target amount.");
        return;
      }
    }

    if (!currentValue && currentKey !== "lump_sum_investment") {
      showErrorMessage("Please answer this question before continuing.");
      return;
    }

    setFade(false);
    setTimeout(() => {
      setStep((prev) => Math.min(prev + 1, questions.length));
      setFade(true);
    }, 300);
  };

  // Go back to previous onboarding step
  const prevStep = () => {
    if (step === 0) {
      if (onBack) {
        onBack();
      } else {
        console.log("No onBack callback provided");
      }
    } else {
      setFade(false);
      setTimeout(() => {
        setStep((prev) => Math.max(prev - 1, 0));
        setFade(true);
      }, 300);
    }
  };

  // Handle form submission
  const handleSubmit = async () => {
    // Validation
    if (
      !formData.years_of_experience ||
      !formData.investment_goal ||
      !formData.target_amount ||
      (!formData.lump_sum_investment && !formData.monthly_investment) ||
      !formData.timeframe ||
      !formData.income
    ) {
      showErrorMessage("Please complete all questions.");
      return;
    }
    if (!formData.consent) {
      showErrorMessage("Please confirm this is a learning tool.");
      return;
    }

    console.log("📝 Form validation passed. Submitting data:", formData);

    // Get real user data for submission
    try {
      const user = JSON.parse(localStorage.getItem("user") || "{}");
      const userId = user?.id;
      const userName = user?.name;

      if (!userId) {
        console.error("❌ No user ID found. User must log in again.");
        showErrorMessage("Session expired. Please log in again.");
        return;
      }

      console.log("✅ User data found:", { userId, userName });

      // Prepare real payload for API
      const payload = {
        years_of_experience: parseInt(formData.years_of_experience),
        loss_tolerance: formData.loss_tolerance,
        panic_behavior: formData.panic_behavior,
        financial_behavior: formData.financial_behavior,
        engagement_level: formData.engagement_level,
        goal: formData.investment_goal,
        target_value: parseFloat(formData.target_amount),
        lump_sum: formData.lump_sum_investment
          ? parseFloat(formData.lump_sum_investment)
          : null,
        monthly: formData.monthly_investment
          ? parseFloat(formData.monthly_investment)
          : null,
        timeframe:
          formData.timeframe === "Under 1 year"
            ? 1
            : formData.timeframe === "1–5 years"
            ? 5
            : 10,
        income_bracket: formData.income,
        consent: formData.consent,
        name: userName || null,
        user_id: userId,
      };

      console.log("📤 Submitting payload to backend:", payload);

      // 🚀 NAVIGATE TO LOADING SCREEN IMMEDIATELY
      localStorage.setItem("userId", userId);
      localStorage.setItem("isCreatingPortfolio", "true");

      // Scroll to top immediately
      window.scrollTo({ top: 0, behavior: "smooth" });

      // Show loading screen right away
      if (onShowLoading) {
        console.log("🔄 Transitioning to LoadingScreen immediately");
        onShowLoading();
      }

      // MAKE THE API CALL IN THE BACKGROUND
      try {
        const accessToken = localStorage.getItem("access_token");

        if (!accessToken) {
          throw new Error("No access token found");
        }

        console.log("🔄 Making API call to create simulation in background...");

        const response = await fetch("http://localhost:8000/onboarding/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
          },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          throw new Error(
            `API call failed: ${response.status} ${response.statusText}`
          );
        }

        const simulationData = await response.json();
        console.log("✅ Simulation created successfully:", simulationData);

        // Store ALL the simulation data
        const simulationId = simulationData?.id;

        if (simulationId) {
          localStorage.setItem("simulationId", simulationId);
          localStorage.setItem("portfolioData", JSON.stringify(simulationData));
          localStorage.removeItem("isCreatingPortfolio");

          // UPDATE THE PORTFOLIO CONTEXT
          setPortfolioData(simulationData);

          console.log("💾 Stored simulation data:", {
            simulationId,
            userId,
            fullData: "stored in portfolioData key",
            contextUpdated: true,
          });
        } else {
          throw new Error("Invalid simulation response - missing ID");
        }
      } catch (apiError) {
        console.error("❌ API call failed:", apiError);
        localStorage.removeItem("isCreatingPortfolio");
        showErrorMessage(
          `Failed to create portfolio: ${apiError.message}. Please try again.`
        );
      }
    } catch (error) {
      console.error("❌ Error processing form submission:", error);
      localStorage.removeItem("isCreatingPortfolio");
      showErrorMessage("An error occurred. Please try again.");
    }
  };

  return (
    <div
      className={`flex flex-col items-center justify-center text-center px-12 py-12 font-sans mt-12 transition-all duration-1000 ${
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
      }`}
    >
      {/* Toast Notification Component - Portal ensures true screen positioning */}
      <ToastNotification message={errorMessage} visible={showError} />

      {showGreeting ? (
        <div className="animate-fade-in-scale">
          <h2 className="text-3xl font-bold text-[#00A8FF] mb-4">
            {userName ? `Hello ${userName}!` : "Hello!"}
          </h2>
          <p className="text-gray-600 text-lg">
            Let's build your investment portfolio...
          </p>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center min-h-[500px] w-full max-w-4xl">
          <div
            className={`transition-all duration-500 w-full transform ${
              fade ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
            }`}
          >
            {step < questions.length ? (
              <div className="space-y-8">
                <h2 className="text-2xl font-semibold mb-8 text-gray-800 animate-fade-in-down">
                  {questions[step].label}
                </h2>
                <div className="mb-16">{renderQuestion(questions[step])}</div>
              </div>
            ) : (
              <div className="text-center animate-fade-in-scale">
                <h2 className="text-2xl font-semibold mb-6 text-gray-800">
                  Declaration
                </h2>
                <div
                  className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6 animate-fade-in-up"
                  style={{ animationDelay: "0.2s" }}
                >
                  <p className="text-gray-700 text-base mb-4 leading-relaxed">
                    Thank you for sharing your details! We're excited to
                    simulate a portfolio tailored to your goals. Please confirm
                    that you understand this tool is for learning purposes only
                    and not financial advice.
                  </p>
                </div>
                <label
                  className="flex items-center gap-3 text-base justify-center text-gray-800 mb-6 animate-fade-in-up"
                  style={{ animationDelay: "0.4s" }}
                >
                  <input
                    type="checkbox"
                    className="w-5 h-5 text-[#00A8FF] border-2 border-gray-300 rounded focus:ring-[#00A8FF] focus:ring-2 transition-all duration-200"
                    checked={formData.consent}
                    onChange={(e) =>
                      setFormData({ ...formData, consent: e.target.checked })
                    }
                    required
                  />
                  I agree this is a learning tool and not financial advice.
                </label>

                {/* Declaration Screen Navigation */}
                <div
                  className="flex justify-center items-center gap-6 animate-fade-in-up"
                  style={{ animationDelay: "0.6s" }}
                >
                  {/* Back Button */}
                  <button
                    onClick={prevStep}
                    className="bg-gray-500 text-white font-bold px-6 py-4 rounded-[15px] hover:bg-gray-600 transition-all duration-300 transform hover:scale-105 hover:shadow-lg flex items-center gap-2"
                    disabled={isLoading}
                  >
                    <ArrowLeft className="w-5 h-5" />
                    Back
                  </button>

                  {/* Submit Button */}
                  <button
                    className="bg-[#00A8FF] text-white font-bold px-8 py-4 rounded-[15px] hover:brightness-110 transition-all duration-300 transform hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                    onClick={handleSubmit}
                    disabled={isLoading}
                  >
                    {isLoading
                      ? "Creating Portfolio..."
                      : "Let's Build Your Portfolio"}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Progress bar with animation */}
          {step < questions.length && (
            <div
              className="animate-fade-in-up mt-8"
              style={{ animationDelay: "0.5s" }}
            >
              <ProgressDots total={questions.length} current={step} />
            </div>
          )}

          {/* Navigation Arrows with animation */}
          {step < questions.length && (
            <div
              className="flex justify-center items-center space-x-10 mt-8 animate-fade-in-up"
              style={{ animationDelay: "0.7s" }}
            >
              <button
                onClick={prevStep}
                className="bg-[#00A8FF] text-white w-14 h-14 rounded-full flex items-center justify-center hover:brightness-110 transition-all duration-300 transform hover:scale-110 hover:shadow-lg"
                disabled={isLoading}
              >
                <ArrowLeft className="text-white w-6 h-6" />
              </button>

              <button
                onClick={nextStep}
                className="bg-[#00A8FF] text-white w-14 h-14 rounded-full flex items-center justify-center hover:brightness-110 transition-all duration-300 transform hover:scale-110 hover:shadow-lg"
                disabled={isLoading}
              >
                <ArrowRight className="text-white w-6 h-6" />
              </button>
            </div>
          )}
        </div>
      )}

      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes fadeInDown {
          from {
            opacity: 0;
            transform: translateY(-20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes fadeInScale {
          from {
            opacity: 0;
            transform: scale(0.95);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }

        .animate-fade-in-up {
          animation: fadeInUp 0.6s ease-out forwards;
          opacity: 0;
        }

        .animate-fade-in-down {
          animation: fadeInDown 0.6s ease-out forwards;
          opacity: 0;
        }

        .animate-fade-in-scale {
          animation: fadeInScale 0.8s ease-out forwards;
          opacity: 0;
        }
      `}</style>
    </div>
  );
};

export default OnboardingForm;
