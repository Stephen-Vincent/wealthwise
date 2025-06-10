// OnboardingForm captures user input (name, goal, experience, lump sum, etc.)
// and sends it to the backend to generate a simulation.
// The backend returns a simulation object, which includes a calculated risk score
// and is stored in localStorage and context for later use.
/**
 * OnboardingForm.jsx
 *
 * This component handles the onboarding flow for new users, collecting
 * their investment preferences, experience, and goals through a multi-step form.
 *
 * What the component does:
 * - Guides users through a series of questions to gather data required to simulate
 *   a personalized investment portfolio.
 * - Validates user inputs at each step.
 * - Submits the collected data to the backend API and fetches the resulting
 *   simulated portfolio data.
 * - Updates the global PortfolioContext with the simulation results for use
 *   throughout the dashboard.
 *
 * What data is expected in the form:
 * - years_of_experience: Number of years the user has been investing (integer).
 * - investment_goal: User's main investment goal (string, e.g. "buy a house").
 * - target_amount: The target value to reach with investments (number).
 * - lump_sum_investment: Optional lump sum amount to invest (number).
 * - monthly_investment: Optional monthly contribution amount (number).
 * - timeframe: Desired investment timeframe, selected from options (string, mapped to integer).
 * - income: User's income bracket (string: "low", "medium", "high").
 * - consent: Confirmation that the user understands this is a learning tool (boolean).
 * - name/user_id: User's name and ID, fetched from localStorage.
 *
 * What happens when the form is submitted:
 * - The form validates that all required fields are filled and consent is given.
 * - The form data is transformed into a payload matching backend expectations.
 * - A POST request is sent to the backend /onboarding endpoint with this data.
 * - On success, the returned onboarding submission ID is used to fetch portfolio simulation data.
 * - The simulation data is stored in PortfolioContext and the user is navigated to the loading screen.
 *
 * What data is sent to the backend:
 * - Payload structure:
 *   {
 *     years_of_experience: number,
 *     goal: string,
 *     target_value: number,
 *     lump_sum: number|null,
 *     monthly: number|null,
 *     timeframe: number,
 *     income_bracket: string,
 *     consent: boolean,
 *     name: string|null,
 *     user_id: number|null
 *   }
 *
 * How the response is handled and passed to PortfolioContext:
 * - The backend responds with the onboarding submission record (including an ID).
 * - The component then fetches the simulated portfolio data for this submission.
 * - The simulation data is set in PortfolioContext using setPortfolioData, making it available globally.
 * - The user is then navigated to the loading screen (and then to the dashboard).
 */
import { useState, useEffect, useContext } from "react";
import { usePortfolio } from "../context/PortfolioContext";
import { useNavigate } from "react-router-dom";
import PortfolioContext from "../context/PortfolioContext";
import logo from "../assets/wealthwise.png";
import { ArrowLeft, ArrowRight } from "lucide-react";
import ProgressDots from "./ProgressDots";
import LoadingScreen from "./LoadingScreen";
import axios from "axios";

// Main onboarding form component
export default function OnboardingForm() {
  const navigate = useNavigate();
  const { setPortfolioData } = usePortfolio();
  const [step, setStep] = useState(0);
  const [fade, setFade] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  // Local state to capture all onboarding form fields
  const [formData, setFormData] = useState({
    years_of_experience: "",
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

  // On mount: fetch user info from localStorage and show greeting
  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("user"));
    const storedName = user?.name;
    console.log("Login check: user_name from localStorage:", storedName);
    if (storedName) {
      setUserName(storedName);
    }
    const storedId = user?.id;
    console.log("Fetched userId from localStorage:", storedId);
    if (!storedId) {
      alert("No user ID found. Please log in again.");
      return;
    }
    const timeout = setTimeout(() => {
      setShowGreeting(false);
    }, 2000);
    return () => clearTimeout(timeout);
  }, []);

  // List of questions/steps for onboarding
  const questions = [
    {
      key: "years_of_experience",
      label: "How much investing experience do you have?",
    },
    { key: "investment_goal", label: "What is your main goal for investing?" },
    { key: "target_amount", label: "What is your target investment value?" },
    { key: "lump_sum_investment", label: "How much would you like to invest?" },
    {
      key: "timeframe",
      label: "What is your ideal time frame to reach your goal?",
    },
    {
      key: "income",
      label: "Which income bracket best represents your household?",
    },
  ];

  // Map human-friendly timeframe labels to integer years for backend
  const timeframeMap = {
    "Under 1 year": 1,
    "1–5 years": 5,
    "5–10 years": 10,
  };

  // Advance to next onboarding step, with validation
  const nextStep = () => {
    const currentKey = questions[step]?.key;
    const currentValue = formData[currentKey];

    // Special validation for lump_sum_investment/monthly_investment step
    if (
      currentKey === "lump_sum_investment" &&
      !formData.lump_sum_investment &&
      !formData.monthly_investment
    ) {
      return alert("Please enter a lump sum or a monthly contribution.");
    }

    if (!currentValue && currentKey !== "lump_sum_investment") {
      return alert("Please answer this question before continuing.");
    }

    setFade(false);
    setTimeout(() => {
      setStep((prev) => Math.min(prev + 1, questions.length));
      setFade(true);
    }, 500);
  };

  // Go back to previous onboarding step
  const prevStep = () => {
    if (step === 0) navigate("/");
    else {
      setFade(false);
      setTimeout(() => {
        setStep((prev) => Math.max(prev - 1, 0));
        setFade(true);
      }, 500);
    }
  };

  /**
   * Handles form submission at the end of the onboarding steps.
   * Validates required fields, prepares payload, and sends data to backend.
   * On success, fetches simulation data and updates PortfolioContext.
   */
  const handleSubmit = async () => {
    if (
      !formData.years_of_experience ||
      !formData.investment_goal ||
      !formData.target_amount ||
      (!formData.lump_sum_investment && !formData.monthly_investment) ||
      !formData.timeframe ||
      !formData.income
    ) {
      return alert("Please complete all questions.");
    }
    if (!formData.consent) {
      return alert("Please confirm this is a learning tool.");
    }

    // Show loading and scroll to top immediately after consent check passes
    window.scrollTo({ top: 0, behavior: "smooth" });
    setIsLoading(true);

    console.log(
      "Raw lump sum input (before parse):",
      formData.lump_sum_investment
    );
    const lump_sum = parseFloat(formData.lump_sum_investment || 0) || 0;
    const monthly_contribution =
      parseFloat(formData.monthly_investment || 0) || 0;

    // Compute variables, but do not block based on feasibility
    let timeframeInYears;
    switch (formData.timeframe) {
      case "Under 1 year":
        timeframeInYears = 1;
        break;
      case "1–5 years":
        timeframeInYears = 5;
        break;
      case "5–10 years":
        timeframeInYears = 10;
        break;
      default:
        timeframeInYears = 5; // fallback to a reasonable default
    }
    const portfolioAmount =
      lump_sum + monthly_contribution * 12 * timeframeInYears;

    if (lump_sum === 0 && monthly_contribution === 0) {
      setIsLoading(false);
      return alert("Please enter a lump sum or a monthly contribution.");
    }

    const user = JSON.parse(localStorage.getItem("user"));
    const userId = user?.id;
    console.log("Fetched userId from localStorage (submit):", userId);
    if (!userId) {
      setIsLoading(false);
      alert("No user ID found. Please log in again.");
      return;
    }

    try {
      // Payload shape:
      // {
      //   name: string,
      //   years_of_experience: number,
      //   goal: string,
      //   target_value: number,
      //   lump_sum: number,
      //   monthly: number,
      //   risk: string,             // Initial input (not used by backend as final value)
      //   investment_style: string
      // }
      // Prepare payload for onboarding submission. Use backend field naming convention.
      const payload = {
        years_of_experience: parseInt(formData.years_of_experience),
        goal: formData.investment_goal,
        target_value: parseFloat(formData.target_amount),
        lump_sum: formData.lump_sum_investment
          ? parseFloat(formData.lump_sum_investment)
          : null,
        monthly: formData.monthly_investment
          ? parseFloat(formData.monthly_investment)
          : null,
        timeframe: timeframeMap[formData.timeframe],
        income_bracket: formData.income,
        consent: formData.consent,
        name: userName || null,
        user_id: userId || null,
        // risk and investment_style could be added here if needed for future expansion
      };
      console.log("📤 Sending payload:", payload);

      // Send onboarding data to backend API (POST /onboarding/)
      const accessToken = localStorage.getItem("access_token");
      const config = {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      };
      // Use the simulation result directly from the onboarding POST response
      const response = await axios.post(
        "http://localhost:8000/onboarding/",
        payload,
        config
      );
      // The backend returns the full simulation including the final calculated risk
      const simulationData = response.data;
      // Save simulation to context using usePortfolio hook
      setPortfolioData(simulationData);
      // Immediately store simulationId and userId in localStorage after response
      // Use simulation.risk for any downstream logic (do not use local risk input)
      const simulationId = simulationData?.id;
      const storedUserId = localStorage.getItem("userId") || userId;

      if (simulationId && storedUserId) {
        localStorage.setItem("simulationId", simulationId);
        localStorage.setItem("userId", storedUserId);
        console.log("✅ Stored simulationId and userId in localStorage:", {
          simulationId,
          userId: storedUserId,
        });
        // If you need the risk value, always use simulationData.risk hereafter
        navigate("/loading");
      } else {
        console.error(
          "❌ Missing simulationId or userId in response/localStorage",
          {
            simulation: simulationData,
            userId: storedUserId,
          }
        );
      }
    } catch (error) {
      console.error("Saving onboarding data failed", error);
      alert("Something went wrong saving your data.");
      return;
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center text-center px-12 py-12 font-sans">
      {isLoading && <LoadingScreen />}

      {showGreeting ? (
        <h2 className="text-2xl font-bold">Hello {userName || "there"} 👋</h2>
      ) : (
        <div className="flex flex-col items-center justify-center min-h-[300px]  w-full max-w-xl">
          <div
            className={`transition-opacity duration-500 w-full ${
              fade ? "opacity-100" : "opacity-0"
            }`}
          >
            {step < questions.length ? (
              <>
                <h2 className="text-xl font-semibold mb-4">
                  {questions[step].label}
                </h2>

                {step === 0 && (
                  <div className="flex justify-center items-center space-x-2">
                    <select
                      className="w-[300px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                      value={formData.years_of_experience}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          years_of_experience: e.target.value,
                        })
                      }
                      required
                    >
                      <option value="" disabled>
                        Select years of experience
                      </option>
                      {[...Array(31).keys()].map((year) => (
                        <option key={year} value={year}>
                          {year}
                        </option>
                      ))}
                    </select>
                    <span className="text-lg font-bold">years</span>
                  </div>
                )}

                {step === 1 && (
                  <div className="flex justify-center items-center">
                    <select
                      name="goal"
                      className="w-[700px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                      value={formData.investment_goal}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          investment_goal: e.target.value,
                        })
                      }
                      required
                    >
                      <option value="">Select your goal</option>
                      <option value="buy a house">Buy a house</option>
                      <option value="vacation">Vacation</option>
                      <option value="emergency fund">Emergency fund</option>
                      <option value="retirement">Retirement</option>
                      <option value="save for a car">Save for a car</option>
                    </select>
                  </div>
                )}

                {step === 2 && (
                  <div className="flex justify-center items-center">
                    <input
                      type="text"
                      className="w-[700px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                      placeholder="Target value e.g. 20000"
                      value={formData.target_amount}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          target_amount: e.target.value,
                        })
                      }
                      required
                    />
                  </div>
                )}

                {step === 3 && (
                  <div className="flex justify-center items-center space-x-4 w-full max-w-2xl">
                    <input
                      type="number"
                      step="0.01"
                      className="w-[170px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                      placeholder="Lump sum amount"
                      value={formData.lump_sum_investment}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          lump_sum_investment: e.target.value,
                        })
                      }
                    />
                    <input
                      type="number"
                      step="0.01"
                      className="w-[170px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                      placeholder="Monthly amount"
                      value={formData.monthly_investment}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          monthly_investment: e.target.value,
                        })
                      }
                    />
                  </div>
                )}

                {step === 4 && (
                  <div className="flex justify-center items-center gap-8  w-full max-w-3xl">
                    {["Under 1 year", "1–5 years", "5–10 years"].map(
                      (label) => (
                        <button
                          key={label}
                          type="button"
                          className={`font-bold min-w-[160px] px-6 py-3 rounded-[15px] hover:brightness-110 transition ${
                            formData.timeframe === label
                              ? "bg-white text-[#00A8FF] border-2 border-[#00A8FF]"
                              : "bg-[#00A8FF] text-white"
                          }`}
                          onClick={() =>
                            setFormData({ ...formData, timeframe: label })
                          }
                        >
                          {label}
                        </button>
                      )
                    )}
                  </div>
                )}

                {step === 5 && (
                  <div className="flex justify-center items-center gap-8 w-full max-w-3xl">
                    <select
                      name="income_bracket"
                      className="w-[400px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                      value={formData.income}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          income: e.target.value,
                        })
                      }
                      required
                    >
                      <option value="">Select your income bracket</option>
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </select>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center">
                <h2 className="text-xl font-semibold mb-4">Declaration</h2>
                <p className="text-gray-600 text-sm mb-4 max-w-md mx-auto">
                  Thank you for sharing your details! We're excited to simulate
                  a portfolio tailored to your goals. Please confirm that you
                  understand this tool is for learning purposes only and not
                  financial advice.
                </p>
                <label className="flex items-center gap-2 text-sm justify-center text-black">
                  <input
                    type="checkbox"
                    checked={formData.consent}
                    onChange={(e) =>
                      setFormData({ ...formData, consent: e.target.checked })
                    }
                    required
                  />
                  I agree this is a learning tool and not financial advice.
                </label>
                <button
                  className="mt-4 bg-[#00A8FF] text-white font-bold px-6 py-3 rounded-[15px] hover:brightness-110 transition"
                  onClick={handleSubmit}
                >
                  Let’s Build Your Portfolio
                </button>
              </div>
            )}
          </div>
          {/* Progress bar */}
          {step < questions.length && (
            <ProgressDots total={questions.length} current={step} />
          )}
          {/* Navigation Arrows */}
          {step < questions.length && (
            <div className="flex justify-center items-center space-x-10 mt-12">
              <button
                onClick={prevStep}
                className="bg-[#00A8FF] text-white w-12 h-12 rounded-full flex items-center justify-center p-0 m-0 hover:brightness-110 transition"
              >
                <ArrowLeft className="text-white w-6 h-6" />
              </button>

              <button
                onClick={nextStep}
                className="bg-[#00A8FF] text-white w-12 h-12 rounded-full flex items-center justify-center p-0 m-0 hover:brightness-110 transition"
              >
                <ArrowRight className="text-white w-6 h-6" />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
