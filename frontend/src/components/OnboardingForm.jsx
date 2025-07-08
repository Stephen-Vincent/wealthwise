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

  // Complete list of questions/steps for onboarding
  const questions = [
    {
      key: "years_of_experience",
      label: "How many years of investing experience do you have?",
      type: "select",
      options: [...Array(31).keys()].map((year) => ({
        value: year,
        label: `${year} years`,
      })),
      placeholder: "Select years of experience",
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
      type: "select",
      options: [
        { value: "buy a house", label: "Buy a house" },
        { value: "vacation", label: "Vacation" },
        { value: "emergency fund", label: "Emergency fund" },
        { value: "retirement", label: "Retirement" },
        { value: "save for a car", label: "Save for a car" },
        { value: "wealth building", label: "General wealth building" },
      ],
      placeholder: "Select your goal",
    },
    {
      key: "target_amount",
      label: "What is your target investment value?",
      type: "input",
      inputType: "number",
      placeholder: "Target value e.g. 20000",
    },
    {
      key: "lump_sum_investment",
      label: "How much would you like to invest?",
      type: "dual_input",
      inputs: [
        {
          key: "lump_sum_investment",
          placeholder: "Lump sum amount",
          type: "number",
        },
        {
          key: "monthly_investment",
          placeholder: "Monthly amount",
          type: "number",
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
      type: "select",
      options: [
        { value: "low", label: "< £25,000" },
        { value: "medium", label: "£25,000 - £50,000" },
        { value: "high", label: "> £50,000" },
      ],
      placeholder: "Select your income bracket",
    },
  ];

  // Map human-friendly timeframe labels to integer years for backend
  const timeframeMap = {
    "Under 1 year": 1,
    "1–5 years": 5,
    "5–10 years": 10,
  };

  // Render different question types
  const renderQuestion = (question) => {
    switch (question.type) {
      case "select":
        return (
          <div className="flex justify-center items-center">
            <select
              className="w-[700px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
              value={formData[question.key]}
              onChange={(e) =>
                setFormData({ ...formData, [question.key]: e.target.value })
              }
              required
            >
              <option value="" disabled>
                {question.placeholder}
              </option>
              {question.options.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        );

      case "input":
        return (
          <div className="flex justify-center items-center">
            <input
              type={question.inputType || "text"}
              className="w-[700px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
              placeholder={question.placeholder}
              value={formData[question.key]}
              onChange={(e) =>
                setFormData({ ...formData, [question.key]: e.target.value })
              }
              required
            />
          </div>
        );

      case "dual_input":
        return (
          <div className="flex justify-center items-center space-x-4 w-full max-w-2xl">
            {question.inputs.map((input) => (
              <input
                key={input.key}
                type={input.type}
                step="0.01"
                className="w-[170px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                placeholder={input.placeholder}
                value={formData[input.key]}
                onChange={(e) =>
                  setFormData({ ...formData, [input.key]: e.target.value })
                }
              />
            ))}
          </div>
        );

      case "buttons":
        return (
          <div className="flex justify-center items-center gap-4 w-full max-w-4xl">
            <div className="flex flex-col gap-3 w-full">
              {question.options.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  className={`font-bold px-6 py-4 rounded-[15px] hover:brightness-110 transition text-left ${
                    formData[question.key] === option.value
                      ? "bg-white text-[#00A8FF] border-2 border-[#00A8FF]"
                      : "bg-[#00A8FF] text-white"
                  }`}
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
    if (!userId) {
      setIsLoading(false);
      alert("No user ID found. Please log in again.");
      return;
    }

    try {
      // Prepare payload for onboarding submission. Use backend field naming convention.
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
        timeframe: timeframeMap[formData.timeframe],
        income_bracket: formData.income,
        consent: formData.consent,
        name: userName || null,
        user_id: userId || null,
      };

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
        <div className="flex flex-col items-center justify-center min-h-[300px] w-full max-w-xl">
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
                {renderQuestion(questions[step])}
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
                  Let's Build Your Portfolio
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
