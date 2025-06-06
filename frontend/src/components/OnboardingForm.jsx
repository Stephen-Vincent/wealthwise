import { useState, useEffect, useContext } from "react";
import { useNavigate } from "react-router-dom";
import PortfolioContext from "../context/PortfolioContext";
import logo from "../assets/wealthwise.png";
import { ArrowLeft, ArrowRight } from "lucide-react";
import ProgressDots from "./ProgressDots";
import LoadingScreen from "./LoadingScreen";
import axios from "axios";

export default function OnboardingForm() {
  const navigate = useNavigate();
  const { setPortfolioData } = useContext(PortfolioContext);
  const [step, setStep] = useState(0);
  const [fade, setFade] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState({
    experience: "",
    goal: "",
    target: "",
    lumpSum: "",
    monthly: "",
    timeframe: "",
    incomeBracket: "",
    consent: false,
  });
  const [showGreeting, setShowGreeting] = useState(true);
  const [userName, setUserName] = useState("");

  useEffect(() => {
    const storedName = localStorage.getItem("user_name");
    console.log("Login check: user_name from localStorage:", storedName);
    console.log("Fetched username from localStorage:", storedName);
    if (storedName) {
      setUserName(storedName);
    }
    const storedId = localStorage.getItem("userId");
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

  const questions = [
    { key: "experience", label: "How much investing experience do you have?" },
    { key: "goal", label: "What is your main goal for investing?" },
    { key: "target", label: "What is your target investment value?" },
    { key: "lumpSum", label: "How much would you like to invest?" },
    {
      key: "timeframe",
      label: "What is your ideal time frame to reach your goal?",
    },
    {
      key: "incomeBracket",
      label: "Which income bracket best represents your household?",
    },
  ];

  const nextStep = () => {
    const currentKey = questions[step]?.key;
    const currentValue = formData[currentKey];

    // Special validation for lumpSum/monthly step
    if (currentKey === "lumpSum" && !formData.lumpSum && !formData.monthly) {
      return alert("Please enter a lump sum or a monthly contribution.");
    }

    if (!currentValue && currentKey !== "lumpSum") {
      return alert("Please answer this question before continuing.");
    }

    setFade(false);
    setTimeout(() => {
      setStep((prev) => Math.min(prev + 1, questions.length));
      setFade(true);
    }, 500);
  };
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

  const handleSubmit = async () => {
    if (
      !formData.experience ||
      !formData.goal ||
      !formData.target ||
      (!formData.lumpSum && !formData.monthly) ||
      !formData.timeframe ||
      !formData.incomeBracket
    ) {
      return alert("Please complete all questions.");
    }
    if (!formData.consent) {
      return alert("Please confirm this is a learning tool.");
    }

    console.log("Raw lump sum input (before parse):", formData.lumpSum);
    const lumpSum = parseFloat(formData.lumpSum || 0) || 0;
    const monthly = parseFloat(formData.monthly || 0) || 0;

    if (lumpSum === 0 && monthly === 0) {
      return alert("Please enter a lump sum or a monthly contribution.");
    }

    const userId = localStorage.getItem("userId");
    console.log("Fetched userId from localStorage (submit):", userId);
    if (!userId) {
      alert("No user ID found. Please log in again.");
      return;
    }

    let savedUserData;
    let onboardingSubmissionId;
    try {
      console.log("Parsed lumpSum:", lumpSum);
      console.log("Raw lumpSum from formData:", formData.lumpSum);
      // Log the target value being sent for onboarding
      console.log(
        "✅ target_value:",
        formData.target,
        "| Full payload preview:",
        {
          ...formData,
          target_value: parseFloat(formData.target),
          user_id: localStorage.getItem("userId"),
          name: localStorage.getItem("user_name"),
        }
      );
      const fullPayload = {
        user_id: parseInt(userId, 10),
        name: localStorage.getItem("user_name"),
        experience: Number(formData.experience),
        goal: formData.goal,
        lumpSum: parseFloat(formData.lumpSum),
        monthly: parseFloat(formData.monthly),
        timeframe: formData.timeframe,
        consent: formData.consent,
        income_bracket: formData.incomeBracket,
        target_value: parseFloat(formData.target),
        risk: null,
        risk_score: null,
        target_achieved: false,
      };
      console.log(
        "📤 Full submission payload:",
        JSON.stringify(fullPayload, null, 2)
      );

      // Use axios to POST onboarding data to FastAPI (update to match CORS origin)
      const onboardingResponse = await axios.post(
        "http://localhost:5000/onboarding",
        fullPayload
      );
      savedUserData = onboardingResponse.data;
      onboardingSubmissionId = savedUserData.id; // ✅ Get the new onboarding_submission ID
      console.log("Saved user data:", savedUserData);
      console.log("Onboarding submission ID:", onboardingSubmissionId);
    } catch (error) {
      console.error("Saving onboarding data failed", error);
      alert("Something went wrong saving your data.");
      return;
    }
  };

  return (
    <div className="flex flex-col items-center text-center min-h-screen px-12 py-12 font-sans">
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
                  <div className="flex items-center justify-center space-x-2">
                    <select
                      className="w-[250px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                      value={formData.experience}
                      onChange={(e) =>
                        setFormData({ ...formData, experience: e.target.value })
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
                  <input
                    type="text"
                    className="w-[600px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                    placeholder="Your investment goal"
                    value={formData.goal}
                    onChange={(e) =>
                      setFormData({ ...formData, goal: e.target.value })
                    }
                    required
                  />
                )}

                {step === 2 && (
                  <input
                    type="text"
                    className="w-[600px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                    placeholder="Target value e.g. 20000"
                    value={formData.target}
                    onChange={(e) =>
                      setFormData({ ...formData, target: e.target.value })
                    }
                    required
                  />
                )}

                {step === 3 && (
                  <div className="flex space-x-4 justify-center">
                    <input
                      type="number"
                      step="0.01"
                      className="w-[290px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                      placeholder="Lump sum amount"
                      value={formData.lumpSum}
                      onChange={(e) =>
                        setFormData({ ...formData, lumpSum: e.target.value })
                      }
                    />
                    <input
                      type="number"
                      step="0.01"
                      className="w-[290px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                      placeholder="Monthly amount"
                      value={formData.monthly}
                      onChange={(e) =>
                        setFormData({ ...formData, monthly: e.target.value })
                      }
                    />
                  </div>
                )}

                {step === 4 && (
                  <div className="flex justify-center space-x-4">
                    {["< 1 year", "1–5 years", "5+ years"].map((label) => (
                      <button
                        key={label}
                        type="button"
                        className={`font-bold px-6 py-3 rounded-[15px] hover:brightness-110 transition ${
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
                    ))}
                  </div>
                )}

                {step === 5 && (
                  <div className="flex justify-center space-x-4 flex-wrap max-w-[600px]">
                    {[
                      "Low (under £30k)",
                      "Medium (£30k–£70k)",
                      "High (over £70k)",
                    ].map((label) => (
                      <button
                        key={label}
                        type="button"
                        className={`font-bold px-6 py-3 rounded-[15px] m-2 hover:brightness-110 transition ${
                          formData.incomeBracket === label
                            ? "bg-white text-[#00A8FF] border-2 border-[#00A8FF]"
                            : "bg-[#00A8FF] text-white"
                        }`}
                        onClick={() =>
                          setFormData({ ...formData, incomeBracket: label })
                        }
                      >
                        {label}
                      </button>
                    ))}
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
