import { useState, useEffect, useContext } from "react";
import { useNavigate } from "react-router-dom";
import PortfolioContext from "../context/PortfolioContext";
import logo from "../assets/wealthwise.png";
import { ArrowLeft, ArrowRight } from "lucide-react";
import ProgressDots from "./ProgressDots";

export default function OnboardingForm() {
  const navigate = useNavigate();
  const { setPortfolioData } = useContext(PortfolioContext);
  const [step, setStep] = useState(0);
  const [fade, setFade] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    experience: "",
    goal: "",
    lumpSum: "",
    monthly: "",
    timeframe: "",
    risk: "",
    consent: false,
  });

  const questions = [
    { key: "name", label: "What’s your name?" },
    { key: "experience", label: "How much investing experience do you have?" },
    { key: "goal", label: "What is your main goal for investing?" },
    { key: "amount", label: "How much would you like to invest?" },
    {
      key: "timeframe",
      label: "What is your ideal time frame to reach your goal?",
    },
    {
      key: "risk",
      label:
        "How would you describe your risk level? (Cautious, Balanced, Adventurous)",
    },
  ];

  const nextStep = () => {
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
    if (!formData.name || !formData.goal || !formData.risk) {
      return alert("Please complete all questions.");
    }
    if (!formData.consent) {
      return alert("Please confirm this is a learning tool.");
    }

    let savedData;
    try {
      const formattedData = {
        ...formData,
        experience: parseInt(formData.experience, 10),
        lump_sum: formData.lumpSum,
        monthly: formData.monthly,
      };
      console.log("Submitting formData to backend:", formattedData);

      const saveResponse = await fetch("http://localhost:8000/onboarding", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formattedData),
      });

      if (!saveResponse.ok) {
        throw new Error("Failed to save onboarding data");
      }

      try {
        savedData = await saveResponse.json();
      } catch {
        throw new Error("Could not parse backend response.");
      }
      console.log("Saved user data:", savedData);
    } catch (error) {
      console.error("Saving onboarding data failed", error);
      alert("Something went wrong saving your data.");
      return;
    }

    const payload = { id: savedData.id };

    try {
      setIsLoading(true);
      const response = await fetch("http://localhost:8000/simulate-portfolio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await response.json();
      console.log("Portfolio Simulation Result:", result);
      console.log("✅ Portfolio Simulation Result:");
      console.log("Name:", formData.name);
      console.log("Goal:", formData.goal);
      console.log("Risk Profile:", formData.risk);
      console.log("Timeframe:", formData.timeframe);
      console.log("Initial Investment:", formData.lumpSum);
      console.log("Monthly Investment:", formData.monthly);

      console.log("📈 Final Balance:", result.final_balance);
      console.log("💰 Starting Value (invested):", result.total_start);
      console.log("📊 Total Value at End:", result.total_end);

      console.log("🪙 Portfolio Breakdown:");
      Object.entries(result.portfolio).forEach(([ticker, data]) => {
        console.log(
          `- ${ticker}: Start £${data.start_price}, End £${data.end_price}, Growth: ${data.growth_pct}%, Final Value: £${data.final_value}`
        );
      });

      console.log("📅 Timeline Sample (First 5 Days):");
      console.log(result.timeline.slice(0, 5));
      setPortfolioData({
        ...result,
        name: formData.name,
        goal: formData.goal,
      });
      navigate("/dashboard");
      setIsLoading(false);
    } catch (error) {
      console.error("Simulation request failed", error);
      alert("Something went wrong. Please try again.");
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center text-center min-h-screen px-12 py-12 font-sans">
      {isLoading && (
        <div className="fixed top-0 left-0 w-full h-full bg-white bg-opacity-80 flex items-center justify-center z-50">
          <div className="text-xl font-bold text-[#00A8FF] animate-pulse">
            Loading your portfolio...
          </div>
        </div>
      )}
      <div className="flex justify-center mb-8">
        <img
          src={logo}
          alt="WealthWise logo"
          className="w-[200px] h-[200px] object-contain"
        />
      </div>

      <div className="flex flex-col items-center justify-center min-h-[300px] space-y-6 w-full max-w-xl">
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
                <input
                  type="text"
                  className="w-[600px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                  placeholder="What's your name?"
                  value={formData.name}
                  onChange={(e) =>
                    setFormData({ ...formData, name: e.target.value })
                  }
                  required
                />
              )}

              {step === 1 && (
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

              {step === 2 && (
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

              {step === 3 && (
                <div className="flex space-x-4 justify-center">
                  <input
                    type="text"
                    className="w-[290px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
                    placeholder="Lump sum amount"
                    value={formData.lumpSum}
                    onChange={(e) =>
                      setFormData({ ...formData, lumpSum: e.target.value })
                    }
                  />
                  <input
                    type="text"
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
                <div className="flex justify-center space-x-4">
                  {["Cautious", "Balanced", "Adventurous"].map((label) => (
                    <button
                      key={label}
                      type="button"
                      className={`font-bold px-6 py-3 rounded-[15px] hover:brightness-110 transition ${
                        formData.risk === label
                          ? "bg-white text-[#00A8FF] border-2 border-[#00A8FF]"
                          : "bg-[#00A8FF] text-white"
                      }`}
                      onClick={() => setFormData({ ...formData, risk: label })}
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
    </div>
  );
}
