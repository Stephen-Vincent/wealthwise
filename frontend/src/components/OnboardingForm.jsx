import { useState, useEffect, useContext } from "react";
import { useNavigate } from "react-router-dom";
import PortfolioContext from "../context/PortfolioContext";
import logo from "../assets/wealthwise.png";
import { ArrowLeft, ArrowRight } from "lucide-react";
import ProgressDots from "./ProgressDots";
import LoadingScreen from "./LoadingScreen";

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
    target: "",
    lumpSum: "",
    monthly: "",
    timeframe: "",
    incomeBracket: "",
    consent: false,
  });

  const questions = [
    { key: "name", label: "What’s your name?" },
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
    if (!formData.name || !formData.goal) {
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

    let savedData;
    let formattedData = {};
    try {
      console.log("Parsed lumpSum:", lumpSum);
      console.log("Raw lumpSum from formData:", formData.lumpSum);
      formattedData = {
        name: formData.name,
        experience: parseInt(formData.experience, 10),
        goal: formData.goal,
        target_value: parseFloat(formData.target),
        timeframe: formData.timeframe,
        income_bracket: formData.incomeBracket,
        consent: formData.consent,
        lumpSum: lumpSum,
        monthly: monthly,
      };
      console.log(
        "📤 Full submission payload:",
        JSON.stringify(formattedData, null, 2)
      );

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

    try {
      const startTime = Date.now();
      setIsLoading(true);

      const response = await fetch("http://localhost:8000/simulate-portfolio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: savedData.id,
          name: formData.name,
          experience: parseInt(formData.experience, 10),
          goal: formData.goal,
          lumpSum: lumpSum,
          monthly: monthly,
          timeframe: formData.timeframe,
          income_bracket: formData.incomeBracket,
          risk: savedData.risk,
          risk_score: savedData.risk_score,
          selected_stocks: null,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Simulation response error text:", errorText);
        throw new Error("Simulation request failed.");
      }
      const result = await response.json();

      // Build payload for portfolioData context
      const payload = {
        name: formData.name,
        goal: formData.goal,
        target_value: parseFloat(formData.target),
        starting_balance: lumpSum,
        total_start: lumpSum,
        risk: result.risk,
        risk_score: result.risk_score,
        recommendations: result.recommendations,
        monthly_contribution: monthly,
        initial_investment: lumpSum,
        timeline: result.timeline,
        portfolio: result.portfolio,
      };

      Object.entries(result.portfolio).forEach(([ticker, data]) => {
        console.log(
          `- ${ticker}: Start £${data.start_price}, End £${data.end_price}, Growth: ${data.growth_pct}%, Final Value: £${data.final_value}`
        );
      });

      console.log("📅 Timeline Sample (First 5 Days):");
      console.log(result.timeline.slice(0, 5));

      setPortfolioData(payload);

      const elapsed = Date.now() - startTime;
      const remainingTime = Math.max(0, 3000 - elapsed);

      setTimeout(() => {
        // Ensure full 3-second loading screen duration
        setTimeout(() => {
          navigate("/dashboard");
          setIsLoading(false);
        }, 3000);
      }, remainingTime);
    } catch (error) {
      console.error("Simulation request failed", error);
      alert("Something went wrong. Please try again.");
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center text-center min-h-screen px-12 py-12 font-sans">
      {isLoading && <LoadingScreen />}
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

              {step === 4 && (
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

              {step === 5 && (
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

              {step === 6 && (
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
