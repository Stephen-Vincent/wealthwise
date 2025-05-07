import { useState } from "react";
import { useNavigate } from "react-router-dom";
import logo from "../assets/wealthwise.png";
import { ArrowLeft, ArrowRight } from "lucide-react";
import ProgressDots from "./ProgressDots";

export default function OnboardingForm() {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
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

  const nextStep = () =>
    setStep((prev) => Math.min(prev + 1, questions.length));
  const prevStep = () => {
    if (step === 0) navigate("/");
    else setStep((prev) => Math.max(prev - 1, 0));
  };

  const handleSubmit = () => {
    if (!formData.consent)
      return alert("Please confirm this is a learning tool.");
    console.log(formData);
    navigate("/loading");
  };

  return (
    <div className="flex flex-col items-center text-center min-h-screen px-4 py-8 bg-gradient-to-b from-white to-[#EAF6FB] font-sans pt-32">
      <div className="mt-10 mb-8">
        <img
          src={logo}
          alt="WealthWise logo"
          className=" w-[250px] h-[250px] mx-auto"
        />
      </div>

      <div className="flex flex-col items-center justify-center min-h-[300px] space-y-6 w-full max-w-xl">
        {step < questions.length ? (
          <>
            <h2 className="text-xl font-semibold">{questions[step].label}</h2>

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
                  className="w-[600px] h-[70px] border border-gray-300 rounded-[15px] px-4 text-lg font-bold"
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
                    className="bg-[#00A8FF] text-white font-bold px-6 py-3 rounded-[15px] hover:brightness-110 transition"
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
                    className="bg-[#00A8FF] text-white font-bold px-6 py-3 rounded-[15px] hover:brightness-110 transition"
                    onClick={() => setFormData({ ...formData, risk: label })}
                  >
                    {label}
                  </button>
                ))}
              </div>
            )}

            {/* Progress bar */}
            <ProgressDots total={questions.length} current={step} />

            {/* Navigation Arrows */}
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
    </div>
  );
}
