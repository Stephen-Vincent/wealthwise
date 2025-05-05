import { useState } from "react";
import { useNavigate } from "react-router-dom";
import logo from "../assets/wealthwise.png"; // Replace with your real logo path
import { ArrowLeft, ArrowRight } from "lucide-react"; // Optional icon library

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

  const handleChange = (e) => {
    setFormData({ ...formData, [questions[step].key]: e.target.value });
  };

  const nextStep = () =>
    setStep((prev) => Math.min(prev + 1, questions.length));
  const prevStep = () => {
    if (step === 0) {
      navigate("/");
    } else {
      setStep((prev) => Math.max(prev - 1, 0));
    }
  };

  const handleSubmit = () => {
    if (!formData.consent)
      return alert("Please confirm this is a learning tool.");
    console.log(formData);
    navigate("/loading");
  };

  return (
    <div className="p-6 max-w-md mx-auto text-center space-y-6">
      <img
        src={logo}
        alt="WealthWise logo"
        className="w-32 h-32 mb-6 mx-auto"
      />
      {step < questions.length ? (
        <>
          <h2 className="text-xl font-semibold">{questions[step].label}</h2>
          {step === 0 && (
            <div className="flex flex-col gap-5 items-center">
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
            </div>
          )}
          {step === 1 && (
            <div className="flex flex-col gap-5 items-center">
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
            </div>
          )}
          {step === 2 && (
            <div className="flex flex-col gap-5 items-center">
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
            </div>
          )}
          {step === 3 && (
            <div className="flex flex-col gap-5 items-center">
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
            </div>
          )}
          {step === 4 && (
            <div className="flex flex-col gap-5 items-center">
              <div className="flex justify-center space-x-4">
                {["< 1 year", "1–5 years", "5+ years"].map((label) => (
                  <button
                    key={label}
                    type="button"
                    className="bg-[#00A8FF] text-white font-bold px-6 py-3 rounded-[15px] border-none hover:brightness-110 transition"
                    onClick={() =>
                      setFormData({ ...formData, timeframe: label })
                    }
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )}
          {step === 5 && (
            <div className="flex flex-col gap-5 items-center">
              <div className="flex justify-center space-x-4">
                {["Cautious", "Balanced", "Adventurous"].map((label) => (
                  <button
                    key={label}
                    type="button"
                    className="bg-[#00A8FF] text-white font-bold px-6 py-3 rounded-[15px] border-none hover:brightness-110 transition"
                    onClick={() => setFormData({ ...formData, risk: label })}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="w-[500px] mx-auto flex justify-between items-center mt-4 my-4">
            <button
              onClick={prevStep}
              className="bg-[#00A8FF] text-white font-bold w-12 h-12 rounded-full border-none hover:brightness-110 transition flex items-center justify-center"
              aria-label="Back"
            >
              <ArrowLeft className="text-white p-1 m-0" />
            </button>
            <div className="flex space-x-2 items-center mt-4">
              {questions.map((_, i) => (
                <span
                  key={i}
                  className={`inline-block w-3 h-3 rounded-full transition ${
                    i === step ? "bg-blue-600" : "bg-gray-300"
                  }`}
                ></span>
              ))}
            </div>
            <button
              onClick={nextStep}
              className="bg-[#00A8FF] text-white font-bold w-12 h-12 rounded-full border-none hover:brightness-110 transition flex items-center justify-center"
              aria-label="Next"
            >
              <ArrowRight className="text-white p-1 m-0" />
            </button>
          </div>
        </>
      ) : (
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-4 text-center text-black">
            Declaration
          </h2>
          <label className="flex items-center gap-2 text-sm justify-center text-black text-center">
            <input
              type="checkbox"
              checked={formData.consent}
              onChange={(e) =>
                setFormData({ ...formData, consent: e.target.checked })
              }
              required
              className="text-white"
            />
            I agree this is a learning tool and not financial advice.
          </label>
          <button
            className="mt-4 bg-[#00A8FF] text-white font-bold px-6 py-3 rounded-[15px] border-none hover:brightness-110 transition"
            onClick={handleSubmit}
          >
            Let’s Build Your Portfolio
          </button>
        </div>
      )}
    </div>
  );
}
