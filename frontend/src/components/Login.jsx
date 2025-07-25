import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

export default function Login({ onBack, onShowSignup, onShowWelcomeScreen }) {
  const [form, setForm] = useState({
    email: "",
    password: "",
  });
  const navigate = useNavigate();

  useEffect(() => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("user");
  }, []);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await res.json();

      if (res.ok) {
        const userId = data.user?.id;
        if (!userId) {
          alert("Login response did not include a user ID.");
          return;
        }

        localStorage.setItem("user", JSON.stringify(data.user));
        localStorage.setItem("userId", String(userId));
        localStorage.setItem("access_token", data.access_token);

        // Show welcome screen instead of navigating directly
        onShowWelcomeScreen(data.user);
      } else {
        alert(data.detail || "Login failed");
      }
    } catch (error) {
      console.error("Login error:", error);
      alert("An error occurred during login. Please try again.");
    }
  };

  const handleForgotPassword = () => {
    navigate("/forgot-password");
  };

  return (
    <div className="flex flex-col justify-center items-center">
      <form
        onSubmit={handleSubmit}
        className="bg-white p-8 rounded-lg shadow-md w-full max-w-sm space-y-4"
      >
        <h2 className="text-2xl font-bold text-center">Log In</h2>
        <input
          type="email"
          name="email"
          value={form.email}
          onChange={handleChange}
          placeholder="Email"
          className="w-full p-3 border rounded"
          required
        />
        <input
          type="password"
          name="password"
          value={form.password}
          onChange={handleChange}
          placeholder="Password"
          className="w-full p-3 border rounded"
          required
        />
        <button
          type="submit"
          className="bg-[#00A8FF] w-full text-white py-3 rounded font-bold"
        >
          Log In
        </button>

        {/* Forgot Password Link */}
        <div className="text-center">
          <button
            type="button"
            onClick={handleForgotPassword}
            className="text-[#00A8FF] text-sm hover:underline focus:outline-none"
          >
            Forgot your password?
          </button>
        </div>

        <p className="text-center text-sm">
          New here?{" "}
          <span
            className="text-[#00A8FF] cursor-pointer underline"
            onClick={onShowSignup}
          >
            Create an account
          </span>
        </p>
      </form>
      <button
        onClick={onBack}
        className="mt-4 text-[#00A8FF] underline text-sm"
      >
        ← Back to Welcome
      </button>
    </div>
  );
}
