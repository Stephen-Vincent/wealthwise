// src/components/Login.jsx
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    email: "",
    password: "",
  });

  useEffect(() => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("user");
  }, []);

  // Handle input field changes
  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    // Send login request to backend
    const res = await fetch("http://localhost:8000/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form),
    });

    const data = await res.json();
    console.log("Login response:", data); // ← Add this line

    if (res.ok) {
      // Safer user info extraction
      const userId = data.user?.id;
      console.log("✅ Extracted user ID:", userId);

      const userName = data.user?.name;

      if (!userId) {
        alert("Login response did not include a user ID.");
        return;
      }

      localStorage.setItem("user", JSON.stringify(data.user));
      localStorage.setItem("userId", String(userId));
      console.log(
        "📦 Stored user ID in localStorage:",
        localStorage.getItem("user")
      );
      localStorage.setItem("access_token", data.access_token);

      try {
        const simRes = await fetch("http://localhost:8000/simulations", {
          headers: {
            Authorization: `Bearer ${data.access_token}`,
          },
        });
        const simulations = await simRes.json();

        if (simRes.ok && Array.isArray(simulations)) {
          navigate(simulations.length > 0 ? "/simulations" : "/onboarding");
        } else {
          console.warn("No simulations found or response error:", simulations);
          navigate("/onboarding");
        }
      } catch (err) {
        console.error("Simulation fetch error:", err);
        navigate("/onboarding");
      }
    } else {
      alert(data.detail || "Login failed");
    }
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

        <p className="text-center text-sm">
          New here?{" "}
          <span
            className="text-[#00A8FF] cursor-pointer underline"
            onClick={() => navigate("/signup")}
          >
            Create an account
          </span>
        </p>
      </form>

      <button
        onClick={() => navigate("/")}
        className="mt-4 text-[#00A8FF] underline text-sm"
      >
        ← Back to Welcome
      </button>
    </div>
  );
}
