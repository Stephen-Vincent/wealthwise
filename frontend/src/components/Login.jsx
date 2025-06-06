// src/components/Login.jsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    email: "",
    password: "",
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const res = await fetch("http://localhost:8000/auth/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(form),
    });

    const data = await res.json();

    if (res.ok) {
      console.log("Login successful", data);
      localStorage.setItem("userId", data.user_id);
      localStorage.setItem("user_name", data.name);
      console.log("Saved user_name:", localStorage.getItem("user_name"));

      const simRes = await fetch(
        `http://localhost:8000/users/${data.user_id}/simulations`
      );
      const simulations = await simRes.json();
      if (simRes.ok && Array.isArray(simulations)) {
        if (simulations.length > 0) {
          navigate(`/simulations`);
        } else {
          navigate("/onboarding");
        }
      } else {
        console.error("Error fetching simulations:", simulations);
        alert("Error checking simulations. Please try again.");
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
