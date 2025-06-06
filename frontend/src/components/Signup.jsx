// src/components/Signup.jsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Signup() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
  });
  const [error, setError] = useState(""); // NEW

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const res = await fetch("http://localhost:8000/auth/signup", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(form),
    });

    const data = await res.json();

    if (res.ok) {
      console.log("Signup successful", data);
      navigate("/login"); // redirect after successful signup
    } else {
      const errorMsg = data.detail || "Signup failed";
      setError(errorMsg); // UPDATED
    }
  };

  return (
    <div className="flex flex-col items-center justify-center ">
      <form
        onSubmit={handleSubmit}
        className="bg-white p-8 rounded-lg shadow-md w-full max-w-sm space-y-4"
      >
        <h2 className="text-2xl font-bold text-center">Create an Account</h2>

        <input
          type="text"
          name="name"
          value={form.name}
          onChange={handleChange}
          placeholder="Full Name"
          className="w-full p-3 border rounded"
          required
        />

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

        {error && (
          <p className="text-red-500 text-sm text-center">{error}</p> // NEW
        )}

        <button
          type="submit"
          className="bg-[#00A8FF] w-full text-white py-3 rounded font-bold"
        >
          Sign Up
        </button>

        <p className="text-center text-sm">
          Already have an account?{" "}
          <span
            className="text-[#00A8FF] cursor-pointer underline"
            onClick={() => navigate("/login")}
          >
            Log In
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
