/**
 * Signup Component
 *
 * Purpose:
 *   Renders a sign-up form for new users to create an account.
 *   Handles form state, input changes, submission to the backend, and error display.
 *   After successful signup, it switches to the login panel.
 *
 * Main Features:
 *   - Collects user's name, email, and password.
 *   - Validates required fields.
 *   - Submits user data to the backend API for account creation.
 *   - Displays error messages on failure.
 *   - Provides navigation to login and back to the welcome screen.
 *
 * Props:
 *   - onBack: Function to call when the user wants to return to the welcome screen.
 *   - onShowLogin: Function to call to show the login panel (after successful signup or when clicking "Log In").
 */
import { useState } from "react";

// Modified Signup component to work with panel system
export default function Signup({ onBack, onShowLogin }) {
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
  });
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const res = await fetch(`${import.meta.env.VITE_API_URL}/auth/signup`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(form),
    });

    const data = await res.json();

    if (res.ok) {
      console.log("Signup successful", data);
      // After successful signup, show login panel
      onShowLogin();
    } else {
      const errorMsg = data.detail || "Signup failed";
      setError(errorMsg);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center">
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

        {error && <p className="text-red-500 text-sm text-center">{error}</p>}

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
            onClick={onShowLogin}
          >
            Log In
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
