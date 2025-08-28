/**
 * PasswordResetConfirm.jsx
 *
 * This component provides the password reset confirmation page.
 * It verifies the reset token (via `/auth/verify-reset-token`), allows the user to set a new password,
 * and validates password strength using several criteria (length, uppercase, lowercase, number, special character).
 * The component handles different states: loading (token verification), invalid/expired token,
 * successful reset, and the reset form itself.
 * The password reset action is performed via the `/auth/reset-password` endpoint.
 */
// src/components/PasswordResetConfirm.jsx
import { useState, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import wealthwiseLogo from "../assets/wealthwise.png";

const PasswordResetConfirm = () => {
  const [searchParams] = useSearchParams();
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [success, setSuccess] = useState(false);
  const [tokenValid, setTokenValid] = useState(null);
  const [tokenEmail, setTokenEmail] = useState("");
  const navigate = useNavigate();

  const token = searchParams.get("token");

  useEffect(() => {
    if (!token) {
      setMessage("Invalid reset link");
      setTokenValid(false);
      return;
    }

    // Verify token on component mount
    verifyToken();
  }, [token]);

  const verifyToken = async () => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/auth/verify-reset-token`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ token }),
        }
      );

      const data = await response.json();

      if (data.valid) {
        setTokenValid(true);
        setTokenEmail(data.email);
      } else {
        setTokenValid(false);
        setMessage(data.message || "Invalid or expired reset token");
      }
    } catch (error) {
      setTokenValid(false);
      setMessage("Error verifying reset token");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (password !== confirmPassword) {
      setMessage("Passwords do not match");
      return;
    }

    setLoading(true);
    setMessage("");

    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/auth/reset-password`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            token,
            new_password: password,
            confirm_password: confirmPassword,
          }),
        }
      );

      const data = await response.json();

      if (response.ok && data.success) {
        setSuccess(true);
        setMessage("Password reset successfully!");
      } else {
        setMessage(data.detail || data.message || "Failed to reset password");
      }
    } catch (error) {
      setMessage("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const getPasswordStrength = (password) => {
    let score = 0;
    const checks = {
      length: password.length >= 8,
      uppercase: /[A-Z]/.test(password),
      lowercase: /[a-z]/.test(password),
      number: /\d/.test(password),
      special: /[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]/.test(password),
    };

    Object.values(checks).forEach((check) => {
      if (check) score++;
    });

    return { score, checks };
  };

  const passwordStrength = getPasswordStrength(password);

  if (tokenValid === false) {
    return (
      <div
        className="min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8"
        style={{
          background: "linear-gradient(to bottom, white 0%, #a3cde0 100%)",
        }}
      >
        <div className="max-w-md w-full space-y-8">
          <div className="text-center">
            {/* Logo */}
            <div className="flex justify-center mb-6">
              <img
                src={wealthwiseLogo}
                alt="WealthWise"
                className="h-16 w-auto"
              />
            </div>

            <div className="mx-auto h-12 w-12 bg-red-100 rounded-full flex items-center justify-center">
              <svg
                className="h-6 w-6 text-red-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M6 18L18 6M6 6l12 12"
                ></path>
              </svg>
            </div>
            <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
              Invalid Reset Link
            </h2>
            <p className="mt-2 text-sm text-gray-600">{message}</p>
            <div className="mt-6 space-y-4">
              <button
                onClick={() => navigate("/forgot-password")}
                className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-[#00A8FF] hover:bg-[#0088CC] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#00A8FF]"
              >
                Request New Reset Link
              </button>
              <button
                onClick={() => navigate("/login")}
                className="w-full flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#00A8FF]"
              >
                Back to Login
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (success) {
    return (
      <div
        className="min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8"
        style={{
          background: "linear-gradient(to bottom, white 0%, #a3cde0 100%)",
        }}
      >
        <div className="max-w-md w-full space-y-8">
          <div className="text-center">
            {/* Logo */}
            <div className="flex justify-center mb-6">
              <img
                src={wealthwiseLogo}
                alt="WealthWise"
                className="h-16 w-auto"
              />
            </div>

            <div className="mx-auto h-12 w-12 bg-green-100 rounded-full flex items-center justify-center">
              <svg
                className="h-6 w-6 text-green-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M5 13l4 4L19 7"
                ></path>
              </svg>
            </div>
            <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
              Password Reset Complete
            </h2>
            <p className="mt-2 text-sm text-gray-600">
              Your password has been successfully reset.
            </p>
            <button
              onClick={() => navigate("/login")}
              className="mt-6 w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-[#00A8FF] hover:bg-[#0088CC] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#00A8FF]"
            >
              Sign In with New Password
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (tokenValid === null) {
    return (
      <div
        className="min-h-screen flex items-center justify-center"
        style={{
          background: "linear-gradient(to bottom, white 0%, #a3cde0 100%)",
        }}
      >
        <div className="text-center">
          {/* Logo */}
          <div className="flex justify-center mb-6">
            <img
              src={wealthwiseLogo}
              alt="WealthWise"
              className="h-16 w-auto"
            />
          </div>
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-[#00A8FF]"></div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8"
      style={{
        background: "linear-gradient(to bottom, white 0%, #a3cde0 100%)",
      }}
    >
      <div className="max-w-md w-full space-y-8">
        {/* Logo */}
        <div className="flex justify-center">
          <img src={wealthwiseLogo} alt="WealthWise" className="h-16 w-auto" />
        </div>

        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Set New Password
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Creating new password for:{" "}
            <span className="font-medium">{tokenEmail}</span>
          </p>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-700"
              >
                New Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                className="mt-1 appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-[#00A8FF] focus:border-[#00A8FF] focus:z-10 sm:text-sm"
                placeholder="Enter new password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            <div>
              <label
                htmlFor="confirmPassword"
                className="block text-sm font-medium text-gray-700"
              >
                Confirm New Password
              </label>
              <input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                required
                className="mt-1 appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-[#00A8FF] focus:border-[#00A8FF] focus:z-10 sm:text-sm"
                placeholder="Confirm new password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
              />
            </div>
          </div>

          {/* Password Strength Indicator */}
          {password && (
            <div className="space-y-2">
              <div className="text-sm font-medium text-gray-700">
                Password Strength:
              </div>
              <div className="space-y-1">
                <div className="flex space-x-1">
                  {[1, 2, 3, 4, 5].map((level) => (
                    <div
                      key={level}
                      className={`h-2 flex-1 rounded ${
                        passwordStrength.score >= level
                          ? level <= 2
                            ? "bg-red-400"
                            : level <= 3
                            ? "bg-yellow-400"
                            : level <= 4
                            ? "bg-blue-400"
                            : "bg-green-400"
                          : "bg-gray-200"
                      }`}
                    />
                  ))}
                </div>
                <div className="text-xs text-gray-600 space-y-1">
                  <div
                    className={`flex items-center ${
                      passwordStrength.checks.length
                        ? "text-green-600"
                        : "text-gray-400"
                    }`}
                  >
                    <span className="mr-2">
                      {passwordStrength.checks.length ? "✓" : "○"}
                    </span>
                    At least 8 characters
                  </div>
                  <div
                    className={`flex items-center ${
                      passwordStrength.checks.uppercase
                        ? "text-green-600"
                        : "text-gray-400"
                    }`}
                  >
                    <span className="mr-2">
                      {passwordStrength.checks.uppercase ? "✓" : "○"}
                    </span>
                    Uppercase letter
                  </div>
                  <div
                    className={`flex items-center ${
                      passwordStrength.checks.lowercase
                        ? "text-green-600"
                        : "text-gray-400"
                    }`}
                  >
                    <span className="mr-2">
                      {passwordStrength.checks.lowercase ? "✓" : "○"}
                    </span>
                    Lowercase letter
                  </div>
                  <div
                    className={`flex items-center ${
                      passwordStrength.checks.number
                        ? "text-green-600"
                        : "text-gray-400"
                    }`}
                  >
                    <span className="mr-2">
                      {passwordStrength.checks.number ? "✓" : "○"}
                    </span>
                    Number
                  </div>
                  <div
                    className={`flex items-center ${
                      passwordStrength.checks.special
                        ? "text-green-600"
                        : "text-gray-400"
                    }`}
                  >
                    <span className="mr-2">
                      {passwordStrength.checks.special ? "✓" : "○"}
                    </span>
                    Special character
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Password Match Indicator */}
          {confirmPassword && (
            <div
              className={`text-sm ${
                password === confirmPassword ? "text-green-600" : "text-red-600"
              }`}
            >
              {password === confirmPassword
                ? "✓ Passwords match"
                : "✗ Passwords do not match"}
            </div>
          )}

          {message && (
            <div className="rounded-md bg-red-50 p-4">
              <div className="text-sm text-red-700">{message}</div>
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={
                loading ||
                passwordStrength.score < 5 ||
                password !== confirmPassword
              }
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-[#00A8FF] hover:bg-[#0088CC] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#00A8FF] disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <svg
                  className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
              ) : null}
              {loading ? "Resetting..." : "Reset Password"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default PasswordResetConfirm;
