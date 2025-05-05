import { useNavigate } from "react-router-dom";
import logo from "../assets/wealthwise.png";

export default function LandingScreen() {
  const navigate = useNavigate();

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        padding: "1rem",
        background: "linear-gradient(to bottom, white, #EAF6FB)",
      }}
    >
      <div style={{ textAlign: "center" }}>
        <div style={{ textAlign: "center", marginBottom: "2rem" }}>
          <img
            src={logo}
            alt="WealthWise logo"
            style={{
              width: "250px",
              height: "250px",
              objectFit: "contain",
              margin: "0 auto",
            }}
          />
        </div>

        <h1
          style={{
            fontSize: "1.5rem",
            fontWeight: "bold",
            color: "#333",
            marginBottom: "0.5rem",
          }}
        >
          Welcome to WealthWise
        </h1>
        <p style={{ color: "#666", marginBottom: "1.5rem" }}>
          Your AI-powered investing simulator
        </p>

        <button
          onClick={() => navigate("/onboarding")}
          style={{
            backgroundColor: "#00A8FF",
            color: "white",
            fontWeight: "bold",
            padding: "0.75rem 1.5rem",
            borderRadius: "15px",
            marginBottom: "1rem",
            border: "none",
            cursor: "pointer",
          }}
        >
          Get Started
        </button>
      </div>
    </div>
  );
}
