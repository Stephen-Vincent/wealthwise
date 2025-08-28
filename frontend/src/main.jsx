// Entry point of the React application. Imports global styles, mounts the root component (App) to the DOM, and wraps it in StrictMode for highlighting potential issues.
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
