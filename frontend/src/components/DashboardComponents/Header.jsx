import { useContext } from "react";
import PortfolioContext from "../../context/PortfolioContext";

export default function Header() {
  let userName = "user";

  const storedUser = localStorage.getItem("user");
  if (storedUser) {
    try {
      const parsedUser = JSON.parse(storedUser);
      if (parsedUser.name) {
        userName = parsedUser.name;
      }
    } catch (error) {
      console.error("Error parsing stored user from localStorage:", error);
    }
  }

  return (
    <div className="mb-6">
      <h2 className="text-lg text-gray-600">Hi, {userName}</h2>
      <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
    </div>
  );
}
