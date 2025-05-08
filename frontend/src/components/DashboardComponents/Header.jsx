import { useContext } from "react";
import PortfolioContext from "../../context/PortfolioContext";

export default function Header() {
  const { portfolioData } = useContext(PortfolioContext);
  const userName = portfolioData?.name || "User";

  return (
    <div className="mb-6">
      <h2 className="text-lg text-gray-600">Hi, {userName}</h2>
      <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
    </div>
  );
}
