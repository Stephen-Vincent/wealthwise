/**
 * Header.jsx
 * ----------
 * Dashboard header component.
 * - Greets the user by name (fetched from localStorage)
 * - Shows main Dashboard title
 * - Mobile responsive with proper spacing for floating sidebar button
 * - Includes portfolio quick stats on larger screens
 * - Styled with Tailwind CSS
 */

export default function Header({ portfolioData }) {
  let userName = "user";

  // Attempt to get user's name from localStorage
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

  // Fallback to user_name if user object doesn't have name
  if (userName === "user") {
    const storedUserName = localStorage.getItem("user_name");
    if (storedUserName) {
      userName = storedUserName;
    }
  }

  return (
    <header className="bg-white border-b border-gray-200 px-4 lg:px-8 py-3 lg:py-6 pl-20 lg:pl-8 mb-6">
      <div className="flex flex-col space-y-2 lg:space-y-3">
        {/* Greeting and title */}
        <div>
          <h2 className="text-sm lg:text-lg text-gray-600">Hi, {userName}</h2>
          <h1 className="text-xl lg:text-3xl font-bold text-gray-900">
            Dashboard
          </h1>
        </div>

        {/* Quick stats */}
        {portfolioData && (
          <div className="hidden sm:flex flex-wrap gap-3 lg:gap-6 text-xs lg:text-sm text-gray-600">
            {portfolioData.target_value && (
              <div className="flex items-center space-x-1">
                <span>🎯</span>
                <span>
                  Target: £{(portfolioData.target_value / 1000).toFixed(0)}k
                </span>
              </div>
            )}
            {portfolioData.risk_label && (
              <div className="flex items-center space-x-1">
                <span>📊</span>
                <span>{portfolioData.risk_label} Risk</span>
              </div>
            )}
            {portfolioData.timeframe && (
              <div className="flex items-center space-x-1">
                <span>⏰</span>
                <span>{portfolioData.timeframe} years</span>
              </div>
            )}
            {portfolioData.goal && (
              <div className="flex items-center space-x-1">
                <span>💼</span>
                <span>
                  {portfolioData.goal.charAt(0).toUpperCase() +
                    portfolioData.goal.slice(1)}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
