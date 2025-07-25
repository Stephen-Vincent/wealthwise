import logo from "../../assets/wealthwise.png";
import { useNavigate } from "react-router-dom";
import { useState } from "react";

export default function Sidebar({ scrollToSection, sectionRefs, onClose }) {
  const navigate = useNavigate();
  const userId = localStorage.getItem("userId");
  const [isOpen, setIsOpen] = useState(false);

  const handleLogout = () => {
    // Clear all authentication data
    localStorage.removeItem("access_token");
    localStorage.removeItem("user");
    localStorage.removeItem("userId");
    localStorage.removeItem("user_name");
    localStorage.removeItem("simulationId");
    localStorage.removeItem("portfolioData");
    localStorage.removeItem("simulationData");
    localStorage.removeItem("token");

    // Close sidebar before navigating
    setIsOpen(false);
    if (onClose) onClose();

    // Navigate back to landing page (will show main welcome)
    navigate("/");
  };

  const startNew = () => {
    // Clear simulation data but keep user logged in
    localStorage.removeItem("portfolioData");
    localStorage.removeItem("simulationId");
    localStorage.removeItem("simulationData");
    localStorage.removeItem("token");

    // Close sidebar before navigating
    setIsOpen(false);
    if (onClose) onClose();

    // Navigate back to landing page (will show welcome screen with panels)
    navigate("/");
  };

  const viewSimulations = () => {
    // Close sidebar before navigating
    setIsOpen(false);
    if (onClose) onClose();

    // Navigate back to landing page with a state indicating we want to show simulations
    navigate("/", { state: { showPanel: "simulations" } });
  };

  // Helper function to convert ref keys to readable labels
  const formatLabel = (key) => {
    // Remove 'Ref' suffix and capitalize first letter
    const label = key.replace(/Ref$/, "");
    return label.charAt(0).toUpperCase() + label.slice(1);
  };

  // Helper function to get emoji icon for each actual section key used
  const getIcon = (key) => {
    switch (key) {
      case "summaryRef":
        return "📋";
      case "graphRef":
        return "📈";
      case "aiSummaryRef":
        return "🤖";
      case "pieChartRef":
        return "🥧";
      default:
        return "🔹";
    }
  };

  const handleMenuClick = (ref) => {
    scrollToSection(ref);
    // Close sidebar after navigation
    setIsOpen(false);
    if (onClose) onClose();
  };

  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* Sidebar Toggle Button - Fixed position */}
      <button
        onClick={toggleSidebar}
        className={`
          fixed top-4 left-4 z-50
          w-12 h-12 bg-blue-600 hover:bg-blue-700
          text-white rounded-full shadow-lg
          flex items-center justify-center
          transition-all duration-300 ease-in-out
          lg:hidden
          ${isOpen ? "translate-x-80" : "translate-x-0"}
        `}
        aria-label={isOpen ? "Close menu" : "Open menu"}
      >
        <svg
          className={`w-6 h-6 transition-transform duration-300 ${
            isOpen ? "rotate-180" : ""
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          {isOpen ? (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          ) : (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16M4 18h16"
            />
          )}
        </svg>
      </button>

      {/* Backdrop overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed lg:static top-0 left-0 z-40
          lg:w-1/6 w-72 max-w-full
          h-screen lg:sticky lg:top-0
          flex flex-col
          lg:rounded-tr-3xl lg:rounded-br-3xl
          bg-gradient-to-b from-white via-blue-50 to-blue-100
          shadow-lg lg:shadow-none
          overflow-y-auto
          transition-transform duration-300 ease-in-out
          ${isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
        `}
      >
        {/* Logo Section - No mobile close button here since we have the floating one */}
        <div className="mb-6 mt-6 lg:mt-4 px-4 text-center">
          <img
            src={logo}
            alt="WealthWise Logo"
            className="w-20 h-20 lg:w-24 lg:h-24 mx-auto mb-2"
          />
          <div className="text-sm font-semibold text-gray-700">WealthWise</div>
        </div>

        {/* Navigation Section */}
        <nav className="mb-8 w-full px-4 lg:px-6 flex-1">
          <ul className="flex flex-col space-y-2 lg:space-y-3">
            {sectionRefs &&
              Object.entries(sectionRefs).map(([key, ref]) => (
                <li key={key}>
                  <button
                    onClick={() => handleMenuClick(ref)}
                    className="
                      w-full text-left px-3 lg:px-4 py-3 lg:py-2
                      rounded-lg hover:bg-white hover:bg-opacity-70
                      hover:shadow-sm
                      transition-all duration-200 font-medium 
                      flex items-center space-x-3
                      text-sm lg:text-base text-gray-700
                      border border-transparent hover:border-blue-200
                    "
                  >
                    <span className="text-lg flex-shrink-0">
                      {getIcon(key)}
                    </span>
                    <span className="truncate">{formatLabel(key)}</span>
                  </button>
                </li>
              ))}
          </ul>
        </nav>

        {/* Action Buttons Section */}
        <div className="px-4 lg:px-6 pb-6 space-y-3">
          <div className="w-full border-t border-gray-300 pt-4 mb-3"></div>

          <button
            onClick={handleLogout}
            className="
              w-full bg-red-500 hover:bg-red-600 
              text-white px-4 py-3
              rounded-lg font-medium
              text-sm lg:text-base
              transition-colors duration-200
              flex items-center justify-center space-x-2
              shadow-sm hover:shadow-md
            "
          >
            <span>🚪</span>
            <span>Logout</span>
          </button>

          <button
            onClick={startNew}
            className="
              w-full bg-[#00A8FF] hover:bg-[#0088CC]
              text-white px-4 py-3
              rounded-lg font-semibold
              text-sm lg:text-base
              transition-colors duration-200
              flex items-center justify-center space-x-2
              shadow-sm hover:shadow-md
            "
          >
            <span>✨</span>
            <span>New Simulation</span>
          </button>

          <button
            onClick={viewSimulations}
            className="
              w-full bg-green-500 hover:bg-green-600 
              text-white px-4 py-3
              rounded-lg font-semibold
              text-sm lg:text-base
              transition-colors duration-200
              flex items-center justify-center space-x-2
              shadow-sm hover:shadow-md
            "
          >
            <span>📊</span>
            <span>View All</span>
          </button>
        </div>
      </aside>
    </>
  );
}
