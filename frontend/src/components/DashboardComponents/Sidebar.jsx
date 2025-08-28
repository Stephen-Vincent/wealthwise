/**
 * Sidebar component for the dashboard.
 * - Renders the sidebar navigation for the dashboard.
 * - Provides navigation between sections of the dashboard using refs.
 * - Includes action buttons for Logout, New Simulation, and View All Simulations.
 * - Uses helper functions to format labels and display icons for each section.
 */
import logo from "../../assets/wealthwise.png";
import { useNavigate } from "react-router-dom";

export default function Sidebar({ scrollToSection, sectionRefs, onClose }) {
  const navigate = useNavigate();
  const userId = localStorage.getItem("userId");

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
    if (onClose) onClose();

    // Navigate back to landing page (will show welcome screen with panels)
    navigate("/");
  };

  const viewSimulations = () => {
    // Close sidebar before navigating
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
      case "ShapRef":
        return "🔹";
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
    if (onClose) onClose();
  };

  return (
    <aside
      className="
      w-full h-full
      flex flex-col
      lg:rounded-tr-3xl lg:rounded-br-3xl
      bg-gradient-to-b from-white via-blue-50 to-blue-100
      shadow-lg lg:shadow-none
      overflow-y-auto
    "
    >
      {/* Logo Section */}
      <div className="mb-6 mt-6 lg:mt-4 px-4 text-center">
        <img
          src={logo}
          alt="WealthWise Logo"
          className="w-20 h-20 lg:w-24 lg:h-24 mx-auto mb-2"
        />
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
                  <span className="text-lg flex-shrink-0">{getIcon(key)}</span>
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
  );
}
