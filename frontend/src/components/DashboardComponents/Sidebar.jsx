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

    // Close mobile sidebar before navigating
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

    // Close mobile sidebar before navigating
    if (onClose) onClose();

    // Navigate back to landing page (will show welcome screen with panels)
    navigate("/");
  };

  const viewSimulations = () => {
    // Close mobile sidebar before navigating
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
    // Close mobile sidebar after navigation
    if (onClose) onClose();
  };

  return (
    <aside
      className="
      lg:w-1/6 w-80 max-w-full
      h-screen lg:sticky lg:top-0
      flex flex-col items-center
      lg:rounded-tr-3xl lg:rounded-br-3xl
      bg-gradient-to-b from-white via-blue-50 to-blue-100
      shadow-lg
      overflow-y-auto
    "
    >
      {/* Mobile close button */}
      <div className="lg:hidden w-full flex justify-end p-4">
        <button
          onClick={onClose}
          className="p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100"
          aria-label="Close menu"
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Logo Section */}
      <div className="mb-6 mt-2 lg:mt-4 px-4">
        <img
          src={logo}
          alt="WealthWise Logo"
          className="w-20 h-20 lg:w-24 lg:h-24 mx-auto mb-2"
        />
      </div>

      {/* Navigation Section */}
      <nav className="mb-8 w-full px-4 lg:px-6">
        <ul className="flex flex-col space-y-2 lg:space-y-3">
          {sectionRefs &&
            Object.entries(sectionRefs).map(([key, ref]) => (
              <li key={key}>
                <button
                  onClick={() => handleMenuClick(ref)}
                  className="
                    w-full text-left px-3 lg:px-4 py-2 lg:py-2
                    rounded-l-md hover:bg-gray-200 
                    hover:border-l-4 hover:border-blue-500 
                    transition-colors font-semibold 
                    flex items-center space-x-2 lg:space-x-3
                    text-sm lg:text-base
                  "
                >
                  <span className="text-base lg:text-lg">{getIcon(key)}</span>
                  <span>{formatLabel(key)}</span>
                </button>
              </li>
            ))}
        </ul>
      </nav>

      {/* Action Buttons Section */}
      <div className="flex flex-col items-center mt-auto w-full px-4 lg:px-6 pb-4 lg:pb-6">
        <button
          onClick={handleLogout}
          className="
            mb-4 lg:mb-5 
            bg-red-500 hover:bg-red-600 
            text-white px-4 lg:px-6 py-2 lg:py-3 
            rounded font-semibold w-full
            text-sm lg:text-base
            transition-colors
          "
        >
          Logout
        </button>

        <div className="w-full border-t border-gray-300 mb-4 lg:mb-5"></div>

        <button
          onClick={startNew}
          className="
            mb-4 lg:mb-5 
            bg-[#00A8FF] hover:bg-[#0088CC]
            text-white px-4 lg:px-6 py-2 lg:py-3 
            rounded font-bold w-full
            text-sm lg:text-base
            transition-colors
          "
        >
          Start New Simulation
        </button>

        <button
          onClick={viewSimulations}
          className="
            bg-green-500 hover:bg-green-600 
            text-white px-4 lg:px-6 py-2 lg:py-3 
            rounded font-bold w-full
            text-sm lg:text-base
            transition-colors
          "
        >
          View Simulations
        </button>
      </div>
    </aside>
  );
}
