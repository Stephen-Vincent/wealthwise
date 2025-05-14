export default function DashboardButtons() {
  const handleNewSimulation = async () => {
    try {
      const response = await fetch("http://localhost:8000/clear-database", {
        method: "DELETE",
      });

      if (response.ok) {
        console.log("✅ Database cleared.");
        window.location.href = "/";
      } else {
        console.error("❌ Failed to clear database.");
      }
    } catch (error) {
      console.error("❌ Error clearing database:", error);
    }
  };

  return (
    <div className="flex justify-center space-x-6 mt-8">
      <button
        onClick={handleNewSimulation}
        className="bg-[#00A8FF] text-white font-bold px-6 py-3 rounded-[15px] hover:brightness-110 transition"
      >
        New Simulation
      </button>
      <button className="bg-white text-[#00A8FF] font-bold px-6 py-3 border border-[#00A8FF] rounded-[15px] hover:bg-[#00A8FF] hover:text-white transition">
        Print
      </button>
    </div>
  );
}
