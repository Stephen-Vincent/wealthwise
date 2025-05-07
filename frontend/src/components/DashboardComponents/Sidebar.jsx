import logo from "../../assets/wealthwise.png";

export default function Sidebar() {
  return (
    <aside className="w-1/6 p-4  shadow-md min-h-screen flex flex-col items-center">
      <div className="mb-6 mt-4">
        <img
          src={logo}
          alt="WealthWise Logo"
          className="w-24 h-24 mx-auto mb-2"
        />
      </div>
    </aside>
  );
}
