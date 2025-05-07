// src/components/ProgressDots.jsx
export default function ProgressDots({ total, current }) {
  return (
    <div className="flex justify-center items-center space-x-2 mt-6">
      {Array.from({ length: total }, (_, i) => (
        <span
          key={i}
          className={`inline-block w-3 h-3 rounded-full ${
            i === current ? "bg-[#00A8FF]" : "bg-gray-300"
          }`}
        />
      ))}
    </div>
  );
}
