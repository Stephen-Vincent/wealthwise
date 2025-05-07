import React from "react";

export default function StockFooterActions({ onBack, onPrint }) {
  return (
    <div className="flex justify-between px-6 py-6 border-t mt-8">
      <button
        onClick={onBack}
        className="bg-white text-primary border border-primary font-bold px-6 py-2 rounded-[15px] hover:bg-blue-50 transition"
      >
        Back to Dashboard
      </button>
      <button
        onClick={onPrint}
        className="bg-primary text-white font-bold px-6 py-2 rounded-[15px] hover:brightness-110 transition"
      >
        Print
      </button>
    </div>
  );
}
