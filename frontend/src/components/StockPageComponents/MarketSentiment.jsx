import React from "react";

const sentimentData = [
  { label: "Overall Sentiment", value: "Bullish", color: "text-green-600" },
  { label: "Volume Trend", value: "Above Average", color: "text-blue-600" },
  { label: "Volatility", value: "Moderate", color: "text-yellow-600" },
  { label: "Investor Activity", value: "Rising", color: "text-indigo-600" },
  { label: "Market Confidence", value: "High", color: "text-green-500" },
];

export default function MarketSentiment() {
  return (
    <section className="px-6 py-4">
      <h2 className="text-xl font-bold mb-4">Market Sentiment</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {sentimentData.map((item, index) => (
          <div
            key={index}
            className="border rounded-md p-4 bg-white shadow-sm hover:shadow-md transition-shadow"
          >
            <p className="text-xs text-gray-500">{item.label}</p>
            <p className={`text-lg font-semibold ${item.color}`}>
              {item.value}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
