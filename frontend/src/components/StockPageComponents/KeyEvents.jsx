import React from "react";

const events = [
  {
    date: "2022-11-10",
    title: "Q4 Earnings Beat Expectations",
    description: "Apple reported higher-than-expected revenue and profit.",
  },
  {
    date: "2023-01-15",
    title: "High Market Volatility",
    description:
      "Stock experienced a sharp drop due to macroeconomic uncertainty.",
  },
  {
    date: "2023-03-22",
    title: "Product Launch",
    description: "Announced new product line boosting investor confidence.",
  },
  {
    date: "2023-05-12",
    title: "Dividend Announcement",
    description: "Apple declared a dividend payout for long-term investors.",
  },
];

export default function KeyEvents() {
  return (
    <section className="px-6 py-4">
      <h2 className="text-xl font-bold mb-4">Key Events</h2>
      <div className="space-y-4 border-l-2 border-blue-200 pl-4">
        {events.map((event, index) => (
          <div key={index} className="relative">
            <div className="absolute w-3 h-3 bg-blue-500 rounded-full -left-5 top-1.5"></div>
            <p className="text-sm text-gray-500">{event.date}</p>
            <h3 className="text-md font-semibold">{event.title}</h3>
            <p className="text-sm text-gray-700">{event.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
