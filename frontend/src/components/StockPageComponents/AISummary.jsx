import React from "react";

export default function AISummary() {
  return (
    <section className="px-6 py-4">
      <h2 className="text-xl font-bold mb-4">AI Summary</h2>

      <div className="space-y-6">
        <div>
          <h3 className="text-md font-semibold mb-1">Overview</h3>
          <p className="text-sm text-gray-700">
            This stock showed steady growth over the simulation period.
            Performance was influenced by both macroeconomic factors and
            company-specific events.
          </p>
        </div>

        <div>
          <h3 className="text-md font-semibold mb-1">
            Key Drivers of Performance
          </h3>
          <p className="text-sm text-gray-700">
            Strong earnings reports, consistent innovation, and market optimism
            around new product launches significantly boosted investor
            confidence.
          </p>
        </div>

        <div>
          <h3 className="text-md font-semibold mb-1">Challenges Encountered</h3>
          <p className="text-sm text-gray-700">
            Brief periods of volatility were observed due to global market
            tensions and industry regulation concerns.
          </p>
        </div>

        <div>
          <h3 className="text-md font-semibold mb-1">Overall Sentiment</h3>
          <p className="text-sm text-gray-700">
            AI analysis rates this stock as{" "}
            <span className="font-bold text-green-600">bullish</span> at the end
            of the simulation.
          </p>
        </div>

        <div>
          <h3 className="text-md font-semibold mb-1">Final Insight</h3>
          <p className="text-sm text-gray-700">
            Based on historic performance and market behavior, this stock may
            continue to perform well in the medium to long term with moderate
            risk exposure.
          </p>
        </div>
      </div>
    </section>
  );
}
