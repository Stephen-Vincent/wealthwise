import React from "react";

const StockSummary = () => {
  return (
    <section className="px-6 py-4">
      <h2 className="text-xl font-bold mb-4">Stock Summary</h2>
      <p className="mb-2 text-sm text-gray-700">My Goal: Buy a house</p>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <div className="border p-4 text-center">
          <p className="text-xs text-gray-500">Risk Score:</p>
          <p className="text-lg font-semibold">65</p>
        </div>
        <div className="border p-4 text-center">
          <p className="text-xs text-gray-500">Starting Balance:</p>
          <p className="text-lg font-semibold">£500</p>
        </div>
        <div className="border p-4 text-center">
          <p className="text-xs text-gray-500">Current Balance:</p>
          <p className="text-lg font-semibold">£1,000</p>
        </div>
        <div className="border p-4 text-center">
          <p className="text-xs text-gray-500">Return %:</p>
          <p className="text-lg font-semibold">100%</p>
        </div>
      </div>
    </section>
  );
};

export default StockSummary;
