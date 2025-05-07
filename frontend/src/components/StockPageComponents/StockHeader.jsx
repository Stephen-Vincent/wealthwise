import React from "react";

const StockHeader = ({ userName = "User", stockName = "Apple" }) => {
  return (
    <div className="flex justify-between items-center px-8 pt-6">
      <h2 className="text-xl font-semibold">Hi, {userName}</h2>
      <h1 className="text-2xl font-light">{stockName}</h1>
    </div>
  );
};

export default StockHeader;
