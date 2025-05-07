import React from "react";
import { useParams } from "react-router-dom";
import StockHeader from "./StockPageComponents/StockHeader";
import StockSummary from "./StockPageComponents/StockSummary";
import StockLineChart from "./StockPageComponents/StockLineChart";
import MarketSentiment from "./StockPageComponents/MarketSentiment";
import KeyEvents from "./StockPageComponents/KeyEvents";
import AISummary from "./StockPageComponents/AISummary";
import StockFooterActions from "./StockPageComponents/StockFooterActions";
import logo from "../assets/wealthwise.png";

export default function StockPage() {
  const { stock } = useParams();
  const handleBack = () => {
    window.history.back();
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-[#D4E8F2] font-sans text-gray-800">
      <div className="flex justify-center py-4">
        <img src={logo} alt="WealthWise Logo" className="h-16 w-16" />
      </div>
      <StockHeader userName="Stephen" stockName={stock} />
      <StockSummary />
      <StockLineChart />
      <MarketSentiment />
      <KeyEvents />
      <AISummary />
      <StockFooterActions onBack={handleBack} onPrint={handlePrint} />
    </div>
  );
}
