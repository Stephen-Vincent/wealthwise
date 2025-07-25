import React, { useState, useEffect } from "react";
import { usePortfolio } from "../../context/PortfolioContext";
import logo from "../../assets/wealthwise.png";

export default function DashboardButtons() {
  const { portfolioData } = usePortfolio();
  const [userName, setUserName] = useState("Investor");

  // Function to get user name dynamically
  const fetchUserName = async () => {
    try {
      const userId = portfolioData?.user_id;
      if (!userId) return;

      // Try localStorage first
      const storedUser = localStorage.getItem("user");
      if (storedUser) {
        try {
          const user = JSON.parse(storedUser);
          if (user.name) {
            setUserName(user.name);
            return;
          }
        } catch (e) {
          console.log("Error parsing stored user:", e);
        }
      }

      // Try API call
      const token = localStorage.getItem("token");
      if (token) {
        try {
          const response = await fetch(
            `${import.meta.env.VITE_API_URL}/api/users/${userId}`,
            {
              headers: {
                Authorization: `Bearer ${token}`,
                "Content-Type": "application/json",
              },
            }
          );

          if (response.ok) {
            const userData = await response.json();
            if (userData.name) {
              setUserName(userData.name);
            }
          }
        } catch (error) {
          console.log("API call failed:", error);
        }
      }
    } catch (error) {
      console.error("Error fetching user name:", error);
    }
  };

  useEffect(() => {
    if (portfolioData?.user_id) {
      fetchUserName();
    }
  }, [portfolioData?.user_id]);

  const handlePrint = async () => {
    const printContent = await generatePrintHTML();

    const iframe = document.createElement("iframe");
    iframe.style.position = "absolute";
    iframe.style.top = "-10000px";
    iframe.style.left = "-10000px";
    iframe.style.width = "1px";
    iframe.style.height = "1px";
    iframe.style.opacity = "0";

    document.body.appendChild(iframe);

    const doc = iframe.contentWindow.document;
    doc.open();
    doc.write(printContent);
    doc.close();

    iframe.onload = () => {
      iframe.contentWindow.focus();
      iframe.contentWindow.print();

      setTimeout(() => {
        document.body.removeChild(iframe);
      }, 1000);
    };
  };

  const generatePrintHTML = async () => {
    const currentDate = new Date().toLocaleDateString("en-GB", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });

    // Convert logo to base64
    const getLogoBase64 = () => {
      return new Promise((resolve) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = function () {
          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d");
          canvas.width = this.width;
          canvas.height = this.height;
          ctx.drawImage(this, 0, 0);
          resolve(canvas.toDataURL("image/png"));
        };
        img.onerror = () => resolve("");
        img.src = logo;
      });
    };

    const logoBase64 = await getLogoBase64();

    // Helper functions
    const formatCurrency = (value) => {
      if (value === null || value === undefined || isNaN(value)) return "N/A";
      return new Intl.NumberFormat("en-GB", {
        style: "currency",
        currency: "GBP",
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
      }).format(value);
    };

    const formatPercentage = (value) => {
      if (value === null || value === undefined || isNaN(value)) return "N/A";
      const percentageValue = value > 1 ? value : value * 100;
      return `${percentageValue > 0 ? "+" : ""}${Number(
        percentageValue
      ).toFixed(2)}%`;
    };

    // Extract ALL data from portfolioData - NO HARDCODED VALUES
    console.log("🔍 EXTRACTING DATA FROM PORTFOLIO:");
    console.log("portfolioData.results:", portfolioData?.results);

    const goal = portfolioData?.goal || "Investment Goal";
    const timeframe = portfolioData?.timeframe || "Not specified";
    const riskLabel = portfolioData?.risk_label || "Not specified";
    const riskScore = portfolioData?.risk_score || "Not calculated";
    const targetValue = portfolioData?.target_value || 0;

    // Get financial data from results object
    const totalInvested = portfolioData?.results?.starting_value || 0;
    const portfolioValue = portfolioData?.results?.end_value || 0;
    const rawReturn = portfolioData?.results?.return || 0;
    const displayedReturn = rawReturn > 1 ? rawReturn : rawReturn * 100;

    // Calculate annualized return dynamically
    const annualizedReturn = (() => {
      if (totalInvested <= 0 || portfolioValue <= 0 || timeframe <= 0) return 0;

      // CAGR formula: (End Value / Start Value)^(1/years) - 1
      const cagr = Math.pow(portfolioValue / totalInvested, 1 / timeframe) - 1;
      return cagr * 100; // Convert to percentage
    })();

    console.log("📊 EXTRACTED VALUES:");
    console.log("Goal:", goal);
    console.log("Target Value:", targetValue);
    console.log("Timeframe:", timeframe);
    console.log("Risk Label:", riskLabel);
    console.log("Risk Score:", riskScore);
    console.log("Total Invested (starting_value):", totalInvested);
    console.log("Portfolio Value (end_value):", portfolioValue);
    console.log("Return (decimal):", rawReturn);
    console.log("Return (percentage):", displayedReturn);
    console.log(
      "Calculated Annualized Return:",
      annualizedReturn.toFixed(2) + "%"
    );

    // Handle stocks data
    const stocks = portfolioData?.stocks_picked || [];

    // Enhanced sector mapping
    const getSectorForStock = (stock) => {
      if (stock.sector) return stock.sector;
      if (stock.industry) return stock.industry;
      if (stock.category) return stock.category;
      if (stock.asset_class) return stock.asset_class;

      const symbol = stock.symbol || stock.ticker || "";
      const name = stock.name || stock.company_name || stock.description || "";

      if (symbol.includes("VTI") || name.includes("Total Stock Market"))
        return "Broad Market ETF";
      if (symbol.includes("VEA") || name.includes("Developed Markets"))
        return "International Developed Markets";
      if (symbol.includes("VWO") || name.includes("Emerging Markets"))
        return "Emerging Markets";
      if (symbol.includes("VNQ") || name.includes("Real Estate"))
        return "Real Estate";
      if (symbol.includes("BND") || name.includes("Bond"))
        return "Fixed Income";
      if (symbol.includes("VTEB") || name.includes("Tax-Exempt"))
        return "Fixed Income";
      if (symbol.includes("FTSE")) return "International Equity";

      return "Diversified Portfolio";
    };

    // Format stock table rows
    const stockRowsHTML = stocks
      .map((stock) => {
        const allocation =
          stock.allocation || stock.weight || stock.percentage || 0;
        const sector = getSectorForStock(stock);

        return `
        <tr>
          <td><strong>${stock.symbol || stock.ticker || "N/A"}</strong></td>
          <td>${
            stock.name || stock.company_name || stock.description || "N/A"
          }</td>
          <td>${allocation ? allocation.toFixed(1) + "%" : "N/A"}</td>
          <td>${sector}</td>
        </tr>
      `;
      })
      .join("");

    // Portfolio holdings section
    const portfolioHoldingsSection =
      stocks?.length > 0
        ? `
      <div class="summary-section">
        <h2 class="section-title">Portfolio Holdings</h2>
        <table class="portfolio-table">
          <thead>
            <tr>
              <th>Stock Symbol</th>
              <th>Company</th>
              <th>Allocation</th>
              <th>Sector</th>
            </tr>
          </thead>
          <tbody>
            ${stockRowsHTML}
          </tbody>
        </table>
      </div>
    `
        : "";

    // AI Summary with markdown conversion
    const aiSummary =
      portfolioData?.ai_summary || "Investment analysis not available.";

    const formatAISummary = (text) => {
      if (!text) return "";

      let htmlText = text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*([^*]+)\*/g, "<em>$1</em>")
        .replace(/(\d+\.)\s+/g, "<br><strong>$1</strong> ")
        .replace(/[""]([^""]+)[""]?/g, '<em>"$1"</em>');

      const sentences = htmlText.split(
        /(?<=[.!?])\s*(?=["']|\*\*|<strong>|[A-Z])/
      );
      let formattedText = "";
      let currentParagraph = "";

      sentences.forEach((sentence, index) => {
        sentence = sentence.trim();
        if (!sentence) return;

        if (
          sentence.includes("<strong>") ||
          sentence.includes("Key lessons") ||
          sentence.includes("Remember") ||
          sentence.includes("Important") ||
          sentence.includes("However") ||
          sentence.includes("<br>") ||
          (index > 0 && sentence.length > 120)
        ) {
          if (currentParagraph) {
            formattedText += `<p>${currentParagraph.trim()}</p>`;
            currentParagraph = "";
          }
          currentParagraph = sentence + " ";
        } else {
          currentParagraph += sentence + " ";
        }
      });

      if (currentParagraph) {
        formattedText += `<p>${currentParagraph.trim()}</p>`;
      }

      if (formattedText.length < 50) {
        const allSentences = htmlText.split(/(?<=[.!?])\s+/);
        formattedText = "";

        for (let i = 0; i < allSentences.length; i += 2) {
          const paragraph = allSentences
            .slice(i, i + 2)
            .join(" ")
            .trim();
          if (paragraph) {
            formattedText += `<p>${paragraph}</p>`;
          }
        }
      }

      formattedText = formattedText
        .replace(/<br>\s*<br>/g, "<br>")
        .replace(/\s+/g, " ")
        .trim();

      return formattedText || `<p>${htmlText}</p>`;
    };

    const aiSummarySection = `
      <div class="summary-section">
        <h2 class="section-title">AI Investment Analysis</h2>
        <div class="ai-summary-card">
          ${formatAISummary(aiSummary)}
        </div>
      </div>
    `;

    // Goal achievement section
    const goalAchievementSection =
      portfolioData?.target_achieved !== undefined
        ? `
      <div class="summary-section">
        <h2 class="section-title">Goal Achievement</h2>
        <div class="summary-card">
          <h3>Target Achievement Status</h3>
          <div class="value ${
            portfolioData.target_achieved ? "positive" : "negative"
          }">
            ${
              portfolioData.target_achieved
                ? "✓ Target Achieved"
                : "✗ Target Not Achieved"
            }
          </div>
        </div>
      </div>
    `
        : "";

    return `
      <!DOCTYPE html>
      <html>
        <head>
          <title>WealthWise Investment Summary - ${userName}</title>
          <style>
            @page {
              size: A4;
              margin: 20mm;
            }
            
            * {
              margin: 0;
              padding: 0;
              box-sizing: border-box;
            }
            
            body {
              font-family: 'Arial', sans-serif;
              line-height: 1.6;
              color: #333;
              background: white;
            }
            
            .header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: 30px;
              padding-bottom: 20px;
              border-bottom: 3px solid #00A8FF;
            }
            
            .logo-section {
              display: flex;
              align-items: center;
              gap: 15px;
            }
            
            .logo {
              width: 60px;
              height: 60px;
              margin-right: 15px;
            }
            
            .company-info h1 {
              font-size: 24px;
              color: #00A8FF;
              font-weight: bold;
            }
            
            .company-info p {
              font-size: 12px;
              color: #666;
            }
            
            .report-info {
              text-align: right;
            }
            
            .report-info h2 {
              font-size: 18px;
              color: #333;
              margin-bottom: 5px;
            }
            
            .report-info p {
              font-size: 12px;
              color: #666;
            }
            
            .summary-section {
              margin-bottom: 25px;
            }
            
            .section-title {
              font-size: 16px;
              font-weight: bold;
              color: #00A8FF;
              margin-bottom: 15px;
              border-bottom: 2px solid #f0f0f0;
              padding-bottom: 5px;
            }
            
            .summary-grid {
              display: grid;
              grid-template-columns: repeat(3, 1fr);
              gap: 15px;
              margin-bottom: 20px;
            }
            
            @media print {
              .summary-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
              }
            }
            
            .summary-card {
              background: #f8f9fa;
              padding: 15px;
              border-radius: 8px;
              border-left: 4px solid #00A8FF;
            }
            
            .summary-card h3 {
              font-size: 14px;
              color: #666;
              margin-bottom: 5px;
            }
            
            .summary-card .value {
              font-size: 18px;
              font-weight: bold;
              color: #333;
            }
            
            .performance-metrics {
              display: grid;
              grid-template-columns: repeat(3, 1fr);
              gap: 15px;
              margin-bottom: 25px;
            }
            
            .metric {
              text-align: center;
              padding: 15px;
              background: #f8f9fa;
              border-radius: 8px;
            }
            
            .metric-value {
              font-size: 20px;
              font-weight: bold;
              color: #00A8FF;
            }
            
            .metric-label {
              font-size: 12px;
              color: #666;
              margin-top: 5px;
            }
            
            .portfolio-table {
              width: 100%;
              border-collapse: collapse;
              margin-bottom: 25px;
            }
            
            .portfolio-table th,
            .portfolio-table td {
              padding: 10px;
              text-align: left;
              border-bottom: 1px solid #ddd;
            }
            
            .portfolio-table th {
              background-color: #00A8FF;
              color: white;
              font-weight: bold;
              font-size: 12px;
            }
            
            .portfolio-table td {
              font-size: 11px;
            }
            
            .positive {
              color: #28a745;
              font-weight: bold;
            }
            
            .negative {
              color: #dc3545;
              font-weight: bold;
            }
            
            .footer {
              margin-top: 30px;
              padding-top: 20px;
              border-top: 1px solid #ddd;
              text-align: center;
              font-size: 10px;
              color: #666;
            }
            
            .disclaimer {
              background: #fff3cd;
              border: 1px solid #ffeaa7;
              border-radius: 5px;
              padding: 15px;
              margin-top: 20px;
            }
            
            .disclaimer h4 {
              color: #856404;
              margin-bottom: 10px;
              font-size: 14px;
            }
            
            .disclaimer p {
              color: #856404;
              font-size: 11px;
              line-height: 1.4;
            }

            .ai-summary-card {
              background: #e8f4fd;
              border: 1px solid #00A8FF;
              border-radius: 8px;
              padding: 20px;
              margin-bottom: 20px;
            }

            .ai-summary-card p {
              font-size: 12px;
              line-height: 1.6;
              color: #333;
              margin: 0 0 12px 0;
              text-align: justify;
            }

            .ai-summary-card p:last-child {
              margin-bottom: 0;
            }

            .ai-summary-card strong {
              font-weight: bold;
              color: #00A8FF;
            }

            .ai-summary-card em {
              font-style: italic;
              color: #555;
            }
            
            @media print {
              body {
                font-size: 12px;
              }
              
              .header {
                break-inside: avoid;
              }
              
              .summary-section {
                break-inside: avoid;
              }
              
              .portfolio-table {
                page-break-inside: avoid;
              }
            }
          </style>
        </head>
        <body>
          <div class="header">
            <div class="logo-section">
              ${
                logoBase64
                  ? `<img src="${logoBase64}" alt="WealthWise Logo" class="logo" />`
                  : ""
              }
              <div class="company-info">
                <h1>WealthWise</h1>
                <p>Investment Portfolio Analysis</p>
              </div>
            </div>
            <div class="report-info">
              <h2>Investment Summary Report</h2>
              <p>Generated on ${currentDate}</p>
              <p>For: ${userName}</p>
            </div>
          </div>

          <div class="summary-section">
            <h2 class="section-title">Portfolio Overview</h2>
            <div class="summary-grid">
              <div class="summary-card">
                <h3>Investment Goal</h3>
                <div class="value">${goal}</div>
              </div>
              <div class="summary-card">
                <h3>Target Amount</h3>
                <div class="value">${formatCurrency(targetValue)}</div>
              </div>
              <div class="summary-card">
                <h3>Time Horizon</h3>
                <div class="value">${timeframe}${
      typeof timeframe === "number" ? " years" : ""
    }</div>
              </div>
              <div class="summary-card">
                <h3>Risk Profile</h3>
                <div class="value">${riskLabel}</div>
              </div>
              <div class="summary-card">
                <h3>Risk Score</h3>
                <div class="value">${riskScore}${
      typeof riskScore === "number" ? "/100" : ""
    }</div>
              </div>
              <div class="summary-card">
                <h3>Annualized Return</h3>
                <div class="value ${
                  annualizedReturn >= 0 ? "positive" : "negative"
                }">
                  ${annualizedReturn >= 0 ? "+" : ""}${annualizedReturn.toFixed(
      2
    )}%
                </div>
              </div>
            </div>
          </div>

          <div class="summary-section">
            <h2 class="section-title">Performance Metrics</h2>
            <div class="performance-metrics">
              <div class="metric">
                <div class="metric-value">${formatCurrency(totalInvested)}</div>
                <div class="metric-label">Total Invested</div>
              </div>
              <div class="metric">
                <div class="metric-value">${formatCurrency(
                  portfolioValue
                )}</div>
                <div class="metric-label">Portfolio Value</div>
              </div>
              <div class="metric">
                <div class="metric-value ${
                  displayedReturn >= 0 ? "positive" : "negative"
                }">
                  ${formatPercentage(displayedReturn / 100)}
                </div>
                <div class="metric-label">Total Return</div>
              </div>
            </div>
          </div>

          ${portfolioHoldingsSection}

          ${aiSummarySection}

          ${goalAchievementSection}

          <div class="disclaimer">
            <h4>Important Disclaimer</h4>
            <p>
              This report is generated by WealthWise for educational and simulation purposes only. 
              The projections and analysis contained herein are based on historical data and mathematical models, 
              and should not be considered as financial advice or investment recommendations. 
              Past performance does not guarantee future results. Please consult with a qualified financial 
              advisor before making any investment decisions.
            </p>
          </div>

          <div class="footer">
            <p>© ${new Date().getFullYear()} WealthWise - Investment Portfolio Simulator</p>
            <p>This document was generated automatically and is for informational purposes only.</p>
          </div>
        </body>
      </html>
    `;
  };

  return (
    <div className="flex justify-center space-x-6 mt-8">
      <button
        onClick={handlePrint}
        className="bg-white text-[#00A8FF] font-bold px-6 py-3 border border-[#00A8FF] rounded-[15px] hover:bg-[#00A8FF] hover:text-white transition"
      >
        📄 Print Summary
      </button>
    </div>
  );
}
