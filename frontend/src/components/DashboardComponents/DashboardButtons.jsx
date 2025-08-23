/**
 * DashboardButtons Component
 *
 * This component provides utility buttons for the dashboard view, specifically:
 * - Fetches and displays the user's name from localStorage or via an API call.
 * - Generates a detailed investment summary report (Portfolio Overview, Performance Metrics,
 *   Holdings, AI Analysis, and Goal Achievement).
 * - Dynamically converts the report into printable HTML, embedding the WealthWise logo,
 *   formatted portfolio data, and an AI-generated analysis section.
 * - Supports one-click printing by creating a hidden iframe, loading the report, and triggering print.
 *
 * Usage: Displayed on the dashboard to allow users to generate and print a professional
 * investment summary of their portfolio.
 */

import { useState, useEffect } from "react";
import { usePortfolio } from "../../context/PortfolioContext";
import logo from "../../assets/wealthwise.png";

export default function DashboardButtons() {
  const { portfolioData, apiCall, token } = usePortfolio();
  const [userName, setUserName] = useState("Investor");

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

      // Try API call using the context's apiCall method
      if (token && apiCall) {
        try {
          const userData = await apiCall(`/users/${userId}`);
          if (userData.name) {
            setUserName(userData.name);
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
  }, [portfolioData?.user_id, token, apiCall]);

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

    // Extract data based on actual data structure
    console.log("🔍 EXTRACTING DATA FROM PORTFOLIO:");
    console.log("Full portfolioData:", portfolioData);

    // Basic portfolio information - using actual field names from your data
    const goal = portfolioData?.goal || "Investment Goal";
    const timeframe = portfolioData?.timeframe || "Not specified";
    const riskLabel = portfolioData?.risk_label || "Not specified";
    const riskScore = portfolioData?.risk_score || "Not calculated";
    const targetValue = portfolioData?.target_value || 0;

    // Financial performance data - calculate total invested correctly
    const performanceMetrics = portfolioData?.performance_metrics || {};

    // Calculate total invested: lump sum + monthly contributions over timeframe
    const lumpSum = portfolioData?.lump_sum || 0;
    const monthlyContribution = portfolioData?.monthly || 0;
    const timeframeYears = parseInt(timeframe) || 0;
    const monthlyContributions = monthlyContribution * timeframeYears * 12;

    const totalInvested = lumpSum + monthlyContributions;

    const portfolioValue =
      performanceMetrics?.ending_value || portfolioData?.final_balance || 0;
    const rawReturn =
      performanceMetrics?.total_return || portfolioData?.total_return || 0;
    const displayedReturn = rawReturn; // Already a percentage
    const annualizedReturn = performanceMetrics?.annualized_return || 0;

    console.log("📊 EXTRACTED VALUES:");
    console.log("Goal:", goal);
    console.log("Target Value:", targetValue);
    console.log("Timeframe:", timeframe);
    console.log("Risk Label:", riskLabel);
    console.log("Risk Score:", riskScore);
    console.log("Total Invested:", totalInvested);
    console.log("Portfolio Value:", portfolioValue);
    console.log("Raw Return:", rawReturn);
    console.log("Displayed Return:", displayedReturn);
    console.log("Annualized Return:", annualizedReturn.toFixed(2) + "%");

    // Handle stocks data - using stocks_picked from actual data
    const stocks = portfolioData?.stocks_picked || portfolioData?.stocks || [];
    console.log("📈 STOCKS DATA:", stocks);

    // Enhanced sector mapping
    const getSectorForStock = (stock) => {
      if (stock.sector) return stock.sector;
      if (stock.industry) return stock.industry;
      if (stock.category) return stock.category;
      if (stock.asset_class) return stock.asset_class;

      const symbol = stock.symbol || stock.ticker || "";
      const name = stock.name || stock.company_name || stock.description || "";

      // ETF mappings
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
          stock.allocation ||
          stock.weight ||
          stock.percentage ||
          stock.percent ||
          0;
        const sector = getSectorForStock(stock);
        const symbol = stock.symbol || stock.ticker || stock.code || "N/A";
        const name =
          stock.name ||
          stock.company_name ||
          stock.description ||
          stock.security_name ||
          "N/A";

        return `
        <tr>
          <td><strong>${symbol}</strong></td>
          <td>${name}</td>
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

    // AI Summary - check ai_analysis.summary since ai_summary is null in your data
    const aiSummary =
      portfolioData?.ai_analysis?.summary ||
      portfolioData?.ai_summary ||
      "Investment analysis not available.";

    const formatAISummary = (text) => {
      if (!text) return "";

      // First, clean up any existing HTML class artifacts
      let cleanText = text
        .replace(/"analysis-[^"]*">/g, "") // Remove class name artifacts
        .replace(/analysis-[a-z]+>/g, "") // Remove remaining class fragments
        .trim();

      // Convert markdown-style formatting to HTML
      let htmlText = cleanText
        // Headers
        .replace(/^##\s+(.+)$/gm, '<h3 class="analysis-heading">$1</h3>')
        .replace(/^#\s+(.+)$/gm, '<h3 class="analysis-heading">$1</h3>')

        // Bold text (handle ** patterns)
        .replace(/\*\*(.*?)\*\*/g, '<strong class="analysis-bold">$1</strong>')

        // Italic text
        .replace(/\*([^*\n]+)\*/g, '<em class="analysis-italic">$1</em>')

        // Bullet points
        .replace(/^\s*[\*\-•]\s+(.+)$/gm, '<li class="analysis-bullet">$1</li>')

        // Quotes
        .replace(
          /[""]([^""]+)[""]?/g,
          '<span class="analysis-quote">"$1"</span>'
        )

        // Numbered points
        .replace(
          /^(\d+\.)\s+(.+)$/gm,
          '<div class="analysis-numbered"><strong>$1</strong> $2</div>'
        );

      // Split into sections by double line breaks
      const sections = htmlText.split(/\n\s*\n+/);
      let formattedText = "";

      sections.forEach((section) => {
        section = section.trim();
        if (!section) return;

        // Check if section contains bullet points
        if (section.includes('<li class="analysis-bullet">')) {
          const bulletItems = section.split("\n").filter((line) => line.trim());
          const listItems = bulletItems
            .map((item) =>
              item.includes('<li class="analysis-bullet">')
                ? item
                : `<li class="analysis-bullet">${item}</li>`
            )
            .join("");
          formattedText += `<ul class="analysis-list">${listItems}</ul>`;
        }
        // Check if it's a heading or numbered item
        else if (
          section.includes("<h3") ||
          section.includes('<div class="analysis-numbered">')
        ) {
          formattedText += section;
        }
        // Regular paragraph
        else {
          formattedText += `<div class="analysis-paragraph">${section}</div>`;
        }
      });

      // Clean up any remaining artifacts and normalize whitespace
      formattedText = formattedText
        .replace(/\s+/g, " ")
        .replace(/>\s+</g, "><")
        .replace(/analysis-[a-z]+"?>/g, "") // Final cleanup of any remaining class fragments
        .trim();

      return (
        formattedText || `<div class="analysis-paragraph">${cleanText}</div>`
      );
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
    const targetAchieved =
      portfolioData?.target_achieved !== undefined
        ? portfolioData.target_achieved
        : portfolioValue >= targetValue;

    const goalAchievementSection =
      targetValue > 0
        ? `
      <div class="summary-section">
        <h2 class="section-title">Goal Achievement</h2>
        <div class="summary-card">
          <h3>Target Achievement Status</h3>
          <div class="value ${targetAchieved ? "positive" : "negative"}">
            ${targetAchieved ? "✓ Target Achieved" : "✗ Target Not Achieved"}
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

            .ai-summary-card .analysis-heading {
              font-size: 16px;
              font-weight: bold;
              color: #00A8FF;
              margin: 20px 0 10px 0;
              padding-bottom: 5px;
              border-bottom: 2px solid #00A8FF;
            }

            .ai-summary-card .analysis-heading:first-child {
              margin-top: 0;
            }

            .ai-summary-card .analysis-subheading {
              font-size: 14px;
              font-weight: bold;
              color: #0056b3;
              margin: 15px 0 8px 0;
            }

            .ai-summary-card .analysis-paragraph {
              font-size: 12px;
              line-height: 1.6;
              color: #333;
              margin: 0 0 12px 0;
              text-align: justify;
            }

            .ai-summary-card .analysis-numbered {
              font-size: 12px;
              line-height: 1.6;
              color: #333;
              margin: 8px 0;
              padding: 8px 12px;
              background: #f8f9fa;
              border-left: 3px solid #00A8FF;
              border-radius: 4px;
            }

            .ai-summary-card .analysis-list {
              list-style: none;
              padding: 0;
              margin: 10px 0;
            }

            .ai-summary-card .analysis-bullet {
              font-size: 12px;
              line-height: 1.5;
              color: #333;
              margin: 6px 0;
              padding: 4px 0 4px 20px;
              position: relative;
            }

            .ai-summary-card .analysis-bullet:before {
              content: "•";
              color: #00A8FF;
              font-weight: bold;
              position: absolute;
              left: 8px;
            }

            .ai-summary-card .analysis-bold {
              font-weight: bold;
              color: #00A8FF;
            }

            .ai-summary-card .analysis-italic {
              font-style: italic;
              color: #555;
            }

            .ai-summary-card .analysis-quote {
              font-style: italic;
              color: #666;
              background: #f0f9ff;
              padding: 2px 6px;
              border-radius: 3px;
              border-left: 2px solid #00A8FF;
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
                <div class="value">${timeframe} years</div>
              </div>
              <div class="summary-card">
                <h3>Risk Profile</h3>
                <div class="value">${riskLabel}</div>
              </div>
              <div class="summary-card">
                <h3>Risk Score</h3>
                <div class="value">${riskScore}/100</div>
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
        disabled={!portfolioData}
      >
        📄 Print Summary
      </button>
    </div>
  );
}
