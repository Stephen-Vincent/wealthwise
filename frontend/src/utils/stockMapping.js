// utils/stockMapping.js - Add this file to properly categorize your stocks

export const STOCK_DATABASE = {
  // Technology & Innovation
  QQQ: {
    name: "Invesco QQQ Trust",
    category: "Technology",
    description: "NASDAQ-100 Index ETF - Top 100 non-financial NASDAQ stocks",
    riskLevel: "Medium-High",
  },
  VUG: {
    name: "Vanguard Growth ETF",
    category: "Large Cap Growth",
    description:
      "Large-cap growth companies with above-average growth potential",
    riskLevel: "Medium-High",
  },
  ARKK: {
    name: "ARK Innovation ETF",
    category: "Technology",
    description:
      "Disruptive innovation companies in AI, robotics, and genomics",
    riskLevel: "Very High",
  },
  ARKQ: {
    name: "ARK Autonomous Technology & Robotics ETF",
    category: "Technology",
    description: "Companies in autonomous technology and robotics",
    riskLevel: "Very High",
  },
  FINX: {
    name: "Global X FinTech ETF",
    category: "Technology",
    description: "Financial technology companies transforming finance",
    riskLevel: "High",
  },
  SKYY: {
    name: "First Trust Cloud Computing ETF",
    category: "Technology",
    description:
      "Companies providing cloud computing services and infrastructure",
    riskLevel: "High",
  },

  // Emerging Markets
  VWO: {
    name: "Vanguard Emerging Markets ETF",
    category: "Emerging Markets",
    description: "Broad exposure to emerging market equity securities",
    riskLevel: "High",
  },

  // Healthcare & Biotechnology
  IBB: {
    name: "iShares Biotechnology ETF",
    category: "Healthcare",
    description: "Biotechnology and pharmaceutical companies",
    riskLevel: "Very High",
  },

  // Digital Assets / Cryptocurrency
  BITO: {
    name: "ProShares Bitcoin Strategy ETF",
    category: "Digital Assets",
    description: "Bitcoin futures-based cryptocurrency exposure",
    riskLevel: "Extremely High",
  },

  // Add more common stocks for completeness
  VTI: {
    name: "Vanguard Total Stock Market ETF",
    category: "Broad Market",
    description: "Total US stock market exposure",
    riskLevel: "Medium",
  },
  SPY: {
    name: "SPDR S&P 500 ETF",
    category: "Large Cap",
    description: "S&P 500 index tracking",
    riskLevel: "Medium",
  },
  BND: {
    name: "Vanguard Total Bond Market ETF",
    category: "Bonds",
    description: "Broad US investment-grade bond exposure",
    riskLevel: "Low",
  },
};

export const CATEGORY_INFO = {
  Technology: {
    description: "Technology and innovation-focused investments",
    riskProfile: "Medium-High to Very High",
    color: "#3B82F6",
    icon: "💻",
  },
  "Large Cap Growth": {
    description: "Established companies with strong growth potential",
    riskProfile: "Medium-High",
    color: "#10B981",
    icon: "📈",
  },
  "Emerging Markets": {
    description: "Higher-growth developing country investments",
    riskProfile: "High",
    color: "#F59E0B",
    icon: "🌍",
  },
  Healthcare: {
    description: "Healthcare and biotechnology companies",
    riskProfile: "High to Very High",
    color: "#EF4444",
    icon: "🏥",
  },
  "Digital Assets": {
    description: "Cryptocurrency and digital asset exposure",
    riskProfile: "Extremely High",
    color: "#8B5CF6",
    icon: "₿",
  },
  "Broad Market": {
    description: "Diversified market exposure",
    riskProfile: "Medium",
    color: "#6B7280",
    icon: "📊",
  },
  "Large Cap": {
    description: "Large established company investments",
    riskProfile: "Medium",
    color: "#059669",
    icon: "🏢",
  },
  Bonds: {
    description: "Fixed income and bond investments",
    riskProfile: "Low",
    color: "#0D9488",
    icon: "📋",
  },
};

// Function to enhance portfolio data with proper stock information
export function enhanceStockData(stocksArray) {
  if (!Array.isArray(stocksArray)) return stocksArray;

  return stocksArray.map((stock) => {
    const symbol = stock.symbol?.toUpperCase();
    const stockInfo = STOCK_DATABASE[symbol];

    if (stockInfo) {
      return {
        ...stock,
        name: stockInfo.name,
        category: stockInfo.category,
        description: stockInfo.description,
        riskLevel: stockInfo.riskLevel,
        categoryInfo: CATEGORY_INFO[stockInfo.category],
      };
    }

    // Return original stock if not found in database
    return {
      ...stock,
      category: "Other",
      description: "Investment details not available",
      riskLevel: "Unknown",
      categoryInfo: {
        description: "Uncategorized investment",
        riskProfile: "Unknown",
        color: "#9CA3AF",
        icon: "❓",
      },
    };
  });
}

// Function to categorize stocks by category
export function categorizeStocks(stocksArray) {
  const enhancedStocks = enhanceStockData(stocksArray);
  const categories = {};

  enhancedStocks.forEach((stock) => {
    const category = stock.category || "Other";

    if (!categories[category]) {
      categories[category] = {
        stocks: [],
        totalAllocation: 0,
        info: stock.categoryInfo ||
          CATEGORY_INFO[category] || {
            description: "Uncategorized investments",
            riskProfile: "Unknown",
            color: "#9CA3AF",
            icon: "❓",
          },
      };
    }

    categories[category].stocks.push(stock);
    categories[category].totalAllocation += stock.allocation || 0;
  });

  return categories;
}
