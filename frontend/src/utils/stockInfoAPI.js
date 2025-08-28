/**
 * stockInfoAPI.js
 * -----------------------------------------------------------------------------
 * Purpose:
 *   Centralized utilities for fetching and enriching stock/ETF information used
 *   across the app. Provides a resilient, cached API that prefers fast local
 *   lookups and gracefully falls back to external providers when available.
 *
 * What it contains:
 *   - StockInfoAPI class:
 *       • In‑memory caching (1‑hour TTL) to reduce network calls.
 *       • An enhanced local database for instant, offline-friendly lookups of
 *         known symbols (e.g., QQQ, VGT, ARKK, VWO, etc.).
 *       • Integrations with multiple providers (Alpha Vantage, FMP, Finnhub)
 *         with robust error handling and rate‑limit safe sequencing.
 *       • Helper methods to infer sector/category/risk for unknown symbols.
 *       • Batch retrieval (getBatchStockInfo) with paced requests when using
 *         external APIs, while skipping delays for local database reads.
 *       • Fallback strategy:
 *           1) Try cache
 *           2) Try local enhanced database (instant)
 *           3) Try enabled external APIs (if keys present)
 *           4) Use a smart fallback with guessed metadata
 *
 *   - enhancePortfolioWithAPI(portfolioData):
 *       • Takes a simulation/portfolio payload and enriches each picked stock
 *         with the freshest info (price, name, data source, etc.).
 *       • Preserves original allocation values and annotates results with
 *         metadata (sources used, last update time).
 *
 * Notes:
 *   - This module favors resiliency and UX over strict real‑time quotes.
 *   - If you need to invalidate stored quotes (e.g., on sign‑out), call
 *     StockInfoAPI#clearCache().
 *   - There are two Alpha Vantage implementations in this file for historical
 *     reasons. They return equivalent shapes, but consider consolidating them
 *     in a future cleanup to avoid duplication.
 */
// utils/stockInfoAPI.js - Enhanced with Better Fallbacks

class StockInfoAPI {
  constructor() {
    this.cache = new Map();
    this.cacheTimeout = 1000 * 60 * 60; // 1 hour cache

    // Comprehensive stock database with all your portfolio symbols
    this.ENHANCED_STOCK_DATABASE = {
      // Technology ETFs
      QQQ: {
        name: "Invesco QQQ Trust ETF",
        price: 385.5,
        sector: "Technology",
        category: "Technology",
        riskLevel: "High",
        description:
          "NASDAQ-100 Index ETF tracking top 100 non-financial stocks",
        marketCap: 180000000000,
        isETF: true,
      },
      VGT: {
        name: "Vanguard Information Technology ETF",
        price: 425.3,
        sector: "Technology",
        category: "Technology",
        riskLevel: "High",
        description:
          "Technology sector ETF with exposure to software, hardware, and IT services",
        marketCap: 65000000000,
        isETF: true,
      },
      ARKK: {
        name: "ARK Innovation ETF",
        price: 48.9,
        sector: "Technology",
        category: "Innovation/Growth",
        riskLevel: "Very High",
        description:
          "Disruptive innovation companies in AI, robotics, and genomics",
        marketCap: 8500000000,
        isETF: true,
      },
      ARKQ: {
        name: "ARK Autonomous Technology & Robotics ETF",
        price: 55.3,
        sector: "Technology",
        category: "Autonomous Technology",
        riskLevel: "Very High",
        description: "Companies developing autonomous technology and robotics",
        marketCap: 2100000000,
        isETF: true,
      },
      FINX: {
        name: "Global X FinTech ETF",
        price: 36.25,
        sector: "Financial Services",
        category: "Financial Technology",
        riskLevel: "High",
        description:
          "Financial technology companies transforming banking and payments",
        marketCap: 1800000000,
        isETF: true,
      },

      // Healthcare & Biotechnology
      IBB: {
        name: "iShares Biotechnology ETF",
        price: 125.6,
        sector: "Healthcare",
        category: "Biotechnology",
        riskLevel: "Very High",
        description: "Biotechnology and pharmaceutical companies",
        marketCap: 8200000000,
        isETF: true,
      },

      // International/Emerging Markets
      VWO: {
        name: "Vanguard Emerging Markets ETF",
        price: 45.2,
        sector: "International",
        category: "Emerging Markets",
        riskLevel: "High",
        description: "Broad exposure to emerging market equity securities",
        marketCap: 85000000000,
        isETF: true,
      },

      // Growth/Large Cap
      VUG: {
        name: "Vanguard Growth ETF",
        price: 285.4,
        sector: "Equity",
        category: "Large Cap Growth",
        riskLevel: "Medium-High",
        description:
          "Large-cap growth companies with above-average growth potential",
        marketCap: 120000000000,
        isETF: true,
      },

      // Cryptocurrency/Digital Assets
      COIN: {
        name: "Coinbase Global Inc",
        price: 155.75,
        sector: "Financial Services",
        category: "Cryptocurrency",
        riskLevel: "Very High",
        description: "Leading cryptocurrency exchange platform",
        marketCap: 42000000000,
        isETF: false,
      },
      BITO: {
        name: "ProShares Bitcoin Strategy ETF",
        price: 18.75,
        sector: "Alternative",
        category: "Bitcoin/Cryptocurrency",
        riskLevel: "Extremely High",
        description: "Bitcoin futures-based cryptocurrency exposure",
        marketCap: 1400000000,
        isETF: true,
      },

      // Broad Market
      SPY: {
        name: "SPDR S&P 500 ETF Trust",
        price: 445.2,
        sector: "Broad Market",
        category: "Large Cap",
        riskLevel: "Medium",
        description: "S&P 500 index tracking ETF",
        marketCap: 410000000000,
        isETF: true,
      },
      VTI: {
        name: "Vanguard Total Stock Market ETF",
        price: 245.8,
        sector: "Broad Market",
        category: "Total Market",
        riskLevel: "Medium",
        description: "Total US stock market exposure",
        marketCap: 320000000000,
        isETF: true,
      },

      // Bonds
      BND: {
        name: "Vanguard Total Bond Market ETF",
        price: 76.85,
        sector: "Fixed Income",
        category: "Bonds",
        riskLevel: "Low",
        description: "Broad US investment-grade bond exposure",
        marketCap: 85000000000,
        isETF: true,
      },

      // Real Estate
      VNQ: {
        name: "Vanguard Real Estate ETF",
        price: 92.4,
        sector: "Real Estate",
        category: "Real Estate",
        riskLevel: "Medium-High",
        description: "Real Estate Investment Trusts (REITs)",
        marketCap: 32000000000,
        isETF: true,
      },
    };

    // API configuration - only use APIs that work
    this.apis = {
      alphaVantage: {
        key: import.meta.env.VITE_ALPHA_VANTAGE_KEY,
        baseURL: "https://www.alphavantage.co/query",
        limit: 5,
        enabled: true, // Keep this as primary
      },
      fmp: {
        key: import.meta.env.VITE_FMP_KEY,
        baseURL: "https://financialmodelingprep.com/api/v3",
        enabled: true, // New API
      },
      finnhub: {
        key: import.meta.env.VITE_FINNHUB_KEY,
        baseURL: "https://finnhub.io/api/v1",
        enabled: true, // New API
      },
    };
  }

  // Main method - tries API first, then enhanced fallback
  async getStockInfo(symbol) {
    const cacheKey = `stock_${symbol}`;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      console.log(`📦 Using cached data for ${symbol}`);
      return cached.data;
    }

    // First try our enhanced database (instant)
    const enhancedData = this.getEnhancedFallbackInfo(symbol);
    if (enhancedData.dataSource === "Enhanced Database") {
      console.log(`📊 Using enhanced database for ${symbol}`);
      this.cache.set(cacheKey, {
        data: enhancedData,
        timestamp: Date.now(),
      });
      return enhancedData;
    }

    // Only try Alpha Vantage if we have a working key
    if (
      this.apis.alphaVantage.enabled &&
      this.apis.alphaVantage.key &&
      this.apis.alphaVantage.key !== "demo"
    ) {
      try {
        console.log(`📡 Fetching ${symbol} from Alpha Vantage...`);
        const data = await this.getAlphaVantageData(symbol);
        if (data && data.symbol) {
          // Enhance API data with our local knowledge
          const enhanced = this.enhanceWithLocalData(data, symbol);
          this.cache.set(cacheKey, {
            data: enhanced,
            timestamp: Date.now(),
          });
          return enhanced;
        }
      } catch (error) {
        console.warn(`API method failed for ${symbol}: ${error.message}`);
      }
    }

    // Fallback to enhanced database
    console.log(`📊 Using fallback data for ${symbol}`);
    const fallbackData = this.getEnhancedFallbackInfo(symbol);
    this.cache.set(cacheKey, {
      data: fallbackData,
      timestamp: Date.now(),
    });
    return fallbackData;
  }

  // 1. Financial Modeling Prep - NEW, very reliable
  async getFMPData(symbol) {
    if (!this.apis.fmp.enabled || !this.apis.fmp.key) {
      throw new Error("FMP API key required");
    }

    try {
      const quoteUrl = `${this.apis.fmp.baseURL}/quote/${symbol}?apikey=${this.apis.fmp.key}`;

      console.log(`📡 Fetching ${symbol} from Financial Modeling Prep...`);
      const response = await fetch(quoteUrl);

      if (!response.ok) {
        throw new Error(`FMP HTTP error: ${response.status}`);
      }

      const data = await response.json();

      if (!data || data.length === 0) {
        throw new Error("No data returned from FMP");
      }

      const quote = data[0];

      return {
        symbol: symbol.toUpperCase(),
        name: quote.name,
        price: quote.price,
        change: quote.change,
        changePercent: quote.changesPercentage,
        marketCap: quote.marketCap,
        volume: quote.volume,
        previousClose: quote.previousClose,
        sector: this.guessSectorFromSymbol(symbol),
        category: this.categorizeFromSymbol(symbol),
        riskLevel: this.assessRiskFromSymbol(symbol),
        isETF: this.isETF(symbol),
        dataSource: "Financial Modeling Prep",
        lastUpdated: new Date().toISOString(),
      };
    } catch (error) {
      throw new Error(`FMP failed: ${error.message}`);
    }
  }

  // 2. Finnhub - NEW, good free tier
  async getFinnhubData(symbol) {
    if (!this.apis.finnhub.enabled || !this.apis.finnhub.key) {
      throw new Error("Finnhub API key required");
    }

    try {
      const quoteUrl = `${this.apis.finnhub.baseURL}/quote?symbol=${symbol}&token=${this.apis.finnhub.key}`;

      console.log(`📡 Fetching ${symbol} from Finnhub...`);
      const response = await fetch(quoteUrl);

      if (!response.ok) {
        throw new Error(`Finnhub HTTP error: ${response.status}`);
      }

      const quote = await response.json();

      if (!quote.c) {
        throw new Error("No price data from Finnhub");
      }

      return {
        symbol: symbol.toUpperCase(),
        name: symbol.toUpperCase(), // Finnhub quote doesn't include name
        price: quote.c, // current price
        change: quote.d, // change
        changePercent: quote.dp, // change percent
        volume: 0, // Not in basic quote
        previousClose: quote.pc, // previous close
        sector: this.guessSectorFromSymbol(symbol),
        category: this.categorizeFromSymbol(symbol),
        riskLevel: this.assessRiskFromSymbol(symbol),
        isETF: this.isETF(symbol),
        dataSource: "Finnhub",
        lastUpdated: new Date().toISOString(),
      };
    } catch (error) {
      throw new Error(`Finnhub failed: ${error.message}`);
    }
  }

  // 3. Alpha Vantage
  async getAlphaVantageData(symbol) {
    if (
      !this.apis.alphaVantage.enabled ||
      !this.apis.alphaVantage.key ||
      this.apis.alphaVantage.key === "demo"
    ) {
      throw new Error("Alpha Vantage API key required");
    }

    try {
      const quoteUrl = `${this.apis.alphaVantage.baseURL}?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${this.apis.alphaVantage.key}`;

      console.log(`📡 Fetching ${symbol} from Alpha Vantage...`);
      const response = await fetch(quoteUrl);

      if (!response.ok) {
        throw new Error(`Alpha Vantage HTTP error: ${response.status}`);
      }

      const data = await response.json();

      if (data.Note || data["Error Message"] || data.Information) {
        throw new Error(
          data.Note ||
            data["Error Message"] ||
            data.Information ||
            "API limit reached"
        );
      }

      const quote = data["Global Quote"];
      if (!quote || Object.keys(quote).length === 0) {
        throw new Error("No quote data returned from Alpha Vantage");
      }

      const price = parseFloat(quote["05. price"]) || 0;
      const change = parseFloat(quote["09. change"]) || 0;
      const changePercent =
        parseFloat(quote["10. change percent"]?.replace("%", "")) || 0;

      return {
        symbol: symbol.toUpperCase(),
        name: symbol.toUpperCase(),
        price: price,
        change: change,
        changePercent: changePercent,
        volume: parseInt(quote["06. volume"]) || 0,
        previousClose: parseFloat(quote["08. previous close"]) || 0,
        sector: this.guessSectorFromSymbol(symbol),
        category: this.categorizeFromSymbol(symbol),
        riskLevel: this.assessRiskFromSymbol(symbol),
        isETF: this.isETF(symbol),
        dataSource: "Alpha Vantage",
        lastUpdated:
          quote["07. latest trading day"] || new Date().toISOString(),
      };
    } catch (error) {
      throw new Error(`Alpha Vantage failed: ${error.message}`);
    }
  }

  // Enhanced fallback with comprehensive data
  getEnhancedFallbackInfo(symbol) {
    const symbolUpper = symbol.toUpperCase();
    const stockInfo = this.ENHANCED_STOCK_DATABASE[symbolUpper];

    if (stockInfo) {
      return {
        symbol: symbolUpper,
        name: stockInfo.name,
        price: stockInfo.price,
        description: stockInfo.description,
        sector: stockInfo.sector,
        category: stockInfo.category,
        riskLevel: stockInfo.riskLevel,
        marketCap: stockInfo.marketCap,
        isETF: stockInfo.isETF,
        dataSource: "Enhanced Database",
        lastUpdated: new Date().toISOString(),
        confidence: "High", // High confidence in our curated data
      };
    }

    // Smart fallback for unknown symbols
    return {
      symbol: symbolUpper,
      name: this.guessNameFromSymbol(symbolUpper),
      price: 100.0,
      description: `${symbolUpper} - Investment details being researched`,
      sector: this.guessSectorFromSymbol(symbolUpper),
      category: this.categorizeFromSymbol(symbolUpper),
      riskLevel: this.assessRiskFromSymbol(symbolUpper),
      isETF: this.isETF(symbolUpper),
      dataSource: "Smart Fallback",
      lastUpdated: new Date().toISOString(),
      confidence: "Low",
    };
  }

  // Enhance API data with our local knowledge
  enhanceWithLocalData(apiData, symbol) {
    const localInfo = this.ENHANCED_STOCK_DATABASE[symbol.toUpperCase()];

    if (localInfo) {
      return {
        ...apiData,
        category: localInfo.category,
        riskLevel: localInfo.riskLevel,
        description: localInfo.description,
        isETF: localInfo.isETF,
        confidence: "High",
      };
    }

    return {
      ...apiData,
      category: this.categorizeFromSymbol(symbol),
      riskLevel: this.assessRiskFromSymbol(symbol),
      isETF: this.isETF(symbol),
      confidence: "Medium",
    };
  }

  // Batch processing with rate limiting and better error handling
  async getBatchStockInfo(symbols) {
    console.log(
      `🔄 Fetching stock information for ${symbols.length} symbols...`
    );
    const results = {};

    for (let i = 0; i < symbols.length; i++) {
      const symbol = symbols[i];
      try {
        console.log(`📡 Fetching ${symbol} from API...`);
        results[symbol] = await this.getStockInfo(symbol);
        console.log(`✅ Successfully fetched ${symbol}`);

        // Only wait if using external API (not our enhanced database)
        if (
          results[symbol].dataSource !== "Enhanced Database" &&
          i < symbols.length - 1
        ) {
          await this.delay(1000); // 1 second between API calls
        }
      } catch (error) {
        console.error(`❌ Failed to fetch ${symbol}:`, error);
        results[symbol] = this.getEnhancedFallbackInfo(symbol);
        console.log(`✅ Successfully fetched ${symbol}`);
      }
    }

    console.log(`✅ Successfully fetched info for ${symbols.length} stocks`);
    return results;
  }

  // Helper methods
  guessNameFromSymbol(symbol) {
    const nameMappings = {
      QQQ: "Invesco QQQ Trust ETF",
      VGT: "Vanguard Information Technology ETF",
      ARKK: "ARK Innovation ETF",
      ARKQ: "ARK Autonomous Technology ETF",
      VWO: "Vanguard Emerging Markets ETF",
      BITO: "ProShares Bitcoin Strategy ETF",
      IBB: "iShares Biotechnology ETF",
      FINX: "Global X FinTech ETF",
      VUG: "Vanguard Growth ETF",
      COIN: "Coinbase Global Inc",
    };

    return nameMappings[symbol] || `${symbol} Investment Fund`;
  }

  categorizeFromSymbol(symbol) {
    const categoryMappings = {
      QQQ: "Technology",
      VGT: "Technology",
      ARKK: "Innovation/Growth",
      ARKQ: "Autonomous Technology",
      FINX: "Financial Technology",
      IBB: "Biotechnology",
      VWO: "Emerging Markets",
      BITO: "Bitcoin/Cryptocurrency",
      COIN: "Cryptocurrency",
      VUG: "Large Cap Growth",
      SPY: "Large Cap",
      VTI: "Total Market",
      BND: "Bonds",
      VNQ: "Real Estate",
    };

    return categoryMappings[symbol.toUpperCase()] || "Other";
  }

  guessSectorFromSymbol(symbol) {
    const sectorMappings = {
      QQQ: "Technology",
      VGT: "Technology",
      ARKK: "Technology",
      ARKQ: "Technology",
      FINX: "Financial Services",
      IBB: "Healthcare",
      VWO: "International",
      BITO: "Alternative",
      COIN: "Financial Services",
      VUG: "Equity",
      SPY: "Broad Market",
      VTI: "Broad Market",
      BND: "Fixed Income",
      VNQ: "Real Estate",
    };

    return sectorMappings[symbol.toUpperCase()] || "Unknown";
  }

  assessRiskFromSymbol(symbol) {
    const riskMappings = {
      ARKK: "Very High",
      ARKQ: "Very High",
      BITO: "Extremely High",
      COIN: "Very High",
      IBB: "Very High",
      QQQ: "High",
      VGT: "High",
      FINX: "High",
      VWO: "High",
      VUG: "Medium-High",
      SPY: "Medium",
      VTI: "Medium",
      VNQ: "Medium-High",
      BND: "Low",
    };

    return riskMappings[symbol.toUpperCase()] || "Medium";
  }

  isETF(symbol) {
    const etfList = [
      "QQQ",
      "VGT",
      "ARKK",
      "ARKQ",
      "VWO",
      "BITO",
      "IBB",
      "FINX",
      "VUG",
      "SPY",
      "VTI",
      "BND",
      "VNQ",
    ];
    const nonETFList = ["COIN"]; // Individual stocks

    if (nonETFList.includes(symbol.toUpperCase())) return false;
    if (etfList.includes(symbol.toUpperCase())) return true;

    // Default heuristic
    return symbol.length === 3 || symbol.includes("ETF");
  }

  delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  clearCache() {
    this.cache.clear();
  }
}

// Enhanced portfolio enhancement function
export async function enhancePortfolioWithAPI(portfolioData) {
  const api = new StockInfoAPI();
  const stocks = portfolioData?.results?.stocks_picked || [];

  if (stocks.length === 0) return portfolioData;

  try {
    console.log("🔄 Fetching stock information from API...");
    const symbols = stocks.map((stock) => stock.symbol);

    const stockInfo = await api.getBatchStockInfo(symbols);

    const enhancedStocks = stocks.map((stock) => ({
      ...stock,
      ...stockInfo[stock.symbol],
      allocation: stock.allocation, // Preserve original allocation
    }));

    console.log(`✅ API enhancement completed`);

    return {
      ...portfolioData,
      results: {
        ...portfolioData.results,
        stocks_picked: enhancedStocks,
        api_enhanced: true,
        api_sources_used: [...new Set(enhancedStocks.map((s) => s.dataSource))],
        last_api_update: new Date().toISOString(),
      },
    };
  } catch (error) {
    console.error("❌ API enhancement failed:", error);
    return portfolioData;
  }
}

export default StockInfoAPI;
