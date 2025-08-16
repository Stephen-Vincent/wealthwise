// utils/stockInfoAPI.js - Multi-Source Stock Data API (FIXED VERSION)

class StockInfoAPI {
  constructor() {
    this.cache = new Map();
    this.cacheTimeout = 1000 * 60 * 60; // 1 hour cache

    // API Keys - using your actual Alpha Vantage key
    this.apis = {
      // 1. Alpha Vantage - Most reliable free option
      alphaVantage: {
        key: import.meta.env.VITE_ALPHA_VANTAGE_KEY || "DQ44FHC7WY8ODBKQ", // Your actual key
        baseURL: "https://www.alphavantage.co/query",
        limit: 5, // 5 calls per minute (free tier)
      },

      // 2. Polygon.io - Great free tier
      polygon: {
        key: import.meta.env.VITE_POLYGON_KEY,
        baseURL: "https://api.polygon.io",
        limit: 5, // 5 calls per minute (free tier)
      },

      // 3. Yahoo Finance (unofficial but works) - DISABLED DUE TO CORS
      yahoo: {
        baseURL: "https://query1.finance.yahoo.com/v8/finance/chart",
        limit: 100, // Very generous
        disabled: true, // Disable due to CORS issues
      },

      // 4. Twelve Data - Good free tier
      twelveData: {
        key: import.meta.env.VITE_TWELVE_DATA_KEY,
        baseURL: "https://api.twelvedata.com",
        limit: 8, // 8 calls per minute (free tier)
      },
    };
  }

  // Main method - tries multiple sources
  async getStockInfo(symbol) {
    const cacheKey = `stock_${symbol}`;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      console.log(`📦 Using cached data for ${symbol}`);
      return cached.data;
    }

    // Try different APIs in order of preference (skip Yahoo due to CORS)
    const methods = [
      () => this.getAlphaVantageData(symbol), // Start with Alpha Vantage (you have key)
      () => this.getPolygonData(symbol), // Then Polygon
      () => this.getTwelveDataInfo(symbol), // Then Twelve Data
      // () => this.getYahooFinanceData(symbol), // Skip Yahoo (CORS issues)
    ];

    for (const method of methods) {
      try {
        const data = await method();
        if (data && data.symbol) {
          this.cache.set(cacheKey, {
            data,
            timestamp: Date.now(),
          });
          return data;
        }
      } catch (error) {
        console.warn(`API method failed for ${symbol}:`, error.message);
      }
    }

    // If all APIs fail, return fallback data
    console.log(`📊 Using fallback data for ${symbol}`);
    return this.getFallbackInfo(symbol);
  }

  // 1. Yahoo Finance API (DISABLED DUE TO CORS - kept for reference)
  async getYahooFinanceData(symbol) {
    throw new Error("Yahoo Finance disabled due to CORS policy");

    // This would work from a backend but not from browser due to CORS
    /*
    try {
      const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}`;
      const response = await fetch(url);

      if (!response.ok) throw new Error(`Yahoo API error: ${response.status}`);

      const data = await response.json();
      const result = data.chart?.result?.[0];

      if (!result) throw new Error("No data returned");

      const meta = result.meta;
      const quote = result.indicators?.quote?.[0];

      return {
        symbol: symbol.toUpperCase(),
        name: meta.longName || meta.shortName || symbol,
        price: meta.regularMarketPrice || meta.previousClose,
        currency: meta.currency || "USD",
        exchange: meta.exchangeName || meta.fullExchangeName,
        marketCap: meta.marketCap,
        sector: this.guessSectorFromName(meta.longName || symbol),
        category: this.categorizeFromSymbol(symbol),
        riskLevel: this.assessRiskFromBeta(meta.beta),
        dividendYield: (meta.dividendYield || 0) * 100,
        beta: meta.beta,
        peRatio: meta.trailingPE,
        isETF: this.isETF(symbol),
        dataSource: "Yahoo Finance",
        lastUpdated: new Date().toISOString(),
      };
    } catch (error) {
      throw new Error(`Yahoo Finance failed: ${error.message}`);
    }
    */
  }

  // 2. Alpha Vantage API (Free tier: 5 calls/minute) - FIXED
  async getAlphaVantageData(symbol) {
    if (!this.apis.alphaVantage.key || this.apis.alphaVantage.key === "demo") {
      throw new Error("Alpha Vantage API key required");
    }

    try {
      // Start with just the quote API to avoid rate limits
      const quoteUrl = `${this.apis.alphaVantage.baseURL}?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${this.apis.alphaVantage.key}`;

      console.log(`📡 Fetching ${symbol} from Alpha Vantage...`);
      const quoteResponse = await fetch(quoteUrl);

      if (!quoteResponse.ok) {
        throw new Error(`Alpha Vantage HTTP error: ${quoteResponse.status}`);
      }

      const quote = await quoteResponse.json();

      if (quote.Note || quote["Error Message"]) {
        throw new Error(
          quote.Note || quote["Error Message"] || "API rate limit or error"
        );
      }

      const globalQuote = quote["Global Quote"];

      if (!globalQuote || Object.keys(globalQuote).length === 0) {
        throw new Error("No quote data returned from Alpha Vantage");
      }

      // Parse the response
      const price = parseFloat(globalQuote["05. price"]) || 0;
      const change = parseFloat(globalQuote["09. change"]) || 0;
      const changePercent =
        parseFloat(globalQuote["10. change percent"].replace("%", "")) || 0;

      return {
        symbol: symbol.toUpperCase(),
        name: symbol.toUpperCase(), // Alpha Vantage doesn't return name in GLOBAL_QUOTE
        price: price,
        change: change,
        changePercent: changePercent,
        volume: parseInt(globalQuote["06. volume"]) || 0,
        previousClose: parseFloat(globalQuote["08. previous close"]) || 0,
        sector: this.guessSectorFromSymbol(symbol), // Fallback categorization
        category: this.categorizeFromSymbol(symbol),
        riskLevel: this.assessRiskFromSymbol(symbol),
        isETF: this.isETF(symbol),
        dataSource: "Alpha Vantage",
        lastUpdated:
          globalQuote["07. latest trading day"] || new Date().toISOString(),
      };
    } catch (error) {
      throw new Error(`Alpha Vantage failed: ${error.message}`);
    }
  }

  // 3. Polygon.io API (Free tier: 5 calls/minute)
  async getPolygonData(symbol) {
    if (!this.apis.polygon.key) {
      throw new Error("Polygon API key required");
    }

    try {
      const tickerUrl = `${this.apis.polygon.baseURL}/v3/reference/tickers/${symbol}?apikey=${this.apis.polygon.key}`;
      const quoteUrl = `${this.apis.polygon.baseURL}/v2/last/trade/${symbol}?apikey=${this.apis.polygon.key}`;

      const [tickerResponse, quoteResponse] = await Promise.all([
        fetch(tickerUrl),
        fetch(quoteUrl),
      ]);

      const tickerData = await tickerResponse.json();
      const quoteData = await quoteResponse.json();

      if (tickerData.status !== "OK") {
        throw new Error("Invalid ticker or API error");
      }

      const ticker = tickerData.results;
      const lastTrade = quoteData.results;

      return {
        symbol: symbol.toUpperCase(),
        name: ticker.name,
        description: ticker.description,
        price: lastTrade?.p || null,
        marketCap: ticker.market_cap,
        sector: ticker.sic_description,
        exchange: ticker.primary_exchange,
        category: this.categorizeFromSymbol(symbol),
        riskLevel: "Medium", // Polygon doesn't provide beta easily
        isETF: ticker.type === "ETF",
        dataSource: "Polygon.io",
        lastUpdated: new Date().toISOString(),
      };
    } catch (error) {
      throw new Error(`Polygon failed: ${error.message}`);
    }
  }

  // 4. Twelve Data API (Free tier: 8 calls/minute)
  async getTwelveDataInfo(symbol) {
    if (!this.apis.twelveData.key) {
      throw new Error("Twelve Data API key required");
    }

    try {
      const url = `${this.apis.twelveData.baseURL}/profile?symbol=${symbol}&apikey=${this.apis.twelveData.key}`;
      const response = await fetch(url);
      const data = await response.json();

      if (data.status === "error") {
        throw new Error(data.message);
      }

      return {
        symbol: symbol.toUpperCase(),
        name: data.name,
        description: data.summary,
        sector: data.sector,
        industry: data.industry,
        country: data.country,
        exchange: data.exchange,
        category: this.categorizeFromSector(data.sector, data.industry),
        riskLevel: "Medium",
        isETF: data.type === "ETF",
        dataSource: "Twelve Data",
        lastUpdated: new Date().toISOString(),
      };
    } catch (error) {
      throw new Error(`Twelve Data failed: ${error.message}`);
    }
  }

  // Batch processing with rate limiting
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

        // Rate limiting - wait between requests to avoid hitting limits
        if (i < symbols.length - 1) {
          await this.delay(1000); // 1 second between requests for Alpha Vantage
        }
      } catch (error) {
        console.error(`❌ Failed to fetch ${symbol}:`, error);
        results[symbol] = this.getFallbackInfo(symbol);
      }
    }

    console.log(`✅ Successfully fetched info for ${symbols.length} stocks`);
    return results;
  }

  // Helper methods for categorization (FIXED - these were missing)
  categorizeFromSymbol(symbol) {
    const etfMappings = {
      QQQ: "Technology",
      SPY: "Broad Market",
      VTI: "Broad Market",
      VGT: "Technology",
      ARKK: "Innovation/Growth",
      ARKQ: "Autonomous Technology",
      VWO: "Emerging Markets",
      EEM: "Emerging Markets", // Added EEM
      BND: "Bonds",
      VUG: "Large Cap Growth",
      VNQ: "Real Estate",
      COIN: "Cryptocurrency",
      BITO: "Bitcoin/Cryptocurrency",
      IBB: "Biotechnology",
      FINX: "Financial Technology",
    };

    return etfMappings[symbol.toUpperCase()] || this.categorizeFromName(symbol);
  }

  // FIXED: Added missing categorizeFromName function
  categorizeFromName(name) {
    if (!name || typeof name !== "string") return "Other";

    const nameLower = name.toLowerCase();

    // Technology
    if (
      nameLower.includes("technology") ||
      nameLower.includes("tech") ||
      nameLower.includes("software") ||
      nameLower.includes("internet")
    ) {
      return "Technology";
    }

    // Healthcare/Biotech
    if (
      nameLower.includes("healthcare") ||
      nameLower.includes("biotech") ||
      nameLower.includes("pharmaceutical") ||
      nameLower.includes("medical")
    ) {
      return "Healthcare";
    }

    // Financial
    if (
      nameLower.includes("financial") ||
      nameLower.includes("bank") ||
      nameLower.includes("fintech") ||
      nameLower.includes("insurance")
    ) {
      return "Financial Services";
    }

    // Energy
    if (
      nameLower.includes("energy") ||
      nameLower.includes("oil") ||
      nameLower.includes("gas") ||
      nameLower.includes("renewable")
    ) {
      return "Energy";
    }

    // Real Estate
    if (
      nameLower.includes("real estate") ||
      nameLower.includes("reit") ||
      nameLower.includes("property")
    ) {
      return "Real Estate";
    }

    // Crypto
    if (
      nameLower.includes("bitcoin") ||
      nameLower.includes("crypto") ||
      nameLower.includes("blockchain")
    ) {
      return "Cryptocurrency";
    }

    // Emerging Markets
    if (
      nameLower.includes("emerging") ||
      nameLower.includes("international") ||
      nameLower.includes("global")
    ) {
      return "Emerging Markets";
    }

    return "Other";
  }

  categorizeFromSector(sector, industry) {
    if (!sector) return "Other";

    const sectorMappings = {
      Technology: "Technology",
      Healthcare: "Healthcare",
      Financial: "Financial Services",
      "Financial Services": "Financial Services",
      "Consumer Discretionary": "Consumer Discretionary",
      "Consumer Staples": "Consumer Staples",
      Energy: "Energy",
      Industrials: "Industrials",
      Utilities: "Utilities",
      "Real Estate": "Real Estate",
      Materials: "Materials",
      "Communication Services": "Technology",
    };

    return sectorMappings[sector] || sector;
  }

  guessSectorFromName(name) {
    if (!name) return "Unknown";

    const nameLower = name.toLowerCase();

    if (nameLower.includes("technology") || nameLower.includes("tech"))
      return "Technology";
    if (nameLower.includes("healthcare") || nameLower.includes("biotech"))
      return "Healthcare";
    if (nameLower.includes("financial") || nameLower.includes("bank"))
      return "Financial Services";
    if (nameLower.includes("energy") || nameLower.includes("oil"))
      return "Energy";
    if (nameLower.includes("real estate") || nameLower.includes("reit"))
      return "Real Estate";
    if (nameLower.includes("utility") || nameLower.includes("utilities"))
      return "Utilities";

    return "Unknown";
  }

  // FIXED: Added missing guessSectorFromSymbol function
  guessSectorFromSymbol(symbol) {
    const symbolMappings = {
      // Technology ETFs
      QQQ: "Technology",
      VGT: "Technology",
      ARKK: "Technology",
      ARKQ: "Technology",
      FINX: "Financial Services",

      // Healthcare
      IBB: "Healthcare",

      // International/Emerging
      VWO: "International",
      EEM: "International",

      // Broad Market
      SPY: "Broad Market",
      VTI: "Broad Market",
      VUG: "Broad Market",

      // Crypto
      BITO: "Alternative",
      COIN: "Financial Services",

      // Bonds
      BND: "Fixed Income",

      // Real Estate
      VNQ: "Real Estate",
    };

    return symbolMappings[symbol.toUpperCase()] || "Unknown";
  }

  assessRiskFromBeta(beta) {
    if (!beta || beta === null) return "Medium";

    if (beta > 1.5) return "Very High";
    if (beta > 1.2) return "High";
    if (beta > 0.8) return "Medium-High";
    if (beta > 0.6) return "Medium";
    return "Low";
  }

  // FIXED: Added missing assessRiskFromSymbol function
  assessRiskFromSymbol(symbol) {
    const riskMappings = {
      // Very High Risk
      ARKK: "Very High",
      ARKQ: "Very High",
      BITO: "Very High",
      COIN: "Very High",

      // High Risk
      QQQ: "High",
      VGT: "High",
      IBB: "High",
      VWO: "High",
      EEM: "High",
      FINX: "High",

      // Medium-High Risk
      VUG: "Medium-High",

      // Medium Risk
      SPY: "Medium",
      VTI: "Medium",
      VNQ: "Medium",

      // Low Risk
      BND: "Low",
    };

    return riskMappings[symbol.toUpperCase()] || "Medium";
  }

  isETF(symbol) {
    const commonETFs = [
      "QQQ",
      "SPY",
      "VTI",
      "VGT",
      "ARKK",
      "ARKQ",
      "VWO",
      "EEM",
      "BND",
      "VUG",
      "VNQ",
      "BITO",
      "IBB",
      "FINX",
      "VEA",
      "VTEB",
      "AGG",
      "IWM",
    ];

    return (
      commonETFs.includes(symbol.toUpperCase()) ||
      symbol.length === 3 ||
      symbol.includes("ETF")
    );
  }

  getFallbackInfo(symbol) {
    // Enhanced fallback with better guessing
    const fallbackData = {
      QQQ: {
        name: "Invesco QQQ Trust ETF",
        price: 385.5,
        sector: "Technology",
        category: "Technology",
        riskLevel: "High",
      },
      VGT: {
        name: "Vanguard Information Technology ETF",
        price: 425.3,
        sector: "Technology",
        category: "Technology",
        riskLevel: "High",
      },
      ARKK: {
        name: "ARK Innovation ETF",
        price: 48.9,
        sector: "Technology",
        category: "Innovation/Growth",
        riskLevel: "Very High",
      },
      ARKQ: {
        name: "ARK Autonomous Technology & Robotics ETF",
        price: 55.3,
        sector: "Technology",
        category: "Autonomous Technology",
        riskLevel: "Very High",
      },
      VWO: {
        name: "Vanguard Emerging Markets ETF",
        price: 45.2,
        sector: "International",
        category: "Emerging Markets",
        riskLevel: "High",
      },
      EEM: {
        name: "iShares MSCI Emerging Markets ETF",
        price: 42.85,
        sector: "International",
        category: "Emerging Markets",
        riskLevel: "High",
      },
      VUG: {
        name: "Vanguard Growth ETF",
        price: 285.4,
        sector: "Equity",
        category: "Large Cap Growth",
        riskLevel: "Medium-High",
      },
      COIN: {
        name: "Coinbase Global Inc",
        price: 155.75,
        sector: "Financial Services",
        category: "Cryptocurrency",
        riskLevel: "Very High",
      },
      BITO: {
        name: "ProShares Bitcoin Strategy ETF",
        price: 18.75,
        sector: "Alternative",
        category: "Bitcoin/Cryptocurrency",
        riskLevel: "Very High",
      },
      IBB: {
        name: "iShares Biotechnology ETF",
        price: 125.6,
        sector: "Healthcare",
        category: "Biotechnology",
        riskLevel: "High",
      },
      FINX: {
        name: "Global X FinTech ETF",
        price: 36.25,
        sector: "Financial Services",
        category: "Financial Technology",
        riskLevel: "High",
      },
    };

    const fallback = fallbackData[symbol.toUpperCase()] || {};

    return {
      symbol: symbol.toUpperCase(),
      name: fallback.name || symbol,
      price: fallback.price || 100.0,
      description:
        fallback.description ||
        "Investment instrument - enhanced data unavailable",
      sector: fallback.sector || this.guessSectorFromSymbol(symbol),
      category: fallback.category || this.categorizeFromSymbol(symbol),
      riskLevel: fallback.riskLevel || this.assessRiskFromSymbol(symbol),
      isETF: this.isETF(symbol),
      dataSource: "Enhanced Fallback",
      lastUpdated: new Date().toISOString(),
    };
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
  const api = new StockInfoAPI(); // Fixed class name
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
