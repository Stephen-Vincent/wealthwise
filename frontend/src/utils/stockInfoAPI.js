// utils/stockInfoAPI.js - Multi-Source Stock Data API

class StockInfoAPI {
  constructor() {
    this.cache = new Map();
    this.cacheTimeout = 1000 * 60 * 60; // 1 hour cache

    // API Keys (get free keys from these services)
    this.apis = {
      // 1. Alpha Vantage - Most reliable free option
      alphaVantage: {
        key: import.meta.env.VITE_ALPHA_VANTAGE_KEY || "demo",
        baseURL: "https://www.alphavantage.co/query",
        limit: 5, // 5 calls per minute (free tier)
      },

      // 2. Polygon.io - Great free tier
      polygon: {
        key: import.meta.env.VITE_POLYGON_KEY,
        baseURL: "https://api.polygon.io",
        limit: 5, // 5 calls per minute (free tier)
      },

      // 3. Yahoo Finance (unofficial but works)
      yahoo: {
        baseURL: "https://query1.finance.yahoo.com/v8/finance/chart",
        limit: 100, // Very generous
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

    // Try different APIs in order of preference
    const methods = [
      () => this.getYahooFinanceData(symbol), // Try Yahoo first (most reliable)
      () => this.getAlphaVantageData(symbol), // Then Alpha Vantage
      () => this.getPolygonData(symbol), // Then Polygon
      () => this.getTwelveDataInfo(symbol), // Then Twelve Data
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

  // 1. Yahoo Finance API (Free, no key required)
  async getYahooFinanceData(symbol) {
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
  }

  // 2. Alpha Vantage API (Free tier: 5 calls/minute)
  async getAlphaVantageData(symbol) {
    if (!this.apis.alphaVantage.key || this.apis.alphaVantage.key === "demo") {
      throw new Error("Alpha Vantage API key required");
    }

    try {
      // Get overview data
      const overviewUrl = `${this.apis.alphaVantage.baseURL}?function=OVERVIEW&symbol=${symbol}&apikey=${this.apis.alphaVantage.key}`;
      const quoteUrl = `${this.apis.alphaVantage.baseURL}?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${this.apis.alphaVantage.key}`;

      const [overviewResponse, quoteResponse] = await Promise.all([
        fetch(overviewUrl),
        fetch(quoteUrl),
      ]);

      const overview = await overviewResponse.json();
      const quote = await quoteResponse.json();

      if (overview.Note || quote.Note) {
        throw new Error("API rate limit exceeded");
      }

      const globalQuote = quote["Global Quote"] || {};

      return {
        symbol: symbol.toUpperCase(),
        name: overview.Name || symbol,
        description: overview.Description,
        price: parseFloat(globalQuote["05. price"]) || null,
        sector: overview.Sector,
        industry: overview.Industry,
        marketCap: parseInt(overview.MarketCapitalization) || null,
        dividendYield: parseFloat(overview.DividendYield) * 100 || 0,
        beta: parseFloat(overview.Beta) || null,
        peRatio: parseFloat(overview.PERatio) || null,
        exchange: overview.Exchange,
        country: overview.Country,
        category: this.categorizeFromSector(overview.Sector, overview.Industry),
        riskLevel: this.assessRiskFromBeta(parseFloat(overview.Beta)),
        isETF: overview.AssetType === "ETF" || this.isETF(symbol),
        dataSource: "Alpha Vantage",
        lastUpdated: new Date().toISOString(),
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
        console.log(`📡 Fetching ${symbol} from Python backend...`);
        results[symbol] = await this.getStockInfo(symbol);
        console.log(`✅ Successfully fetched ${symbol}`);

        // Rate limiting - wait between requests
        if (i < symbols.length - 1) {
          await this.delay(100); // 100ms between requests
        }
      } catch (error) {
        console.error(`❌ Failed to fetch ${symbol}:`, error);
        results[symbol] = this.getFallbackInfo(symbol);
      }
    }

    console.log(`✅ Successfully fetched info for ${symbols.length} stocks`);
    return results;
  }

  // Helper methods for categorization
  categorizeFromSymbol(symbol) {
    const etfMappings = {
      QQQ: "Technology",
      SPY: "Broad Market",
      VTI: "Broad Market",
      VGT: "Technology",
      ARKK: "Innovation/Growth",
      ARKQ: "Autonomous Technology",
      VWO: "Emerging Markets",
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

  categorizeFromSector(sector, industry) {
    if (!sector) return "Other";

    const sectorMappings = {
      Technology: "Technology",
      Healthcare: "Healthcare",
      Financial: "Financial Services",
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

  assessRiskFromBeta(beta) {
    if (!beta || beta === null) return "Medium";

    if (beta > 1.5) return "Very High";
    if (beta > 1.2) return "High";
    if (beta > 0.8) return "Medium-High";
    if (beta > 0.6) return "Medium";
    return "Low";
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
        sector: "Technology",
        category: "Technology",
        riskLevel: "High",
      },
      VGT: {
        name: "Vanguard Information Technology ETF",
        sector: "Technology",
        category: "Technology",
        riskLevel: "High",
      },
      ARKK: {
        name: "ARK Innovation ETF",
        sector: "Technology",
        category: "Innovation/Growth",
        riskLevel: "Very High",
      },
      ARKQ: {
        name: "ARK Autonomous Technology & Robotics ETF",
        sector: "Technology",
        category: "Autonomous Technology",
        riskLevel: "Very High",
      },
      VWO: {
        name: "Vanguard Emerging Markets ETF",
        sector: "International",
        category: "Emerging Markets",
        riskLevel: "High",
      },
      VUG: {
        name: "Vanguard Growth ETF",
        sector: "Equity",
        category: "Large Cap Growth",
        riskLevel: "Medium-High",
      },
      COIN: {
        name: "Coinbase Global Inc",
        sector: "Financial Services",
        category: "Cryptocurrency",
        riskLevel: "Very High",
      },
      BITO: {
        name: "ProShares Bitcoin Strategy ETF",
        sector: "Alternative",
        category: "Bitcoin/Cryptocurrency",
        riskLevel: "Very High",
      },
      IBB: {
        name: "iShares Biotechnology ETF",
        sector: "Healthcare",
        category: "Biotechnology",
        riskLevel: "High",
      },
      FINX: {
        name: "Global X FinTech ETF",
        sector: "Financial Services",
        category: "Financial Technology",
        riskLevel: "High",
      },
    };

    const fallback = fallbackData[symbol.toUpperCase()] || {};

    return {
      symbol: symbol.toUpperCase(),
      name: fallback.name || symbol,
      description:
        fallback.description ||
        "Investment instrument - enhanced data unavailable",
      sector: fallback.sector || "Unknown",
      category: fallback.category || this.categorizeFromSymbol(symbol),
      riskLevel: fallback.riskLevel || "Medium",
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
  const api = new MultiSourceStockAPI();
  const stocks = portfolioData?.results?.stocks_picked || [];

  if (stocks.length === 0) return portfolioData;

  try {
    console.log("🔄 Fetching stock information from API...");
    const symbols = stocks.map((stock) => stock.symbol);
    console.log(
      `🔄 Fetching stock information for ${symbols.length} symbols...`
    );

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
