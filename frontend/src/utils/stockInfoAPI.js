// utils/stockInfoAPI.js - Updated to use Python yfinance backend

class StockInfoAPI {
  constructor() {
    this.cache = new Map();
    this.cacheTimeout = 1000 * 60 * 15; // 15 minute cache (since yfinance is more reliable)
    this.baseURL = this.getBaseURL();
    this.instrumentsDB = null;
    this.fallbackData = this.initializeFallbackData();
  }

  getBaseURL() {
    // Automatically detect your backend URL
    if (typeof window !== "undefined") {
      const hostname = window.location.hostname;

      if (hostname === "localhost" || hostname === "127.0.0.1") {
        return "http://localhost:8000"; // Your Python backend port
      } else {
        // Your deployed backend URL
        return "https://your-python-backend.herokuapp.com"; // Update with your actual backend URL
      }
    }
    return "http://localhost:8000";
  }

  // Initialize fallback data (keeping your existing data)
  initializeFallbackData() {
    return {
      ARKK: {
        symbol: "ARKK",
        name: "ARK Innovation ETF",
        description:
          "Actively managed ETF that invests in companies developing technologies and services that potentially benefit from disruptive innovation.",
        sector: "Technology",
        industry: "Exchange Traded Fund",
        category: "high_growth",
        riskLevel: "Very High",
        price: 45.23,
        change: -0.89,
        changePercent: "-1.93",
        marketCap: 3500000000,
        volume: 1250000,
        dividendYield: 0,
        beta: 1.8,
        exchange: "NYSE Arca",
        country: "United States",
        currency: "USD",
        isETF: true,
        icon: "🚀",
        type: "ETF",
      },
      QQQ: {
        symbol: "QQQ",
        name: "Invesco QQQ Trust ETF",
        description:
          "Tracks the Nasdaq-100 Index, which includes 100 of the largest domestic and international non-financial companies.",
        sector: "Technology",
        industry: "Exchange Traded Fund",
        category: "technology",
        riskLevel: "High",
        price: 378.45,
        change: 2.31,
        changePercent: "0.61",
        marketCap: 180000000000,
        volume: 2800000,
        dividendYield: 0.6,
        beta: 1.2,
        exchange: "NASDAQ",
        country: "United States",
        currency: "USD",
        isETF: true,
        icon: "💻",
        type: "ETF",
      },
      // Add other symbols as needed...
    };
  }

  // Load instruments database from your Python backend
  async loadInstrumentsDatabase() {
    if (this.instrumentsDB) {
      return this.instrumentsDB;
    }

    try {
      console.log("📚 Loading instruments database from backend...");

      const response = await fetch(`${this.baseURL}/api/instruments`, {
        method: "GET",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        this.instrumentsDB = await response.json();
        console.log(
          `✅ Loaded ${Object.keys(this.instrumentsDB).length} instruments`
        );
        return this.instrumentsDB;
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      console.warn("⚠️ Failed to load instruments database:", error.message);
      console.log("📊 Using fallback data");
      this.instrumentsDB = this.fallbackData;
      return this.instrumentsDB;
    }
  }

  // Get current stock price from your Python backend
  async getCurrentPrice(symbol) {
    try {
      const response = await fetch(`${this.baseURL}/api/price/${symbol}`, {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        return data.price;
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.warn(
        `⚠️ Failed to get current price for ${symbol}:`,
        error.message
      );
      return null;
    }
  }

  // Get comprehensive stock info from your Python backend
  async getStockInfo(symbol) {
    const cacheKey = `stock_${symbol}`;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      console.log(`📋 Using cached data for ${symbol}`);
      return cached.data;
    }

    try {
      console.log(`📡 Fetching ${symbol} from Python backend...`);

      // Get current price and metadata from your backend
      const [priceResponse, stockResponse] = await Promise.allSettled([
        fetch(`${this.baseURL}/api/price/${symbol}`),
        fetch(`${this.baseURL}/api/stock/${symbol}`),
      ]);

      let stockInfo = null;

      // Process stock info response
      if (stockResponse.status === "fulfilled" && stockResponse.value.ok) {
        const stockData = await stockResponse.value.json();
        stockInfo = this.processBackendStockData(symbol, stockData);
      }

      // Add current price if available
      if (priceResponse.status === "fulfilled" && priceResponse.value.ok) {
        const priceData = await priceResponse.value.json();
        if (stockInfo) {
          stockInfo.price = priceData.price;
          stockInfo.lastUpdated = new Date().toISOString();
        }
      }

      // If backend failed, try instruments database
      if (!stockInfo) {
        const instrumentsDB = await this.loadInstrumentsDatabase();
        stockInfo = this.getInfoFromDatabase(symbol, instrumentsDB);
      }

      // Final fallback
      if (!stockInfo) {
        stockInfo = this.getFallbackInfo(symbol);
      }

      // Cache the result
      this.cache.set(cacheKey, {
        data: stockInfo,
        timestamp: Date.now(),
      });

      console.log(`✅ Successfully fetched ${symbol}`);
      return stockInfo;
    } catch (error) {
      console.error(`❌ Error fetching ${symbol}:`, error);
      return this.getFallbackInfo(symbol);
    }
  }

  // Process data from your Python backend
  processBackendStockData(symbol, backendData) {
    // Your Python script returns comprehensive data
    const info = backendData.info || {};
    const metadata = backendData.metadata || {};

    return {
      symbol: symbol.toUpperCase(),
      name: backendData.name || info.longName || `${symbol} Investment`,
      description:
        backendData.description ||
        info.longBusinessSummary ||
        `Investment in ${symbol}`,
      sector: metadata.sector || info.sector || "Unknown",
      industry: metadata.industry || info.industry || "Unknown",
      category: backendData.category || this.categorizeFromBackend(info),
      riskLevel: backendData.risk || this.assessRiskFromBackend(info),
      price:
        metadata.currentPrice ||
        info.currentPrice ||
        info.regularMarketPrice ||
        0,
      change: this.calculateChange(metadata.currentPrice, info.previousClose),
      changePercent: this.calculateChangePercent(
        metadata.currentPrice,
        info.previousClose
      ),
      volume: info.volume || info.regularMarketVolume || 0,
      marketCap: metadata.marketCap || info.marketCap || 0,
      dividendYield: metadata.dividendYield || info.dividendYield || 0,
      beta: metadata.beta || info.beta || 1.0,
      peRatio: info.trailingPE || null,
      exchange: metadata.exchange || info.exchange || "Unknown",
      country: metadata.country || info.country || "United States",
      currency: metadata.currency || info.currency || "USD",
      isETF: info.quoteType === "ETF" || this.isETF(symbol),
      icon:
        backendData.icon || this.getIconForSymbol(symbol, backendData.category),
      type: backendData.type || (info.quoteType === "ETF" ? "ETF" : "Stock"),
      lastUpdated: new Date().toISOString(),
      dataSource: "python_backend",
    };
  }

  // Get info from instruments database
  getInfoFromDatabase(symbol, database) {
    const dbEntry = database[symbol];
    if (!dbEntry) return null;

    return {
      symbol: symbol.toUpperCase(),
      name: dbEntry.name,
      description: dbEntry.description,
      category: dbEntry.category,
      riskLevel: dbEntry.risk,
      icon: dbEntry.icon,
      type: dbEntry.type,
      sector: dbEntry.metadata?.sector || "Unknown",
      industry: dbEntry.metadata?.industry || "Unknown",
      price: dbEntry.metadata?.currentPrice || 0,
      change: 0,
      changePercent: "0.00",
      volume: 0,
      marketCap: dbEntry.metadata?.marketCap || 0,
      dividendYield: dbEntry.metadata?.dividendYield || 0,
      beta: dbEntry.metadata?.beta || 1.0,
      peRatio: null,
      exchange: dbEntry.metadata?.exchange || "Unknown",
      country: dbEntry.metadata?.country || "United States",
      currency: dbEntry.metadata?.currency || "USD",
      isETF: dbEntry.type === "ETF",
      lastUpdated: dbEntry.metadata?.lastUpdated || new Date().toISOString(),
      dataSource: "instruments_database",
    };
  }

  // Search instruments using your Python backend
  async searchInstruments(query, limit = 10) {
    try {
      const response = await fetch(
        `${this.baseURL}/api/search?query=${encodeURIComponent(
          query
        )}&limit=${limit}`,
        {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
        }
      );

      if (response.ok) {
        const results = await response.json();
        return results.map((item) => ({
          symbol: item.symbol,
          name: item.name,
          description: item.description,
          category: item.category,
          risk: item.risk,
          icon: item.icon,
          type: item.type,
        }));
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.warn("⚠️ Search failed:", error.message);

      // Fallback to local search
      const instrumentsDB = await this.loadInstrumentsDatabase();
      return this.searchLocal(query, instrumentsDB, limit);
    }
  }

  // Local search fallback
  searchLocal(query, database, limit) {
    const queryLower = query.toLowerCase();
    const results = [];

    for (const [symbol, data] of Object.entries(database)) {
      const searchText =
        `${symbol} ${data.name} ${data.category} ${data.type}`.toLowerCase();

      if (searchText.includes(queryLower)) {
        results.push({
          symbol,
          name: data.name,
          description: data.description,
          category: data.category,
          risk: data.risk,
          icon: data.icon,
          type: data.type,
        });
      }

      if (results.length >= limit) break;
    }

    return results;
  }

  // Get multiple stocks at once
  async getBatchStockInfo(symbols) {
    console.log(
      `🔄 Fetching stock information for ${symbols.length} symbols...`
    );

    try {
      // Try batch endpoint first
      const response = await fetch(`${this.baseURL}/api/stocks/batch`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ symbols }),
      });

      if (response.ok) {
        const batchData = await response.json();
        const results = {};

        for (const symbol of symbols) {
          if (batchData[symbol]) {
            results[symbol] = this.processBackendStockData(
              symbol,
              batchData[symbol]
            );
          } else {
            results[symbol] = this.getFallbackInfo(symbol);
          }
        }

        console.log(
          `✅ Successfully fetched batch data for ${
            Object.keys(results).length
          } stocks`
        );
        return results;
      } else {
        throw new Error(`Batch request failed: HTTP ${response.status}`);
      }
    } catch (error) {
      console.warn(
        "⚠️ Batch request failed, falling back to individual requests:",
        error.message
      );

      // Fallback to individual requests
      const results = {};
      const promises = symbols.map(async (symbol) => {
        try {
          const info = await this.getStockInfo(symbol);
          return { symbol, info };
        } catch (error) {
          console.error(`Failed to fetch ${symbol}:`, error);
          return { symbol, info: this.getFallbackInfo(symbol) };
        }
      });

      const stockInfoArray = await Promise.all(promises);
      stockInfoArray.forEach(({ symbol, info }) => {
        results[symbol] = info;
      });

      console.log(
        `✅ Successfully fetched info for ${Object.keys(results).length} stocks`
      );
      return results;
    }
  }

  // Helper methods
  calculateChange(currentPrice, previousClose) {
    if (!currentPrice || !previousClose) return 0;
    return parseFloat((currentPrice - previousClose).toFixed(2));
  }

  calculateChangePercent(currentPrice, previousClose) {
    if (!currentPrice || !previousClose || previousClose === 0) return "0.00";
    const changePercent =
      ((currentPrice - previousClose) / previousClose) * 100;
    return changePercent.toFixed(2);
  }

  categorizeFromBackend(info) {
    // Convert backend categories to your frontend format
    const categoryMap = {
      bonds: "bonds",
      dividend_stocks: "dividend_stocks",
      utilities: "utilities",
      large_cap_growth: "large_cap_growth",
      broad_market: "broad_market",
      international: "international",
      emerging_markets: "emerging_markets",
      technology: "technology",
      high_growth: "high_growth",
      reits: "reits",
      financials: "financials",
    };

    const backendCategory = info.category;
    return categoryMap[backendCategory] || "other";
  }

  assessRiskFromBackend(info) {
    const riskMap = {
      Low: "Low",
      "Low-Medium": "Low-Medium",
      Medium: "Medium",
      "Medium-High": "Medium-High",
      High: "High",
      "Very High": "Very High",
    };

    return riskMap[info.risk] || "Medium";
  }

  getIconForSymbol(symbol, category) {
    const specialIcons = {
      AAPL: "📱",
      MSFT: "💻",
      GOOGL: "🔍",
      AMZN: "📦",
      TSLA: "🚗",
      NVDA: "🤖",
      META: "📘",
      V: "💳",
      MA: "💳",
      JNJ: "💊",
      KO: "🥤",
      PEP: "🥤",
    };

    if (specialIcons[symbol]) return specialIcons[symbol];

    const categoryIcons = {
      bonds: "🏛️",
      dividend_stocks: "💰",
      utilities: "⚡",
      large_cap_growth: "📈",
      broad_market: "📊",
      international: "🌍",
      emerging_markets: "🌏",
      technology: "💻",
      high_growth: "🚀",
      reits: "🏢",
      financials: "🏦",
    };

    return categoryIcons[category] || "📊";
  }

  isETF(symbol) {
    const etfSymbols = [
      "QQQ",
      "SPY",
      "IWM",
      "EEM",
      "VTI",
      "VOO",
      "VGT",
      "VUG",
      "ARKK",
      "ARKQ",
      "IBB",
      "FINX",
      "VWO",
      "XLK",
      "XLF",
      "XLE",
    ];
    return etfSymbols.includes(symbol.toUpperCase()) || symbol.length === 3;
  }

  getFallbackInfo(symbol) {
    console.log(`📊 Using fallback data for ${symbol}`);

    const fallback = this.fallbackData[symbol];
    if (fallback) {
      return {
        ...fallback,
        lastUpdated: new Date().toISOString(),
        dataSource: "fallback",
      };
    }

    // Generate fallback
    const basePrice = 50 + Math.random() * 150;
    const changePercent = (Math.random() - 0.5) * 6;
    const change = basePrice * (changePercent / 100);

    return {
      symbol: symbol.toUpperCase(),
      name: `${symbol} Investment`,
      description: "Investment instrument - live data temporarily unavailable",
      category: this.isETF(symbol) ? "broad_market" : "large_cap_growth",
      riskLevel: "Medium",
      sector: "Unknown",
      industry: "Unknown",
      price: parseFloat(basePrice.toFixed(2)),
      change: parseFloat(change.toFixed(2)),
      changePercent: changePercent.toFixed(2),
      volume: Math.floor(Math.random() * 1000000) + 100000,
      marketCap: Math.floor(Math.random() * 10000000000) + 1000000000,
      dividendYield: 0,
      beta: 1.0,
      exchange: "Unknown",
      country: "United States",
      currency: "USD",
      isETF: this.isETF(symbol),
      icon: "📊",
      type: this.isETF(symbol) ? "ETF" : "Stock",
      lastUpdated: new Date().toISOString(),
      dataSource: "generated_fallback",
    };
  }

  // Test connection to your Python backend
  async testConnection() {
    console.log("🔍 Testing Python backend connection...");

    const results = {
      backend_health: false,
      instruments_db: false,
      price_api: false,
      search_api: false,
    };

    try {
      // Test health endpoint
      const healthResponse = await fetch(`${this.baseURL}/health`);
      results.backend_health = healthResponse.ok;
    } catch (error) {
      console.warn("Backend health check failed:", error.message);
    }

    try {
      // Test instruments database
      const instrumentsResponse = await fetch(
        `${this.baseURL}/api/instruments`
      );
      results.instruments_db = instrumentsResponse.ok;
    } catch (error) {
      console.warn("Instruments DB test failed:", error.message);
    }

    try {
      // Test price API
      const priceResponse = await fetch(`${this.baseURL}/api/price/AAPL`);
      results.price_api = priceResponse.ok;
    } catch (error) {
      console.warn("Price API test failed:", error.message);
    }

    try {
      // Test search API
      const searchResponse = await fetch(
        `${this.baseURL}/api/search?query=apple&limit=5`
      );
      results.search_api = searchResponse.ok;
    } catch (error) {
      console.warn("Search API test failed:", error.message);
    }

    console.log("📊 Backend connection test results:", results);
    return results;
  }

  // Utility methods
  clearCache() {
    this.cache.clear();
    console.log("📋 Cache cleared");
  }

  getCacheStats() {
    return {
      size: this.cache.size,
      entries: Array.from(this.cache.keys()),
    };
  }
}

// Enhanced portfolio function
export async function enhancePortfolioWithAPI(portfolioData) {
  const api = new StockInfoAPI();
  const stocks = portfolioData?.results?.stocks_picked || [];

  if (stocks.length === 0) {
    console.log("📊 No stocks to enhance");
    return portfolioData;
  }

  try {
    console.log("🔄 Enhancing portfolio with Python backend...");

    // Test connection first
    await api.testConnection();

    const symbols = stocks.map((stock) => stock.symbol);
    const stockInfo = await api.getBatchStockInfo(symbols);

    const enhancedStocks = stocks.map((stock) => ({
      ...stock,
      ...stockInfo[stock.symbol],
      allocation: stock.allocation,
      explanation:
        stock.explanation || `${stock.symbol} selected for your portfolio`,
    }));

    console.log("✅ Portfolio enhancement completed with Python backend");

    return {
      ...portfolioData,
      results: {
        ...portfolioData.results,
        stocks_picked: enhancedStocks,
        api_enhanced: true,
        api_source: "python_yfinance_backend",
        last_api_update: new Date().toISOString(),
      },
    };
  } catch (error) {
    console.error("❌ Portfolio enhancement failed:", error);
    return portfolioData;
  }
}

// Create global instance
if (typeof window !== "undefined") {
  window.stockAPI = new StockInfoAPI();
}

export default StockInfoAPI;
