// utils/stockInfoAPI.js - Financial Modeling Prep Integration

class StockInfoAPI {
  constructor() {
    // Get free API key at: https://financialmodelingprep.com/developer/docs
    this.apiKey = import.meta.env.VITE_FMP_API_KEY || "demo"; // Use 'demo' for testing
    this.baseURL = "https://financialmodelingprep.com/api/v3";
    this.cache = new Map();
    this.cacheTimeout = 1000 * 60 * 60; // 1 hour cache
  }

  // Get comprehensive stock/ETF information
  async getStockInfo(symbol) {
    const cacheKey = `stock_${symbol}`;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }

    try {
      // Get multiple data points in parallel
      const [profile, quote, ratios] = await Promise.all([
        this.fetchCompanyProfile(symbol),
        this.fetchQuote(symbol),
        this.fetchRatios(symbol),
      ]);

      const stockInfo = {
        symbol: symbol.toUpperCase(),
        name: profile?.companyName || symbol,
        description: profile?.description || "Investment instrument",
        sector: profile?.sector || "Unknown",
        industry: profile?.industry || "Unknown",
        category: this.categorizeStock(profile, symbol),
        riskLevel: this.assessRiskLevel(profile, ratios),
        price: quote?.price || null,
        marketCap: profile?.mktCap || null,
        dividendYield: ratios?.dividendYield || 0,
        beta: ratios?.beta || null,
        peRatio: ratios?.peRatio || null,
        exchange: profile?.exchange || "Unknown",
        country: profile?.country || "Unknown",
        website: profile?.website || null,
        logo: profile?.image || null,
        isETF: profile?.isEtf || this.isETF(symbol),
        lastUpdated: new Date().toISOString(),
      };

      // Cache the result
      this.cache.set(cacheKey, {
        data: stockInfo,
        timestamp: Date.now(),
      });

      return stockInfo;
    } catch (error) {
      console.error(`Error fetching info for ${symbol}:`, error);
      return this.getFallbackInfo(symbol);
    }
  }

  // Get multiple stocks at once (batch processing)
  async getBatchStockInfo(symbols) {
    const results = {};
    const batchSize = 5; // Process in batches to avoid rate limits

    for (let i = 0; i < symbols.length; i += batchSize) {
      const batch = symbols.slice(i, i + batchSize);
      const promises = batch.map((symbol) =>
        this.getStockInfo(symbol).catch((error) => {
          console.error(`Failed to fetch ${symbol}:`, error);
          return this.getFallbackInfo(symbol);
        })
      );

      const batchResults = await Promise.all(promises);
      batch.forEach((symbol, index) => {
        results[symbol] = batchResults[index];
      });

      // Rate limiting delay
      if (i + batchSize < symbols.length) {
        await this.delay(200); // 200ms delay between batches
      }
    }

    return results;
  }

  // Individual API calls
  async fetchCompanyProfile(symbol) {
    const url = `${this.baseURL}/profile/${symbol}?apikey=${this.apiKey}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    return Array.isArray(data) ? data[0] : data;
  }

  async fetchQuote(symbol) {
    const url = `${this.baseURL}/quote/${symbol}?apikey=${this.apiKey}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    return Array.isArray(data) ? data[0] : data;
  }

  async fetchRatios(symbol) {
    const url = `${this.baseURL}/ratios/${symbol}?apikey=${this.apiKey}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    return Array.isArray(data) ? data[0] : data;
  }

  // Categorization logic
  categorizeStock(profile, symbol) {
    // ETF categorization
    if (profile?.isEtf || this.isETF(symbol)) {
      const name = (profile?.companyName || "").toLowerCase();
      const description = (profile?.description || "").toLowerCase();

      if (name.includes("emerging") || description.includes("emerging"))
        return "Emerging Markets";
      if (name.includes("technology") || name.includes("tech"))
        return "Technology";
      if (name.includes("growth")) return "Large Cap Growth";
      if (name.includes("small") || name.includes("russell 2000"))
        return "Small Cap Growth";
      if (name.includes("bond") || name.includes("fixed income"))
        return "Bonds";
      if (name.includes("reit") || name.includes("real estate"))
        return "Real Estate";
      if (name.includes("dividend")) return "Dividend Stocks";
      if (name.includes("value")) return "Large Cap Value";
      if (name.includes("international") || name.includes("developed"))
        return "International Developed";
      if (name.includes("s&p 500") || name.includes("total stock"))
        return "Broad Market";

      return "Other ETF";
    }

    // Individual stock categorization
    const sector = profile?.sector;
    switch (sector) {
      case "Technology":
      case "Communication Services":
        return "Technology";
      case "Healthcare":
        return "Healthcare";
      case "Financial Services":
      case "Financial":
        return "Financial Services";
      case "Consumer Cyclical":
      case "Consumer Discretionary":
        return "Consumer Discretionary";
      case "Consumer Defensive":
      case "Consumer Staples":
        return "Consumer Staples";
      case "Industrials":
        return "Industrials";
      case "Energy":
        return "Energy";
      case "Utilities":
        return "Utilities";
      case "Real Estate":
        return "Real Estate";
      case "Materials":
        return "Materials";
      default:
        // Categorize by market cap if sector unknown
        const marketCap = profile?.mktCap || 0;
        if (marketCap > 200000000000) return "Large Cap";
        if (marketCap > 10000000000) return "Mid Cap";
        return "Small Cap";
    }
  }

  // Risk assessment
  assessRiskLevel(profile, ratios) {
    const beta = ratios?.beta || 1;
    const sector = profile?.sector;
    const isETF = profile?.isEtf;

    // ETF risk assessment
    if (isETF) {
      const name = (profile?.companyName || "").toLowerCase();
      if (name.includes("ark") || name.includes("leveraged"))
        return "Very High";
      if (name.includes("emerging") || name.includes("small cap"))
        return "High";
      if (name.includes("technology") || name.includes("growth"))
        return "Medium-High";
      if (name.includes("bond") || name.includes("dividend")) return "Low";
      if (name.includes("s&p 500") || name.includes("total stock"))
        return "Medium";
      return "Medium";
    }

    // Individual stock risk assessment
    if (beta > 1.5) return "Very High";
    if (beta > 1.2) return "High";
    if (beta > 0.8) return "Medium-High";
    if (beta > 0.6) return "Medium";
    return "Low";
  }

  // Helper methods
  isETF(symbol) {
    const etfPatterns = ["ETF", "FUND", "TRUST"];
    const etfSuffixes = ["QQQ", "SPY", "IWM", "EEM", "VTI", "VOO"];
    return (
      etfPatterns.some((pattern) => symbol.includes(pattern)) ||
      etfSuffixes.some((suffix) => symbol === suffix) ||
      symbol.length === 3
    ); // Most ETFs are 3 characters
  }

  getFallbackInfo(symbol) {
    return {
      symbol: symbol.toUpperCase(),
      name: symbol,
      description: "Investment instrument - data unavailable",
      category: "Other",
      riskLevel: "Unknown",
      sector: "Unknown",
      isETF: this.isETF(symbol),
      lastUpdated: new Date().toISOString(),
      dataSource: "fallback",
    };
  }

  delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // Clear cache
  clearCache() {
    this.cache.clear();
  }

  // Get cache stats
  getCacheStats() {
    return {
      size: this.cache.size,
      entries: Array.from(this.cache.keys()),
    };
  }
}

// Usage example
export async function enhancePortfolioWithAPI(portfolioData) {
  const api = new StockInfoAPI();
  const stocks = portfolioData?.results?.stocks_picked || [];

  if (stocks.length === 0) return portfolioData;

  try {
    console.log("🔄 Fetching stock information from API...");
    const symbols = stocks.map((stock) => stock.symbol);
    const stockInfo = await api.getBatchStockInfo(symbols);

    // Enhance stocks with API data
    const enhancedStocks = stocks.map((stock) => ({
      ...stock,
      ...stockInfo[stock.symbol],
      // Keep original allocation
      allocation: stock.allocation,
    }));

    // Update portfolio data
    return {
      ...portfolioData,
      results: {
        ...portfolioData.results,
        stocks_picked: enhancedStocks,
        api_enhanced: true,
        last_api_update: new Date().toISOString(),
      },
    };
  } catch (error) {
    console.error("❌ API enhancement failed:", error);
    return portfolioData; // Return original data if API fails
  }
}

export default StockInfoAPI;
