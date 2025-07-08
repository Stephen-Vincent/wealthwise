import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
import json
import time
from datetime import datetime

# Extended list including TOP 100 TICKERS plus popular ETFs, bonds, and REITs
TOP_100_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "LLY", "UNH",
    "JNJ", "V", "XOM", "PG", "MA", "AVGO", "CVX", "MRK", "JPM", "HD",
    "ABBV", "COST", "PEP", "ADBE", "KO", "BAC", "CRM", "TMO", "WMT", "ACN",
    "NFLX", "ORCL", "MCD", "ABT", "INTC", "DHR", "LIN", "CSCO", "NKE", "AMD",
    "NEE", "QCOM", "TXN", "WFC", "AMGN", "PM", "UPS", "BMY", "MS", "RTX",
    "AMAT", "UNP", "CVS", "LOW", "HON", "GS", "CAT", "IBM", "GE", "LMT",
    "INTU", "SPGI", "MDT", "BLK", "ISRG", "NOW", "ZTS", "SCHW", "T", "ADI",
    "MU", "DE", "SYK", "PLD", "C", "VRTX", "GILD", "ADP", "REGN", "AXP",
    "CI", "MO", "ELV", "TJX", "PGR", "SO", "PNC", "BDX", "MMC", "CL",
    "DUK", "CB", "ETN", "FISV", "ADSK", "ITW", "EW", "SHW", "NSC", "AON"
]

# Popular ETFs, Bonds, and REITs to expand database
POPULAR_ETFS_AND_FUNDS = [
    # Broad Market ETFs
    "VTI", "VOO", "SPY", "IVV", "VEA", "IEFA", "VWO", "IEMG", "VT", "VXUS",
    
    # Bond ETFs
    "BND", "AGG", "TLT", "IEF", "SHY", "VTEB", "LQD", "HYG", "TIP", "SCHZ",
    
    # Sector ETFs
    "QQQ", "VGT", "XLK", "FTEC", "IYW", "XLF", "XLE", "XLV", "XLI", "XLU",
    
    # High Growth/Innovation
    "ARKK", "ARKQ", "ARKG", "VUG", "IWF", "SCHG", "MGK",
    
    # International
    "VGK", "EWJ", "FEZ", "EWG", "EWU", "EWC", "EWA", "IEUS",
    
    # Emerging Markets
    "EEM", "SCHE", "VPL", "EEMV", "DEM", "SPEM", "EWZ", "FXI",
    
    # REITs
    "VNQ", "SCHH", "IYR", "XLRE", "RWR", "FREL", "REZ", "USRT", "VNQI", "REET"
]

# Combine all symbols
ALL_SYMBOLS = TOP_100_TICKERS + POPULAR_ETFS_AND_FUNDS

def categorize_instrument(info: Dict) -> str:
    """Categorize instrument based on yfinance info"""
    sector = info.get('sector', '').lower()
    industry = info.get('industry', '').lower()
    quote_type = info.get('quoteType', '').lower()
    long_name = info.get('longName', '').lower()
    short_name = info.get('shortName', '').lower()
    
    # ETF Detection
    if quote_type == 'etf' or 'etf' in long_name or 'etf' in short_name:
        if 'bond' in long_name or 'treasury' in long_name or 'fixed income' in long_name:
            return 'bonds'
        elif 'reit' in long_name or 'real estate' in long_name:
            return 'reits'
        elif 'international' in long_name or 'developed markets' in long_name:
            return 'international'
        elif 'emerging' in long_name:
            return 'emerging_markets'
        elif 'technology' in long_name or 'tech' in long_name or 'nasdaq' in long_name:
            return 'technology'
        elif 'innovation' in long_name or 'growth' in long_name:
            return 'high_growth'
        elif 'total' in long_name and 'market' in long_name:
            return 'broad_market'
        else:
            return 'broad_market'
    
    # Stock categorization
    elif quote_type == 'equity':
        market_cap = info.get('marketCap', 0)
        
        # Utility stocks
        if sector == 'utilities':
            return 'utilities'
        
        # Financial services
        elif sector == 'financial services' or 'financial' in sector:
            return 'financials'
        
        # Dividend stocks (dividend yield > 2%)
        elif info.get('dividendYield', 0) and info.get('dividendYield', 0) > 0.02:
            return 'dividend_stocks'
        
        # Large cap growth (market cap > 10B)
        elif market_cap and market_cap > 10_000_000_000:
            if sector in ['technology', 'communication services', 'consumer discretionary']:
                return 'large_cap_growth'
            else:
                return 'dividend_stocks'
        
        # High growth stocks
        elif sector == 'technology' or info.get('pegRatio', 0) > 1.5:
            return 'high_growth'
        
        else:
            return 'large_cap_growth'
    
    # Bond detection
    elif 'bond' in quote_type or 'bond' in long_name:
        return 'bonds'
    
    return 'other'

def get_risk_level(info: Dict, category: str) -> str:
    """Determine risk level based on instrument info"""
    beta = info.get('beta')
    
    if category == 'bonds':
        return 'Low'
    elif category in ['utilities', 'dividend_stocks']:
        return 'Low-Medium'
    elif category in ['large_cap_growth', 'broad_market', 'financials']:
        return 'Medium'
    elif category in ['international', 'reits']:
        return 'Medium'
    elif category == 'technology':
        return 'Medium-High'
    elif category in ['emerging_markets', 'high_growth']:
        if beta and beta > 1.5:
            return 'Very High'
        else:
            return 'High'
    else:
        return 'Medium'

def get_instrument_icon(category: str, info: Dict) -> str:
    """Get emoji icon for instrument category"""
    # Special cases based on company name
    long_name = info.get('longName', '').lower()
    symbol = info.get('symbol', '').upper()
    
    special_icons = {
        'AAPL': '📱',
        'MSFT': '💻',
        'GOOGL': '🔍',
        'AMZN': '📦',
        'TSLA': '🚗',
        'NVDA': '🤖',
        'META': '📘',
        'V': '💳',
        'MA': '💳',
        'JNJ': '💊',
        'KO': '🥤',
        'PEP': '🥤',
        'PG': '🧴'
    }
    
    if symbol in special_icons:
        return special_icons[symbol]
    
    # Category-based icons
    icon_map = {
        'bonds': '🏛️',
        'dividend_stocks': '💰',
        'utilities': '⚡',
        'large_cap_growth': '📈',
        'broad_market': '📊',
        'international': '🌍',
        'emerging_markets': '🌏',
        'technology': '💻',
        'high_growth': '🚀',
        'reits': '🏢',
        'financials': '🏦',
        'other': '📊'
    }
    
    return icon_map.get(category, '📊')

def get_instrument_database(symbols: List[str] = None, save_to_file: bool = True) -> Dict:
    """
    Create a comprehensive instrument database with categorization.
    
    Args:
        symbols: List of symbols to fetch (defaults to ALL_SYMBOLS)
        save_to_file: Whether to save the database to a JSON file
    
    Returns:
        Dict: Instrument database compatible with your React component
    """
    if symbols is None:
        symbols = ALL_SYMBOLS
    
    print(f"🔄 Building instrument database for {len(symbols)} symbols...")
    
    instruments = {}
    failed_symbols = []
    
    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"  [{i}/{len(symbols)}] Fetching {symbol}...", end=" ")
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'longName' not in info:
                print("❌ No data")
                failed_symbols.append(symbol)
                continue
            
            category = categorize_instrument(info)
            risk = get_risk_level(info, category)
            icon = get_instrument_icon(category, info)
            
            # Create instrument entry compatible with your React component
            instruments[symbol] = {
                'name': info.get('longName', symbol),
                'type': info.get('quoteType', 'Unknown').replace('EQUITY', 'Stock').replace('ETF', 'ETF').title(),
                'category': category,
                'description': (info.get('longBusinessSummary', '') or 
                              info.get('description', '') or 
                              f"{info.get('longName', symbol)} investment")[:150] + "..." if info.get('longBusinessSummary') else f"{info.get('longName', symbol)} investment",
                'risk': risk,
                'icon': icon,
                # Additional metadata for advanced features
                'metadata': {
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'marketCap': info.get('marketCap'),
                    'dividendYield': info.get('dividendYield'),
                    'beta': info.get('beta'),
                    'pegRatio': info.get('pegRatio'),
                    'priceToBook': info.get('priceToBook'),
                    'currentPrice': info.get('currentPrice') or info.get('regularMarketPrice'),
                    'currency': info.get('currency', 'USD'),
                    'exchange': info.get('exchange'),
                    'country': info.get('country'),
                    'lastUpdated': datetime.now().isoformat()
                }
            }
            
            print("✅")
            
            # Rate limiting to be respectful to Yahoo Finance
            if i % 10 == 0:
                time.sleep(1)
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            failed_symbols.append(symbol)
            continue
    
    print(f"\n✅ Successfully fetched {len(instruments)} instruments")
    if failed_symbols:
        print(f"⚠️  Failed to fetch {len(failed_symbols)} symbols: {failed_symbols}")
    
    # Save to file if requested
    if save_to_file:
        output_file = 'instrument_database.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(instruments, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved database to {output_file}")
    
    # Print summary by category
    category_counts = {}
    for instrument in instruments.values():
        category = instrument['category']
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print("\n📊 Summary by category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} instruments")
    
    return instruments

def get_stock_data(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Your original function - unchanged for backward compatibility
    """
    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            continue  # Skip symbols with no data

        # Ensure datetime index
        hist.index = pd.to_datetime(hist.index)

        # Calculate additional indicators
        hist["Return"] = hist["Close"].pct_change()  # Daily returns
        hist["MA50"] = hist["Close"].rolling(window=50).mean()  # 50-day moving average
        hist["MA200"] = hist["Close"].rolling(window=200).mean()  # 200-day moving average
        hist["Volatility20"] = hist["Return"].rolling(window=20).std()  # 20-day rolling volatility

        # Keep only end-of-month values for training
        hist = hist.resample('M').last()

        data[symbol] = hist.dropna()  # Remove any rows with NaNs

    return data

def get_current_price(symbol: str) -> float:
    """
    Your original function - unchanged for backward compatibility
    """
    ticker = yf.Ticker(symbol)
    price = ticker.info.get('currentPrice')
    return price

def search_instruments(query: str, instruments_db: Dict = None, limit: int = 10) -> List[Dict]:
    """
    Search instruments by name, symbol, or category
    
    Args:
        query: Search query
        instruments_db: Instrument database (will load from file if None)
        limit: Maximum number of results
    
    Returns:
        List of matching instruments
    """
    if instruments_db is None:
        try:
            with open('instrument_database.json', 'r', encoding='utf-8') as f:
                instruments_db = json.load(f)
        except FileNotFoundError:
            print("⚠️  Instrument database not found. Run get_instrument_database() first.")
            return []
    
    query = query.lower()
    results = []
    
    for symbol, data in instruments_db.items():
        # Search in symbol, name, category, and type
        searchable_text = ' '.join([
            symbol.lower(),
            data.get('name', '').lower(),
            data.get('category', '').lower(),
            data.get('type', '').lower()
        ])
        
        if query in searchable_text:
            results.append({
                'symbol': symbol,
                **data
            })
    
    return results[:limit]

# Example usage and endpoint functions for your API
def create_instrument_endpoint():
    """
    Example function to create API endpoint that returns instrument database
    """
    try:
        # Try to load existing database
        with open('instrument_database.json', 'r', encoding='utf-8') as f:
            instruments = json.load(f)
        print("📚 Loaded existing instrument database")
    except FileNotFoundError:
        # Create new database if it doesn't exist
        print("🔨 Creating new instrument database...")
        instruments = get_instrument_database()
    
    return instruments

def update_instrument_database():
    """
    Function to update the instrument database with latest data
    """
    print("🔄 Updating instrument database...")
    return get_instrument_database()

# Main execution
if __name__ == "__main__":
    # Create or update the instrument database
    database = get_instrument_database()
    
    # Example searches
    print("\n🔍 Example searches:")
    apple_results = search_instruments("apple", database)
    print(f"Search 'apple': {len(apple_results)} results")
    
    tech_results = search_instruments("technology", database)
    print(f"Search 'technology': {len(tech_results)} results")
    
    bond_results = search_instruments("bond", database)
    print(f"Search 'bond': {len(bond_results)} results")