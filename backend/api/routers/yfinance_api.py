# api/routers/yfinance_api.py - New router for your stock data

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import yfinance as yf
import json
import os
import time
from datetime import datetime
import logging

# Import your existing functions
# Adjust the import path based on your project structure
try:
    from your_yfinance_script import (
        get_instrument_database,
        search_instruments, 
        get_current_price,
        create_instrument_endpoint,
        update_instrument_database,
        TOP_100_TICKERS,
        POPULAR_ETFS_AND_FUNDS,
        ALL_SYMBOLS,
        categorize_instrument,
        get_risk_level,
        get_instrument_icon
    )
except ImportError:
    # Fallback functions if the import fails
    def get_current_price(symbol: str) -> float:
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info.get('currentPrice') or ticker.info.get('regularMarketPrice')
        except:
            return None
    
    def create_instrument_endpoint():
        return {}
    
    def search_instruments(query: str, instruments_db: Dict = None, limit: int = 10):
        return []

router = APIRouter()
logger = logging.getLogger(__name__)

# Global cache for instruments database
instruments_cache = None
cache_timestamp = None
CACHE_DURATION = 3600  # 1 hour in seconds

# Pydantic models for request/response
class BatchStockRequest(BaseModel):
    symbols: List[str]

class StockPriceResponse(BaseModel):
    symbol: str
    price: float
    timestamp: str

class StockInfoResponse(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    changePercent: float
    volume: int
    marketCap: Optional[int]
    sector: Optional[str]
    industry: Optional[str]
    exchange: Optional[str]
    currency: str
    lastUpdated: str

class SearchResult(BaseModel):
    symbol: str
    name: str
    description: str
    category: str
    risk: str
    icon: str
    type: str

def get_cached_instruments():
    """Get instruments database with caching"""
    global instruments_cache, cache_timestamp
    
    current_time = time.time()
    
    # Check if cache is valid
    if (instruments_cache is None or 
        cache_timestamp is None or 
        current_time - cache_timestamp > CACHE_DURATION):
        
        logger.info("🔄 Refreshing instruments cache...")
        try:
            instruments_cache = create_instrument_endpoint()
            cache_timestamp = current_time
            logger.info(f"✅ Cached {len(instruments_cache)} instruments")
        except Exception as e:
            logger.error(f"❌ Failed to refresh instruments cache: {e}")
            if instruments_cache is None:
                instruments_cache = {}
    
    return instruments_cache

@router.get("/health")
async def yfinance_health():
    """Health check for yfinance API endpoints"""
    try:
        # Test with a simple stock
        test_symbol = "AAPL"
        ticker = yf.Ticker(test_symbol)
        price = ticker.info.get('currentPrice')
        
        instruments_count = len(get_cached_instruments())
        
        return {
            "status": "healthy",
            "service": "YFinance API",
            "test_symbol": test_symbol,
            "test_price": price,
            "instruments_cached": instruments_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ YFinance health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"YFinance service unhealthy: {str(e)}")

@router.get("/instruments")
async def get_instruments():
    """Get all instruments database"""
    try:
        instruments = get_cached_instruments()
        return JSONResponse(content=instruments)
    except Exception as e:
        logger.error(f"❌ Error getting instruments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/instruments/update")
async def update_instruments(background_tasks: BackgroundTasks):
    """Force update instruments database in background"""
    try:
        def update_cache():
            global instruments_cache, cache_timestamp
            logger.info("🔄 Background updating instruments database...")
            try:
                instruments_cache = update_instrument_database()
                cache_timestamp = time.time()
                logger.info(f"✅ Background update completed: {len(instruments_cache)} instruments")
            except Exception as e:
                logger.error(f"❌ Background update failed: {e}")
        
        background_tasks.add_task(update_cache)
        
        return {
            "message": "Instruments database update started in background",
            "status": "initiated",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error initiating instruments update: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/price/{symbol}")
async def get_price(symbol: str):
    """Get current price for a symbol"""
    try:
        symbol = symbol.upper()
        
        # Add timeout and error handling
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if price is None:
            logger.warning(f"⚠️ No price data available for {symbol}")
            raise HTTPException(status_code=404, detail=f"No price data available for {symbol}")
        
        return StockPriceResponse(
            symbol=symbol,
            price=float(price),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting price for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch price: {str(e)}")

@router.get("/stock/{symbol}")
async def get_stock_info(symbol: str):
    """Get comprehensive stock information"""
    try:
        symbol = symbol.upper()
        logger.info(f"📡 Fetching comprehensive info for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or not info.get('longName'):
            logger.warning(f"⚠️ No data available for {symbol}")
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Get historical data for additional metrics
        try:
            hist = ticker.history(period="5d")
        except:
            hist = None
        
        # Calculate price change
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        previous_close = info.get('previousClose') or current_price
        change = current_price - previous_close
        change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
        
        # Get volume from history or info
        volume = 0
        if hist is not None and not hist.empty:
            volume = int(hist['Volume'].iloc[-1]) if len(hist) > 0 else 0
        if volume == 0:
            volume = info.get('volume', 0) or info.get('regularMarketVolume', 0)
        
        # Prepare response data
        stock_data = {
            "symbol": symbol,
            "name": info.get('longName', symbol),
            "info": {
                "longName": info.get('longName'),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "currentPrice": current_price,
                "previousClose": previous_close,
                "marketCap": info.get('marketCap'),
                "dividendYield": info.get('dividendYield'),
                "beta": info.get('beta'),
                "pegRatio": info.get('pegRatio'),
                "priceToBook": info.get('priceToBook'),
                "currency": info.get('currency', 'USD'),
                "exchange": info.get('exchange'),
                "country": info.get('country'),
                "quoteType": info.get('quoteType'),
                "longBusinessSummary": info.get('longBusinessSummary', '')[:200] + "..." if info.get('longBusinessSummary') else "",
                "volume": volume,
                "trailingPE": info.get('trailingPE')
            },
            "metadata": {
                "currentPrice": float(current_price),
                "previousClose": float(previous_close),
                "change": round(change, 2),
                "changePercent": round(change_percent, 2),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "marketCap": info.get('marketCap'),
                "dividendYield": info.get('dividendYield'),
                "beta": info.get('beta'),
                "pegRatio": info.get('pegRatio'),
                "priceToBook": info.get('priceToBook'),
                "currency": info.get('currency', 'USD'),
                "exchange": info.get('exchange'),
                "country": info.get('country'),
                "volume": volume,
                "lastUpdated": datetime.now().isoformat()
            },
            "history": {
                "prices": hist['Close'].tail(5).tolist() if hist is not None and not hist.empty else [],
                "volumes": hist['Volume'].tail(5).tolist() if hist is not None and not hist.empty else [],
                "dates": [d.strftime('%Y-%m-%d') for d in hist.index.tail(5)] if hist is not None and not hist.empty else []
            },
            # Add categorization from your script
            "category": categorize_instrument(info) if 'categorize_instrument' in globals() else 'other',
            "risk": get_risk_level(info, categorize_instrument(info)) if 'get_risk_level' in globals() else 'Medium',
            "icon": get_instrument_icon(categorize_instrument(info), info) if 'get_instrument_icon' in globals() else '📊',
            "type": "ETF" if info.get('quoteType') == 'ETF' else "Stock"
        }
        
        return JSONResponse(content=stock_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting stock info for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock info: {str(e)}")

@router.post("/stocks/batch")
async def get_batch_stocks(request: BatchStockRequest):
    """Get information for multiple stocks"""
    try:
        symbols = request.symbols
        
        if not symbols:
            raise HTTPException(status_code=400, detail="No symbols provided")
        
        if len(symbols) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Too many symbols. Maximum 50 allowed.")
        
        logger.info(f"🔄 Processing batch request for {len(symbols)} symbols...")
        
        results = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                symbol = symbol.upper()
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info and info.get('longName'):
                    # Get basic info for batch processing (faster)
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
                    previous_close = info.get('previousClose') or current_price
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
                    
                    results[symbol] = {
                        "symbol": symbol,
                        "name": info.get('longName', symbol),
                        "info": {
                            "longName": info.get('longName'),
                            "sector": info.get('sector'),
                            "industry": info.get('industry'),
                            "currentPrice": current_price,
                            "previousClose": previous_close,
                            "marketCap": info.get('marketCap'),
                            "dividendYield": info.get('dividendYield'),
                            "beta": info.get('beta'),
                            "currency": info.get('currency', 'USD'),
                            "exchange": info.get('exchange'),
                            "country": info.get('country'),
                            "quoteType": info.get('quoteType'),
                            "longBusinessSummary": info.get('longBusinessSummary', '')[:150] + "..." if info.get('longBusinessSummary') else "",
                            "volume": info.get('volume', 0) or info.get('regularMarketVolume', 0)
                        },
                        "metadata": {
                            "currentPrice": float(current_price),
                            "previousClose": float(previous_close),
                            "change": round(change, 2),
                            "changePercent": round(change_percent, 2),
                            "sector": info.get('sector'),
                            "industry": info.get('industry'),
                            "marketCap": info.get('marketCap'),
                            "dividendYield": info.get('dividendYield'),
                            "beta": info.get('beta'),
                            "currency": info.get('currency', 'USD'),
                            "exchange": info.get('exchange'),
                            "country": info.get('country'),
                            "volume": info.get('volume', 0) or info.get('regularMarketVolume', 0),
                            "lastUpdated": datetime.now().isoformat()
                        },
                        "category": categorize_instrument(info) if 'categorize_instrument' in globals() else 'other',
                        "risk": get_risk_level(info, categorize_instrument(info)) if 'get_risk_level' in globals() else 'Medium',
                        "icon": get_instrument_icon(categorize_instrument(info), info) if 'get_instrument_icon' in globals() else '📊',
                        "type": "ETF" if info.get('quoteType') == 'ETF' else "Stock"
                    }
                else:
                    failed_symbols.append(symbol)
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                continue
        
        response_data = {
            "results": results,
            "success_count": len(results),
            "total_requested": len(symbols),
            "failed_symbols": failed_symbols,
            "timestamp": datetime.now().isoformat()
        }
        
        # Return the results dictionary directly for easier frontend processing
        logger.info(f"✅ Batch request completed: {len(results)}/{len(symbols)} successful")
        return JSONResponse(content=results)  # Return just the results dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in batch request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch request failed: {str(e)}")

@router.get("/search")
async def search_stocks(query: str, limit: int = 10):
    """Search instruments by name, symbol, or category"""
    try:
        if not query or len(query.strip()) < 1:
            raise HTTPException(status_code=400, detail="Query parameter is required")
        
        if limit > 50:
            limit = 50  # Cap the limit
        
        logger.info(f"🔍 Searching for '{query}' with limit {limit}")
        
        # Try to use your existing search function
        try:
            instruments_db = get_cached_instruments()
            results = search_instruments(query, instruments_db, limit)
            
            # Convert to expected format
            formatted_results = []
            for result in results:
                formatted_results.append(SearchResult(
                    symbol=result.get('symbol', ''),
                    name=result.get('name', ''),
                    description=result.get('description', ''),
                    category=result.get('category', ''),
                    risk=result.get('risk', 'Medium'),
                    icon=result.get('icon', '📊'),
                    type=result.get('type', 'Stock')
                ))
            
            logger.info(f"✅ Search completed: {len(formatted_results)} results")
            return formatted_results
            
        except Exception as search_error:
            logger.warning(f"⚠️ Advanced search failed, using basic search: {search_error}")
            
            # Fallback to basic search
            basic_results = []
            query_upper = query.upper()
            
            # Search in common symbols
            from your_yfinance_script import ALL_SYMBOLS
            matching_symbols = [s for s in ALL_SYMBOLS if query_upper in s][:limit]
            
            for symbol in matching_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if info.get('longName'):
                        basic_results.append(SearchResult(
                            symbol=symbol,
                            name=info.get('longName', symbol),
                            description=info.get('longBusinessSummary', '')[:100] + "..." if info.get('longBusinessSummary') else f"Investment in {symbol}",
                            category='other',
                            risk='Medium',
                            icon='📊',
                            type="ETF" if info.get('quoteType') == 'ETF' else "Stock"
                        ))
                        
                        if len(basic_results) >= limit:
                            break
                            
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get info for {symbol}: {e}")
                    continue
            
            logger.info(f"✅ Basic search completed: {len(basic_results)} results")
            return basic_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/categories")
async def get_categories():
    """Get all available investment categories"""
    categories = {
        "bonds": {
            "name": "Bonds",
            "description": "Government and corporate bonds",
            "risk": "Low",
            "icon": "🏛️"
        },
        "dividend_stocks": {
            "name": "Dividend Stocks", 
            "description": "High dividend yield stocks",
            "risk": "Low-Medium",
            "icon": "💰"
        },
        "utilities": {
            "name": "Utilities",
            "description": "Electric, gas, and water utilities",
            "risk": "Low-Medium",
            "icon": "⚡"
        },
        "large_cap_growth": {
            "name": "Large Cap Growth",
            "description": "Large company growth stocks",
            "risk": "Medium",
            "icon": "📈"
        },
        "broad_market": {
            "name": "Broad Market",
            "description": "Diversified market ETFs",
            "risk": "Medium",
            "icon": "📊"
        },
        "international": {
            "name": "International",
            "description": "International developed markets",
            "risk": "Medium",
            "icon": "🌍"
        },
        "technology": {
            "name": "Technology",
            "description": "Technology sector investments",
            "risk": "Medium-High",
            "icon": "💻"
        },
        "emerging_markets": {
            "name": "Emerging Markets",
            "description": "Emerging market investments",
            "risk": "High",
            "icon": "🌏"
        },
        "high_growth": {
            "name": "High Growth",
            "description": "High growth potential stocks",
            "risk": "Very High",
            "icon": "🚀"
        },
        "reits": {
            "name": "REITs",
            "description": "Real Estate Investment Trusts",
            "risk": "Medium",
            "icon": "🏢"
        },
        "financials": {
            "name": "Financials",
            "description": "Banking and financial services",
            "risk": "Medium",
            "icon": "🏦"
        }
    }
    
    return categories

@router.get("/popular")
async def get_popular_stocks():
    """Get popular stocks and ETFs"""
    try:
        popular_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "SPY", "QQQ", "VTI"
        ]
        
        results = {}
        for symbol in popular_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info.get('longName'):
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    previous_close = info.get('previousClose')
                    change = (current_price - previous_close) if current_price and previous_close else 0
                    change_percent = (change / previous_close * 100) if previous_close else 0
                    
                    results[symbol] = {
                        "symbol": symbol,
                        "name": info.get('longName'),
                        "price": current_price,
                        "change": round(change, 2),
                        "changePercent": round(change_percent, 2),
                        "type": "ETF" if info.get('quoteType') == 'ETF' else "Stock"
                    }
            except Exception as e:
                logger.warning(f"⚠️ Failed to get popular stock {symbol}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error getting popular stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))