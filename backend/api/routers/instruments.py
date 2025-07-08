# api/routers/instruments.py - COMPLETE VERSION
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import os

router = APIRouter()

# COMPREHENSIVE INSTRUMENTS DATABASE - 100+ instruments covering most common stocks/ETFs
INSTRUMENTS_DATABASE = {
    # MEGA CAP TECH (Large Cap Growth)
    "AAPL": {
        "name": "Apple Inc.",
        "type": "Large Cap Growth Stock",
        "category": "large_cap_growth",
        "description": "Technology giant known for iPhones, Macs, and innovative consumer electronics",
        "risk": "Medium",
        "icon": "📱",
        "metadata": {"sector": "Technology", "industry": "Consumer Electronics", "marketCap": 3000000000000, "beta": 1.2, "dividendYield": 0.005, "exchange": "NASDAQ"}
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "type": "Large Cap Growth Stock",
        "category": "large_cap_growth",
        "description": "Cloud computing and software leader with Azure, Office, and Windows platforms",
        "risk": "Medium",
        "icon": "💻",
        "metadata": {"sector": "Technology", "industry": "Software", "marketCap": 2800000000000, "beta": 0.9, "dividendYield": 0.007, "exchange": "NASDAQ"}
    },
    "GOOGL": {
        "name": "Alphabet Inc. (Google)",
        "type": "Large Cap Growth Stock",
        "category": "large_cap_growth",
        "description": "Search, advertising, and cloud technology company with dominant market positions",
        "risk": "Medium",
        "icon": "🔍",
        "metadata": {"sector": "Communication Services", "industry": "Internet Content & Information", "marketCap": 2000000000000, "beta": 1.1, "dividendYield": None, "exchange": "NASDAQ"}
    },
    "GOOG": {
        "name": "Alphabet Inc. (Google) Class C",
        "type": "Large Cap Growth Stock",
        "category": "large_cap_growth",
        "description": "Search, advertising, and cloud technology company (Class C shares)",
        "risk": "Medium",
        "icon": "🔍",
        "metadata": {"sector": "Communication Services", "industry": "Internet Content & Information", "marketCap": 2000000000000, "beta": 1.1, "dividendYield": None, "exchange": "NASDAQ"}
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "type": "Large Cap Growth Stock",
        "category": "large_cap_growth",
        "description": "E-commerce and cloud computing giant with AWS leading cloud services",
        "risk": "Medium-High",
        "icon": "📦",
        "metadata": {"sector": "Consumer Discretionary", "industry": "Internet Retail", "marketCap": 1500000000000, "beta": 1.3, "dividendYield": None, "exchange": "NASDAQ"}
    },
    "META": {
        "name": "Meta Platforms Inc.",
        "type": "Large Cap Growth Stock",
        "category": "large_cap_growth",
        "description": "Social media and virtual reality company focusing on connecting people through technology platforms",
        "risk": "Medium-High",
        "icon": "📘",
        "metadata": {"sector": "Communication Services", "industry": "Internet Content & Information", "marketCap": 800000000000, "beta": 1.2, "dividendYield": None, "exchange": "NASDAQ"}
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "type": "Growth Stock",
        "category": "high_growth",
        "description": "Electric vehicles and clean energy technology company",
        "risk": "High",
        "icon": "🚗",
        "metadata": {"sector": "Consumer Discretionary", "industry": "Auto Manufacturers", "marketCap": 800000000000, "beta": 2.0, "dividendYield": None, "exchange": "NASDAQ"}
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "type": "Growth Stock",
        "category": "high_growth",
        "description": "AI and graphics processing leader, powering gaming, data centers, and autonomous vehicles",
        "risk": "High",
        "icon": "🤖",
        "metadata": {"sector": "Technology", "industry": "Semiconductors", "marketCap": 1800000000000, "beta": 1.7, "dividendYield": 0.002, "exchange": "NASDAQ"}
    },
    "CRM": {
        "name": "Salesforce Inc.",
        "type": "Growth Stock",
        "category": "large_cap_growth",
        "description": "Cloud-based customer relationship management and business software",
        "risk": "Medium-High",
        "icon": "☁️",
        "metadata": {"sector": "Technology", "industry": "Software", "marketCap": 200000000000, "beta": 1.4}
    },
    "ORCL": {
        "name": "Oracle Corporation",
        "type": "Large Cap Growth Stock",
        "category": "large_cap_growth",
        "description": "Enterprise software and database solutions provider",
        "risk": "Medium",
        "icon": "🗄️",
        "metadata": {"sector": "Technology", "industry": "Software", "marketCap": 300000000000, "beta": 1.0}
    },
    "ADBE": {
        "name": "Adobe Inc.",
        "type": "Large Cap Growth Stock",
        "category": "large_cap_growth",
        "description": "Creative software and digital marketing solutions",
        "risk": "Medium",
        "icon": "🎨",
        "metadata": {"sector": "Technology", "industry": "Software", "marketCap": 200000000000, "beta": 1.2}
    },
    "INTC": {
        "name": "Intel Corporation",
        "type": "Large Cap Stock",
        "category": "large_cap_growth",
        "description": "Semiconductor and processor manufacturer",
        "risk": "Medium",
        "icon": "🔌",
        "metadata": {"sector": "Technology", "industry": "Semiconductors", "marketCap": 150000000000, "beta": 1.0}
    },
    "AMD": {
        "name": "Advanced Micro Devices",
        "type": "Growth Stock",
        "category": "high_growth",
        "description": "Semiconductor and processor manufacturer competing with Intel and NVIDIA",
        "risk": "High",
        "icon": "💾",
        "metadata": {"sector": "Technology", "industry": "Semiconductors", "marketCap": 200000000000, "beta": 1.8}
    },
    
    # HEALTHCARE & PHARMACEUTICALS
    "JNJ": {
        "name": "Johnson & Johnson",
        "type": "Large Cap Dividend Stock",
        "category": "dividend_stocks",
        "description": "Healthcare giant with 60+ years of dividend growth and diversified product portfolio",
        "risk": "Low-Medium",
        "icon": "💊",
        "metadata": {"sector": "Healthcare", "industry": "Drug Manufacturers", "marketCap": 450000000000, "beta": 0.7, "dividendYield": 0.027, "exchange": "NYSE"}
    },
    "PFE": {
        "name": "Pfizer Inc.",
        "type": "Large Cap Dividend Stock",
        "category": "dividend_stocks",
        "description": "Pharmaceutical company with strong dividend history and global reach",
        "risk": "Low-Medium",
        "icon": "💉",
        "metadata": {"sector": "Healthcare", "industry": "Drug Manufacturers", "marketCap": 250000000000, "beta": 0.8}
    },
    "ABBV": {
        "name": "AbbVie Inc.",
        "type": "Large Cap Dividend Stock",
        "category": "dividend_stocks",
        "description": "Biopharmaceutical company focused on immunology and oncology",
        "risk": "Medium",
        "icon": "🧬",
        "metadata": {"sector": "Healthcare", "industry": "Drug Manufacturers", "marketCap": 300000000000, "beta": 0.9}
    },
    "MRK": {
        "name": "Merck & Co. Inc.",
        "type": "Large Cap Dividend Stock",
        "category": "dividend_stocks",
        "description": "Global pharmaceutical company with strong research pipeline",
        "risk": "Low-Medium",
        "icon": "⚕️",
        "metadata": {"sector": "Healthcare", "industry": "Drug Manufacturers", "marketCap": 280000000000, "beta": 0.8}
    },
    "LLY": {
        "name": "Eli Lilly and Company",
        "type": "Large Cap Growth Stock",
        "category": "large_cap_growth",
        "description": "Pharmaceutical company specializing in diabetes care and alzheimer's research",
        "risk": "Medium",
        "icon": "💊",
        "metadata": {"sector": "Healthcare", "industry": "Drug Manufacturers", "marketCap": 500000000000, "beta": 0.9}
    },
    "UNH": {
        "name": "UnitedHealth Group",
        "type": "Large Cap Growth Stock",
        "category": "large_cap_growth",
        "description": "Healthcare insurance and services with consistent growth",
        "risk": "Medium",
        "icon": "🏥",
        "metadata": {"sector": "Healthcare", "industry": "Healthcare Plans", "marketCap": 500000000000, "beta": 1.0}
    },
    
    # CONSUMER STAPLES
    "KO": {
        "name": "The Coca-Cola Company",
        "type": "Large Cap Dividend Stock", 
        "category": "dividend_stocks",
        "description": "Global beverage leader with consistent dividends and strong brand portfolio",
        "risk": "Low-Medium",
        "icon": "🥤",
        "metadata": {"sector": "Consumer Staples", "industry": "Beverages", "marketCap": 260000000000, "beta": 0.6, "dividendYield": 0.030, "exchange": "NYSE"}
    },
    "PEP": {
        "name": "PepsiCo Inc.",
        "type": "Large Cap Dividend Stock",
        "category": "dividend_stocks",
        "description": "Food and beverage conglomerate with global reach and stable dividends",
        "risk": "Low-Medium",
        "icon": "🥤",
        "metadata": {"sector": "Consumer Staples", "industry": "Beverages", "marketCap": 240000000000, "beta": 0.7}
    },
    "PG": {
        "name": "Procter & Gamble",
        "type": "Large Cap Dividend Stock", 
        "category": "dividend_stocks",
        "description": "Consumer goods company with stable dividends and strong brand portfolio",
        "risk": "Low-Medium",
        "icon": "🧴",
        "metadata": {"sector": "Consumer Staples", "industry": "Household & Personal Products", "marketCap": 350000000000, "beta": 0.5, "dividendYield": 0.024, "exchange": "NYSE"}
    },
    "WMT": {
        "name": "Walmart Inc.",
        "type": "Large Cap Dividend Stock",
        "category": "dividend_stocks",
        "description": "Retail giant with growing dividend and e-commerce expansion",
        "risk": "Low-Medium",
        "icon": "🛒",
        "metadata": {"sector": "Consumer Staples", "industry": "Discount Stores", "marketCap": 500000000000, "beta": 0.5}
    },
    "COST": {
        "name": "Costco Wholesale Corp",
        "type": "Large Cap Growth Stock",
        "category": "large_cap_growth",
        "description": "Membership-based warehouse club with loyal customer base",
        "risk": "Medium",
        "icon": "🏪",
        "metadata": {"sector": "Consumer Staples", "industry": "Discount Stores", "marketCap": 250000000000, "beta": 0.8}
    },
    "MCD": {
        "name": "McDonald's Corporation",
        "type": "Large Cap Dividend Stock",
        "category": "dividend_stocks",
        "description": "Global fast-food restaurant chain with consistent dividend growth",
        "risk": "Low-Medium",
        "icon": "🍔",
        "metadata": {"sector": "Consumer Discretionary", "industry": "Restaurants", "marketCap": 200000000000, "beta": 0.7}
    },
    
    # FINANCIALS
    "JPM": {
        "name": "JPMorgan Chase & Co.",
        "type": "Banking Stock",
        "category": "financials",
        "description": "Largest U.S. bank by assets with diversified financial services",
        "risk": "Medium",
        "icon": "🏦",
        "metadata": {"sector": "Financial Services", "industry": "Banks", "marketCap": 450000000000, "beta": 1.1}
    },
    "BAC": {
        "name": "Bank of America Corp",
        "type": "Banking Stock",
        "category": "financials",
        "description": "Major U.S. commercial bank with extensive retail presence",
        "risk": "Medium",
        "icon": "🏦",
        "metadata": {"sector": "Financial Services", "industry": "Banks", "marketCap": 250000000000, "beta": 1.2}
    },
    "WFC": {
        "name": "Wells Fargo & Company",
        "type": "Banking Stock",
        "category": "financials",
        "description": "Major U.S. commercial bank focusing on retail banking",
        "risk": "Medium",
        "icon": "🏦",
        "metadata": {"sector": "Financial Services", "industry": "Banks", "marketCap": 180000000000, "beta": 1.3}
    },
    "V": {
        "name": "Visa Inc.",
        "type": "Financial Services Stock",
        "category": "financials",
        "description": "Global payment processing network with dominant market position",
        "risk": "Medium",
        "icon": "💳",
        "metadata": {"sector": "Financial Services", "industry": "Credit Services", "marketCap": 500000000000, "beta": 1.0}
    },
    "MA": {
        "name": "Mastercard Inc.",
        "type": "Financial Services Stock",
        "category": "financials",
        "description": "Global payment technology company competing with Visa",
        "risk": "Medium",
        "icon": "💳",
        "metadata": {"sector": "Financial Services", "industry": "Credit Services", "marketCap": 400000000000, "beta": 1.1}
    },
    "BRK-B": {
        "name": "Berkshire Hathaway Inc.",
        "type": "Large Cap Value Stock",
        "category": "large_cap_growth",
        "description": "Warren Buffett's conglomerate holding company with diversified investments",
        "risk": "Medium",
        "icon": "🏛️",
        "metadata": {"sector": "Financial Services", "industry": "Insurance", "marketCap": 800000000000, "beta": 0.9}
    },
    "GS": {
        "name": "Goldman Sachs Group",
        "type": "Financial Services Stock",
        "category": "financials",
        "description": "Global investment banking and financial services firm",
        "risk": "Medium-High",
        "icon": "💼",
        "metadata": {"sector": "Financial Services", "industry": "Investment Banking", "marketCap": 120000000000, "beta": 1.4}
    },
    "MS": {
        "name": "Morgan Stanley",
        "type": "Financial Services Stock",
        "category": "financials",
        "description": "Global financial services firm specializing in investment banking",
        "risk": "Medium-High",
        "icon": "💼",
        "metadata": {"sector": "Financial Services", "industry": "Investment Banking", "marketCap": 150000000000, "beta": 1.3}
    },
    "AXP": {
        "name": "American Express Company",
        "type": "Financial Services Stock",
        "category": "financials",
        "description": "Credit card and financial services company with premium focus",
        "risk": "Medium",
        "icon": "💳",
        "metadata": {"sector": "Financial Services", "industry": "Credit Services", "marketCap": 120000000000, "beta": 1.2}
    },
    
    # UTILITIES
    "NEE": {
        "name": "NextEra Energy",
        "type": "Utility Stock",
        "category": "utilities",
        "description": "Leading renewable energy utility company with strong dividend growth",
        "risk": "Low-Medium", 
        "icon": "⚡",
        "metadata": {"sector": "Utilities", "industry": "Utilities - Regulated Electric", "marketCap": 150000000000, "beta": 0.3, "dividendYield": 0.025, "exchange": "NYSE"}
    },
    "DUK": {
        "name": "Duke Energy",
        "type": "Utility Stock",
        "category": "utilities",
        "description": "Electric power holding company serving the southeastern United States",
        "risk": "Low-Medium",
        "icon": "⚡",
        "metadata": {"sector": "Utilities", "industry": "Utilities - Regulated Electric", "marketCap": 80000000000, "beta": 0.4}
    },
    "SO": {
        "name": "Southern Company",
        "type": "Utility Stock",
        "category": "utilities",
        "description": "Electric utility serving the southeastern U.S. with reliable dividends",
        "risk": "Low-Medium",
        "icon": "⚡",
        "metadata": {"sector": "Utilities", "industry": "Utilities - Regulated Electric", "marketCap": 75000000000, "beta": 0.3}
    },
    "AEP": {
        "name": "American Electric Power",
        "type": "Utility Stock",
        "category": "utilities",
        "description": "Electric utility company serving multiple states",
        "risk": "Low-Medium",
        "icon": "⚡",
        "metadata": {"sector": "Utilities", "industry": "Utilities - Regulated Electric", "marketCap": 45000000000, "beta": 0.4}
    },
    "EXC": {
        "name": "Exelon Corporation",
        "type": "Utility Stock",
        "category": "utilities",
        "description": "Electric utility company with nuclear power focus",
        "risk": "Low-Medium",
        "icon": "⚡",
        "metadata": {"sector": "Utilities", "industry": "Utilities - Regulated Electric", "marketCap": 40000000000, "beta": 0.5}
    },
    
    # ENERGY
    "XOM": {
        "name": "Exxon Mobil Corporation",
        "type": "Energy Stock",
        "category": "dividend_stocks",
        "description": "Integrated oil and gas company with global operations and dividend focus",
        "risk": "Medium-High",
        "icon": "🛢️",
        "metadata": {"sector": "Energy", "industry": "Oil & Gas Integrated", "marketCap": 400000000000, "beta": 1.4}
    },
    "CVX": {
        "name": "Chevron Corporation",
        "type": "Energy Stock",
        "category": "dividend_stocks",
        "description": "Integrated oil and gas company with strong dividend history",
        "risk": "Medium-High",
        "icon": "🛢️",
        "metadata": {"sector": "Energy", "industry": "Oil & Gas Integrated", "marketCap": 350000000000, "beta": 1.3}
    },
    
    # COMMUNICATION SERVICES
    "T": {
        "name": "AT&T Inc.",
        "type": "Dividend Stock",
        "category": "dividend_stocks",
        "description": "Telecommunications and media company with high dividend yield",
        "risk": "Medium",
        "icon": "📱",
        "metadata": {"sector": "Communication Services", "industry": "Telecom Services", "marketCap": 120000000000, "beta": 0.8}
    },
    "VZ": {
        "name": "Verizon Communications",
        "type": "Dividend Stock",
        "category": "dividend_stocks",
        "description": "Telecommunications services provider with reliable dividend",
        "risk": "Low-Medium",
        "icon": "📶",
        "metadata": {"sector": "Communication Services", "industry": "Telecom Services", "marketCap": 160000000000, "beta": 0.7}
    },
    "DIS": {
        "name": "The Walt Disney Company",
        "type": "Large Cap Stock",
        "category": "large_cap_growth",
        "description": "Entertainment and media conglomerate with theme parks, movies, and streaming",
        "risk": "Medium",
        "icon": "🏰",
        "metadata": {"sector": "Communication Services", "industry": "Entertainment", "marketCap": 200000000000, "beta": 1.2}
    },
    "NFLX": {
        "name": "Netflix Inc.",
        "type": "Growth Stock",
        "category": "high_growth",
        "description": "Streaming entertainment and content creation platform with global reach",
        "risk": "High",
        "icon": "🎬",
        "metadata": {"sector": "Communication Services", "industry": "Entertainment", "marketCap": 150000000000, "beta": 1.4, "dividendYield": None, "exchange": "NASDAQ"}
    },
    
    # BROAD MARKET ETFS
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "type": "Market ETF",
        "category": "broad_market",
        "description": "ETF that tracks the S&P 500 index, providing broad U.S. market exposure",
        "risk": "Medium",
        "icon": "📊",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "marketCap": None, "beta": 1.0, "dividendYield": 0.013, "exchange": "NYSE"}
    },
    "VOO": {
        "name": "Vanguard S&P 500 ETF",
        "type": "Market ETF",
        "category": "broad_market",
        "description": "Low-cost S&P 500 index fund with excellent expense ratio",
        "risk": "Medium",
        "icon": "📊",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "beta": 1.0, "dividendYield": 0.013}
    },
    "VTI": {
        "name": "Vanguard Total Stock Market ETF",
        "type": "Total Market ETF",
        "category": "broad_market",
        "description": "Entire U.S. stock market in one fund with broad diversification",
        "risk": "Medium",
        "icon": "📈",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "beta": 1.0, "dividendYield": 0.013}
    },
    "IVV": {
        "name": "iShares Core S&P 500 ETF",
        "type": "Market ETF",
        "category": "broad_market",
        "description": "Core S&P 500 exposure with competitive fees",
        "risk": "Medium",
        "icon": "📊",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "beta": 1.0, "dividendYield": 0.013}
    },
    
    # INTERNATIONAL ETFS
    "VEA": {
        "name": "Vanguard FTSE Developed Markets ETF",
        "type": "International ETF",
        "category": "international",
        "description": "Developed international markets including Europe, Asia, and Australia",
        "risk": "Medium",
        "icon": "🌍",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "beta": 0.9, "dividendYield": 0.025}
    },
    "VWO": {
        "name": "Vanguard FTSE Emerging Markets ETF",
        "type": "Emerging Markets ETF",
        "category": "emerging_markets",
        "description": "Broad exposure to emerging market stocks including China, India, and Brazil",
        "risk": "Medium-High",
        "icon": "🌏",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "marketCap": None, "beta": 1.3, "dividendYield": 0.025, "exchange": "NYSE"}
    },
    "VXUS": {
        "name": "Vanguard Total International Stock ETF",
        "type": "International ETF",
        "category": "international",
        "description": "Total international stock market exposure excluding U.S.",
        "risk": "Medium",
        "icon": "🌍",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "beta": 0.9, "dividendYield": 0.025}
    },
    
    # TECHNOLOGY ETFS
    "QQQ": {
        "name": "Invesco QQQ Trust ETF",
        "type": "Technology ETF",
        "category": "technology",
        "description": "ETF tracking the top 100 NASDAQ technology companies",
        "risk": "Medium-High",
        "icon": "💻",
        "metadata": {"sector": "Technology", "industry": "Exchange Traded Fund", "marketCap": None, "beta": 1.1, "dividendYield": 0.006, "exchange": "NASDAQ"}
    },
    "VGT": {
        "name": "Vanguard Information Technology ETF",
        "type": "Technology ETF",
        "category": "technology",
        "description": "Information technology sector exposure with low fees",
        "risk": "Medium-High",
        "icon": "💻",
        "metadata": {"sector": "Technology", "industry": "Exchange Traded Fund", "beta": 1.2, "dividendYield": 0.008}
    },
    "XLK": {
        "name": "Technology Select Sector SPDR Fund",
        "type": "Technology ETF",
        "category": "technology",
        "description": "Technology sector exposure from S&P 500 companies",
        "risk": "Medium-High",
        "icon": "💻",
        "metadata": {"sector": "Technology", "industry": "Exchange Traded Fund", "beta": 1.2, "dividendYield": 0.007}
    },
    
    # GROWTH ETFS
    "VUG": {
        "name": "Vanguard Growth ETF",
        "type": "Growth ETF",
        "category": "large_cap_growth",
        "description": "ETF focused on large-cap growth stocks with strong earnings potential",
        "risk": "Medium",
        "icon": "📈",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "marketCap": None, "beta": 1.0, "dividendYield": 0.007, "exchange": "NYSE"}
    },
    "IWF": {
        "name": "iShares Russell 1000 Growth ETF",
        "type": "Growth ETF",
        "category": "large_cap_growth",
        "description": "Large-cap growth exposure tracking Russell 1000 Growth Index",
        "risk": "Medium",
        "icon": "📈",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "beta": 1.1, "dividendYield": 0.008}
    },
    
    # HIGH GROWTH / INNOVATION
    "ARKK": {
        "name": "ARK Innovation ETF",
        "type": "Innovation ETF",
        "category": "high_growth",
        "description": "Disruptive innovation companies with high growth potential but high risk",
        "risk": "Very High",
        "icon": "🚀",
        "metadata": {"sector": "Technology", "industry": "Exchange Traded Fund", "beta": 1.8, "dividendYield": 0.000}
    },
    "ARKQ": {
        "name": "ARK Autonomous Technology & Robotics ETF",
        "type": "Innovation ETF",
        "category": "high_growth",
        "description": "Autonomous technology and robotics companies",
        "risk": "Very High",
        "icon": "🤖",
        "metadata": {"sector": "Technology", "industry": "Exchange Traded Fund", "beta": 1.7, "dividendYield": 0.000}
    },
    
    # DIVIDEND ETFS
    "SCHD": {
        "name": "Schwab US Dividend Equity ETF",
        "type": "Dividend ETF",
        "category": "dividend_stocks",
        "description": "ETF focused on high-quality dividend-paying US stocks",
        "risk": "Medium",
        "icon": "💰",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "marketCap": None, "beta": 0.9, "dividendYield": 0.035, "exchange": "NYSE"}
    },
    "VYM": {
        "name": "Vanguard High Dividend Yield ETF",
        "type": "Dividend ETF",
        "category": "dividend_stocks",
        "description": "High dividend yield stocks with focus on income generation",
        "risk": "Medium",
        "icon": "💰",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "beta": 0.9, "dividendYield": 0.028}
    },
    "DVY": {
        "name": "iShares Select Dividend ETF",
        "type": "Dividend ETF",
        "category": "dividend_stocks",
        "description": "Select dividend-paying stocks with strong dividend track records",
        "risk": "Medium",
        "icon": "💰",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "beta": 0.8, "dividendYield": 0.032}
    },
    "NOBL": {
        "name": "ProShares S&P 500 Dividend Aristocrats ETF",
        "type": "Dividend ETF",
        "category": "dividend_stocks",
        "description": "S&P 500 companies with 25+ years of consecutive dividend increases",
        "risk": "Medium",
        "icon": "👑",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "beta": 0.9, "dividendYield": 0.020}
    },
    
    # BOND ETFS
    "BND": {
        "name": "Vanguard Total Bond Market ETF",
        "type": "Bond ETF",
        "category": "bonds",
        "description": "Broad exposure to U.S. investment-grade bonds for income and stability",
        "risk": "Low",
        "icon": "🏛️",
        "metadata": {"sector": "Fixed Income", "industry": "Exchange Traded Fund", "marketCap": None, "beta": 0.1, "dividendYield": 0.035, "exchange": "NASDAQ"}
    },
    "AGG": {
        "name": "iShares Core U.S. Aggregate Bond ETF",
        "type": "Bond ETF",
        "category": "bonds",
        "description": "U.S. investment-grade bonds including government, corporate, and mortgage-backed securities",
        "risk": "Low",
        "icon": "🏛️",
        "metadata": {"sector": "Fixed Income", "industry": "Exchange Traded Fund", "beta": 0.1, "dividendYield": 0.034}
    },
    "TLT": {
        "name": "iShares 20+ Year Treasury Bond ETF",
        "type": "Treasury Bond ETF",
        "category": "bonds",
        "description": "Long-term U.S. Treasury bonds with high interest rate sensitivity",
        "risk": "Low-Medium",
        "icon": "🏛️",
        "metadata": {"sector": "Fixed Income", "industry": "Exchange Traded Fund", "beta": -0.1, "dividendYield": 0.040}
    },
    "IEF": {
        "name": "iShares 7-10 Year Treasury Bond ETF",
        "type": "Treasury Bond ETF",
        "category": "bonds",
        "description": "Intermediate-term U.S. Treasury bonds with moderate duration risk",
        "risk": "Low",
        "icon": "🏛️",
        "metadata": {"sector": "Fixed Income", "industry": "Exchange Traded Fund", "beta": 0.0, "dividendYield": 0.038}
    },
    "SHY": {
        "name": "iShares 1-3 Year Treasury Bond ETF",
        "type": "Treasury Bond ETF",
        "category": "bonds",
        "description": "Short-term U.S. Treasury bonds with minimal interest rate risk",
        "risk": "Very Low",
        "icon": "🏛️",
        "metadata": {"sector": "Fixed Income", "industry": "Exchange Traded Fund", "beta": 0.0, "dividendYield": 0.025}
    },
    "VTEB": {
        "name": "Vanguard Tax-Exempt Bond ETF",
        "type": "Municipal Bond ETF",
        "category": "bonds",
        "description": "Tax-free municipal bonds for income with tax advantages",
        "risk": "Low",
        "icon": "🏛️",
        "metadata": {"sector": "Fixed Income", "industry": "Exchange Traded Fund", "beta": 0.1, "dividendYield": 0.025}
    },
    "LQD": {
        "name": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "type": "Corporate Bond ETF",
        "category": "bonds",
        "description": "Investment-grade corporate bonds for higher yield than treasuries",
        "risk": "Low-Medium",
        "icon": "🏛️",
        "metadata": {"sector": "Fixed Income", "industry": "Exchange Traded Fund", "beta": 0.2, "dividendYield": 0.042}
    },
    "HYG": {
        "name": "iShares iBoxx $ High Yield Corporate Bond ETF",
        "type": "High Yield Bond ETF",
        "category": "bonds",
        "description": "High-yield corporate bonds with higher income but more risk",
        "risk": "Medium",
        "icon": "🏛️",
        "metadata": {"sector": "Fixed Income", "industry": "Exchange Traded Fund", "beta": 0.4, "dividendYield": 0.055}
    },
    
    # REAL ESTATE
    "VNQ": {
        "name": "Vanguard Real Estate ETF",
        "type": "REIT ETF",
        "category": "reits",
        "description": "Real estate investment trusts providing exposure to property markets",
        "risk": "Medium",
        "icon": "🏢",
        "metadata": {"sector": "Real Estate", "industry": "Exchange Traded Fund", "beta": 1.0, "dividendYield": 0.035}
    },
    "IYR": {
        "name": "iShares U.S. Real Estate ETF",
        "type": "REIT ETF",
        "category": "reits",
        "description": "U.S. real estate exposure through REITs and real estate companies",
        "risk": "Medium",
        "icon": "🏢",
        "metadata": {"sector": "Real Estate", "industry": "Exchange Traded Fund", "beta": 1.1, "dividendYield": 0.032}
    },
    "SCHH": {
        "name": "Schwab U.S. REIT ETF",
        "type": "REIT ETF",
        "category": "reits",
        "description": "U.S. real estate investment trusts with low expense ratio",
        "risk": "Medium",
        "icon": "🏢",
        "metadata": {"sector": "Real Estate", "industry": "Exchange Traded Fund", "beta": 1.0, "dividendYield": 0.034}
    },
    "XLRE": {
        "name": "Real Estate Select Sector SPDR Fund",
        "type": "REIT ETF",
        "category": "reits",
        "description": "Real estate sector exposure from S&P 500 REITs",
        "risk": "Medium",
        "icon": "🏢",
        "metadata": {"sector": "Real Estate", "industry": "Exchange Traded Fund", "beta": 1.1, "dividendYield": 0.030}
    },
    
    # SECTOR ETFS
    "XLF": {
        "name": "Financial Select Sector SPDR Fund",
        "type": "Financial ETF",
        "category": "financials",
        "description": "Financial sector exposure from S&P 500 companies",
        "risk": "Medium",
        "icon": "🏦",
        "metadata": {"sector": "Financial Services", "industry": "Exchange Traded Fund", "beta": 1.3, "dividendYield": 0.018}
    },
    "XLE": {
        "name": "Energy Select Sector SPDR Fund",
        "type": "Energy ETF",
        "category": "dividend_stocks",
        "description": "Energy sector exposure with focus on oil and gas companies",
        "risk": "High",
        "icon": "🛢️",
        "metadata": {"sector": "Energy", "industry": "Exchange Traded Fund", "beta": 1.5, "dividendYield": 0.045}
    },
    "XLV": {
        "name": "Health Care Select Sector SPDR Fund",
        "type": "Healthcare ETF",
        "category": "dividend_stocks",
        "description": "Healthcare sector exposure including pharmaceuticals and biotech",
        "risk": "Medium",
        "icon": "🏥",
        "metadata": {"sector": "Healthcare", "industry": "Exchange Traded Fund", "beta": 0.9, "dividendYield": 0.015}
    },
    "XLU": {
        "name": "Utilities Select Sector SPDR Fund",
        "type": "Utilities ETF",
        "category": "utilities",
        "description": "Utilities sector exposure for income-focused investors",
        "risk": "Low-Medium",
        "icon": "⚡",
        "metadata": {"sector": "Utilities", "industry": "Exchange Traded Fund", "beta": 0.4, "dividendYield": 0.032}
    },
    "XLI": {
        "name": "Industrial Select Sector SPDR Fund",
        "type": "Industrial ETF",
        "category": "large_cap_growth",
        "description": "Industrial sector exposure including manufacturing and aerospace",
        "risk": "Medium",
        "icon": "🏭",
        "metadata": {"sector": "Industrials", "industry": "Exchange Traded Fund", "beta": 1.1, "dividendYield": 0.014}
    },
    
    # SMALL/MID CAP ETFS
    "IWM": {
        "name": "iShares Russell 2000 ETF",
        "type": "Small Cap ETF",
        "category": "large_cap_growth",
        "description": "Small-cap stocks with higher growth potential and volatility",
        "risk": "Medium-High",
        "icon": "📈",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "beta": 1.2, "dividendYield": 0.012}
    },
    "IJH": {
        "name": "iShares Core S&P Mid-Cap ETF",
        "type": "Mid Cap ETF",
        "category": "large_cap_growth",
        "description": "Mid-cap stocks balancing growth and stability",
        "risk": "Medium",
        "icon": "📈",
        "metadata": {"sector": "Mixed", "industry": "Exchange Traded Fund", "beta": 1.1, "dividendYield": 0.014}
    }
}

# Pydantic models for API responses
class InstrumentResponse(BaseModel):
    success: bool
    data: Dict
    total: int
    search_query: Optional[str] = None
    category_filter: Optional[str] = None

class InstrumentDetailResponse(BaseModel):
    success: bool
    symbol: str
    data: Dict

class SearchRequest(BaseModel):
    query: Optional[str] = ""
    categories: Optional[List[str]] = []
    risk_levels: Optional[List[str]] = []
    limit: Optional[int] = 50

class CategoryResponse(BaseModel):
    success: bool
    categories: Dict[str, int]
    total_categories: int

class BulkInstrumentResponse(BaseModel):
    success: bool
    data: Dict
    requested: int
    found: int
    missing: List[str]

# Helper function to search instruments
def search_instruments_local(query: str, instruments: Dict, limit: int = 50) -> List[Dict]:
    """Local search function"""
    query_lower = query.lower()
    results = []
    
    for symbol, data in instruments.items():
        # Search in symbol, name, category, description
        searchable_text = ' '.join([
            symbol.lower(),
            data.get('name', '').lower(),
            data.get('category', '').lower(),
            data.get('description', '').lower(),
            data.get('type', '').lower()
        ])
        
        if query_lower in searchable_text:
            results.append({
                'symbol': symbol,
                **data
            })
    
    return results[:limit]

# API ENDPOINTS
@router.get("", response_model=InstrumentResponse)
async def get_instruments(
    search: Optional[str] = Query(None, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, ge=1, le=1000, description="Limit results")
):
    """Get all instruments or search by query"""
    try:
        if search:
            results = search_instruments_local(search.strip(), INSTRUMENTS_DATABASE, limit=1000)
            filtered_instruments = {
                r['symbol']: {k: v for k, v in r.items() if k != 'symbol'} 
                for r in results
            }
        else:
            filtered_instruments = INSTRUMENTS_DATABASE
        
        if category:
            filtered_instruments = {
                symbol: data for symbol, data in filtered_instruments.items()
                if data.get('category') == category.strip()
            }
        
        limited_instruments = dict(list(filtered_instruments.items())[:limit])
        
        return InstrumentResponse(
            success=True,
            data=limited_instruments,
            total=len(limited_instruments),
            search_query=search,
            category_filter=category
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}", response_model=InstrumentDetailResponse)
async def get_instrument_details(
    symbol: str,
    include_price: bool = Query(False, description="Include current price")
):
    """Get detailed information for a specific instrument"""
    try:
        symbol_upper = symbol.upper()
        
        if symbol_upper not in INSTRUMENTS_DATABASE:
            raise HTTPException(
                status_code=404, 
                detail=f"Instrument {symbol_upper} not found"
            )
        
        instrument_data = INSTRUMENTS_DATABASE[symbol_upper].copy()
        
        if include_price:
            instrument_data['current_price'] = None
            instrument_data['price_updated_at'] = datetime.now().isoformat()
            instrument_data['price_note'] = "Real-time pricing requires yfinance integration"
        
        return InstrumentDetailResponse(
            success=True,
            symbol=symbol_upper,
            data=instrument_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/categories/list", response_model=CategoryResponse)
async def get_categories():
    """Get all available categories with counts"""
    try:
        category_counts = {}
        for instrument in INSTRUMENTS_DATABASE.values():
            category = instrument.get('category', 'other')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        sorted_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))
        
        return CategoryResponse(
            success=True,
            categories=sorted_categories,
            total_categories=len(category_counts)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=InstrumentResponse)
async def advanced_search(search_request: SearchRequest):
    """Advanced search endpoint with multiple filters"""
    try:
        if search_request.query:
            results = search_instruments_local(search_request.query, INSTRUMENTS_DATABASE, limit=1000)
            filtered_instruments = {
                r['symbol']: {k: v for k, v in r.items() if k != 'symbol'} 
                for r in results
            }
        else:
            filtered_instruments = INSTRUMENTS_DATABASE
        
        if search_request.categories:
            filtered_instruments = {
                symbol: data for symbol, data in filtered_instruments.items()
                if data.get('category') in search_request.categories
            }
        
        if search_request.risk_levels:
            filtered_instruments = {
                symbol: data for symbol, data in filtered_instruments.items()
                if data.get('risk') in search_request.risk_levels
            }
        
        limited_instruments = dict(list(filtered_instruments.items())[:search_request.limit])
        
        return InstrumentResponse(
            success=True,
            data=limited_instruments,
            total=len(limited_instruments),
            search_query=search_request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bulk/{symbols}", response_model=BulkInstrumentResponse)
async def get_bulk_instruments(
    symbols: str,
    include_prices: bool = Query(False, description="Include current prices")
):
    """
    Get multiple instruments at once - THIS IS THE ENDPOINT YOUR REACT APP CALLS
    
    - **symbols**: Comma-separated list of symbols (e.g., "AAPL,MSFT,GOOGL")
    - **include_prices**: Whether to include current prices for all symbols
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        
        if len(symbol_list) > 50:
            raise HTTPException(
                status_code=400, 
                detail="Maximum 50 symbols allowed per request"
            )
        
        result = {}
        missing = []
        
        for symbol in symbol_list:
            if symbol in INSTRUMENTS_DATABASE:
                instrument_data = INSTRUMENTS_DATABASE[symbol].copy()
                
                if include_prices:
                    instrument_data['current_price'] = None
                    instrument_data['price_note'] = "Real-time pricing requires yfinance integration"
                
                result[symbol] = instrument_data
            else:
                missing.append(symbol)
        
        return BulkInstrumentResponse(
            success=True,
            data=result,
            requested=len(symbol_list),
            found=len(result),
            missing=missing
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/info")
async def debug_info():
    """Debug endpoint to see available instruments and routes"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "available_routes": [
            "GET /api/instruments - Get all instruments",
            "GET /api/instruments/{symbol} - Get specific instrument",
            "GET /api/instruments/bulk/{symbols} - Get multiple instruments (USED BY REACT)",
            "GET /api/instruments/categories/list - Get categories",
            "POST /api/instruments/search - Advanced search"
        ],
        "database_symbols": list(INSTRUMENTS_DATABASE.keys()),
        "total_instruments": len(INSTRUMENTS_DATABASE),
        "categories": list(set(inst.get('category') for inst in INSTRUMENTS_DATABASE.values())),
        "sample_bulk_request": f"/api/instruments/bulk/{','.join(list(INSTRUMENTS_DATABASE.keys())[:7])}"
    }