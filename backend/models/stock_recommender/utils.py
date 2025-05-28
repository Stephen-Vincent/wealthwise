from ticker_remap import ticker_remap

def remap_ticker(ticker: str) -> str:
    ticker = ticker.replace(".", "").upper()
    return ticker_remap.get(ticker, ticker)