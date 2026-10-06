import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
import config

def load_market_data(ticker=config.TICKER, start_date=config.START_DATE, end_date=None):
    """
    Dynamically fetches live real-time OHLCV market data for any specified asset ticker
    from Yahoo Finance API. For TMCV.NS, seamlessly leverages continuous stitched history
    from Jan 2021 up to current date. Caches data in config.DATA_DIR.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    safe_ticker = ticker.replace("^", "").replace(".", "_")
    cache_path = os.path.join(config.DATA_DIR, f"{safe_ticker}_market_cache.csv")
    
    print(f"[LIVE DATA] Fetching market data from Yahoo Finance for {ticker} ({start_date} to {end_date})...")
    
    # 1. Check local cache first for continuous historical series
    cached_df = None
    if os.path.exists(cache_path):
        try:
            cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        except Exception:
            cached_df = None

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.levels[1]:
                df = df.xs(ticker, level=1, axis=1)
            elif ticker in df.columns.levels[0]:
                df = df.xs(ticker, level=0, axis=1)
            else:
                df.columns = df.columns.get_level_values(0)
                
        df = df.dropna().copy()
        
        # If live download returned a short post-demerger series (like TMCV.NS) but cache has full continuous history, merge them
        if cached_df is not None and len(cached_df) > len(df):
            # Update cache with latest closing rows
            combined = pd.concat([cached_df, df])
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            df = combined
            
        if len(df) == 0:
            if cached_df is not None:
                return cached_df
            raise ValueError(f"Yahoo Finance returned an empty dataset for {ticker}.")
            
        print(f"[SUCCESS] Loaded {len(df)} live trading days from {df.index[0].strftime('%Y-%m-%d')} up to {df.index[-1].strftime('%Y-%m-%d')}.")
        df.to_csv(cache_path)
        return df
        
    except Exception as e:
        print(f"[WARNING] Live download failed ({e}). Checking local cache...")
        if cached_df is not None:
            print(f"Loading cached market data from {cache_path} ({len(cached_df)} rows)...")
            return cached_df
        raise e

# Backward-compatible alias
def load_abcapital_data(start_date=config.START_DATE, end_date=None):
    return load_market_data(ticker=config.TICKER, start_date=start_date, end_date=end_date)

if __name__ == "__main__":
    df = load_market_data("TMCV.NS", start_date="2021-01-01")
    print("Latest 5 Trading Days:")
    print(df.tail())
