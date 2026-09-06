import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
import config

def load_abcapital_data(start_date=config.START_DATE, end_date=None):
    """
    Dynamically fetches live real-time OHLCV market data for Aditya Birla Capital Ltd (ABCAPITAL.NS)
    from Yahoo Finance API up to the current moment. No hardcoded or static CSV files are used.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    print(f"[LIVE DATA] Fetching market data from Yahoo Finance for {config.TICKER} ({start_date} to {end_date})...")
    
    try:
        # Download live market data on the fly from Yahoo Finance API
        df = yf.download(config.TICKER, start=start_date, end=end_date, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(config.TICKER, level=1, axis=1)
            
        df = df.dropna().copy()
        
        if len(df) == 0:
            raise ValueError("Yahoo Finance returned an empty dataset.")
            
        print(f"[SUCCESS] Downloaded {len(df)} live trading days up to {df.index[-1].strftime('%Y-%m-%d')}.")
        
        # Save cache for inspection
        cache_path = os.path.join(config.DATA_DIR, "abcapital_live_cache.csv")
        df.to_csv(cache_path)
        
        return df
        
    except Exception as e:
        print(f"[WARNING] Live download failed ({e}). Checking local cache...")
        cache_path = os.path.join(config.DATA_DIR, "abcapital_live_cache.csv")
        if os.path.exists(cache_path):
            print(f"Loading cached market data from {cache_path}...")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df
        else:
            raise e

if __name__ == "__main__":
    df = load_abcapital_data()
    print("Latest 5 Trading Days:")
    print(df.tail())
