import pandas as pd
import numpy as np
from candlestick_strategies import detect_candlestick_strategies

def compute_wilder_rsi(series, period=14):
    """
    Computes Relative Strength Index (RSI) using Wilder's Smoothing.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).copy()
    loss = (-delta.where(delta < 0, 0)).copy()
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def engineer_quant_rsi_features(df):
    """
    Engineers advanced quant finance features combining RSI momentum, slope,
    zones, crossings, structural peaks/troughs, divergence, volatility context,
    and verified high-probability candlestick pattern strategies.
    """
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # 1. Multi-Period RSIs
    df['rsi_14'] = compute_wilder_rsi(close, 14)
    df['rsi_7'] = compute_wilder_rsi(close, 7)
    df['rsi_21'] = compute_wilder_rsi(close, 21)
    
    # 2. RSI Velocity & Acceleration
    df['rsi_diff1'] = df['rsi_14'].diff(1)
    df['rsi_diff3'] = df['rsi_14'].diff(3)
    df['rsi_diff5'] = df['rsi_14'].diff(5)
    df['rsi_accel'] = df['rsi_diff1'].diff(1)
    
    # 3. Rolling Z-Scores for Stationarity
    mean_30 = df['rsi_14'].rolling(30).mean()
    std_30 = df['rsi_14'].rolling(30).std() + 1e-8
    df['rsi_zscore_30'] = (df['rsi_14'] - mean_30) / std_30
    
    mean_60 = df['rsi_14'].rolling(60).mean()
    std_60 = df['rsi_14'].rolling(60).std() + 1e-8
    df['rsi_zscore_60'] = (df['rsi_14'] - mean_60) / std_60
    
    # 4. RSI Zones (0: 0-20, 1: 20-30, 2: 30-40, 3: 40-50, 4: 50-60, 5: 60-70, 6: 70-80, 7: 80-90+)
    df['rsi_zone'] = pd.cut(
        df['rsi_14'],
        bins=[-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf],
        labels=[0, 1, 2, 3, 4, 5, 6, 7]
    ).astype(float)
    
    # 5. Level Crossings
    df['cross_30_up'] = ((df['rsi_14'].shift(1) < 30) & (df['rsi_14'] >= 30)).astype(int)
    df['cross_50_up'] = ((df['rsi_14'].shift(1) < 50) & (df['rsi_14'] >= 50)).astype(int)
    df['cross_70_down'] = ((df['rsi_14'].shift(1) > 70) & (df['rsi_14'] <= 70)).astype(int)
    df['cross_50_down'] = ((df['rsi_14'].shift(1) > 50) & (df['rsi_14'] <= 50)).astype(int)
    
    # 6. Price vs RSI Divergence
    def calc_slope(series_window):
        if len(series_window) < 5: return 0.0
        x = np.arange(len(series_window))
        return np.polyfit(x, series_window, 1)[0]
    
    price_slope_10 = close.rolling(10).apply(calc_slope, raw=True)
    rsi_slope_10 = df['rsi_14'].rolling(10).apply(calc_slope, raw=True)
    
    # Bullish Divergence: Price falling (slope < 0), RSI rising (slope > 0)
    df['bullish_div'] = ((price_slope_10 < 0) & (rsi_slope_10 > 0)).astype(int)
    # Bearish Divergence: Price rising (slope > 0), RSI falling (slope < 0)
    df['bearish_div'] = ((price_slope_10 > 0) & (rsi_slope_10 < 0)).astype(int)
    
    # 7. RSI Structure (Higher Highs / Higher Lows / Failure Swings)
    rsi_max_10 = df['rsi_14'].rolling(10).max()
    rsi_min_10 = df['rsi_14'].rolling(10).min()
    df['rsi_rel_pos_10'] = (df['rsi_14'] - rsi_min_10) / (rsi_max_10 - rsi_min_10 + 1e-8)
    
    df['rsi_higher_low'] = ((rsi_min_10 > rsi_min_10.shift(5)) & (df['rsi_14'] > df['rsi_14'].shift(1))).astype(int)
    df['rsi_lower_high'] = ((rsi_max_10 < rsi_max_10.shift(5)) & (df['rsi_14'] < df['rsi_14'].shift(1))).astype(int)
    
    # Failure Swings
    df['bullish_failure_swing'] = ((df['rsi_14'].shift(5) < 35) & (df['rsi_14'] > rsi_max_10.shift(2))).astype(int)
    df['bearish_failure_swing'] = ((df['rsi_14'].shift(5) > 65) & (df['rsi_14'] < rsi_min_10.shift(2))).astype(int)
    
    # 8. Moving Averages & Trend Context
    df['ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['ema_50'] = close.ewm(span=50, adjust=False).mean()
    df['ema_200'] = close.ewm(span=200, adjust=False).mean()
    
    df['dist_ema_20'] = (close - df['ema_20']) / df['ema_20']
    df['dist_ema_50'] = (close - df['ema_50']) / df['ema_50']
    df['trend_50_200'] = (df['ema_50'] > df['ema_200']).astype(int)
    
    # 9. Volatility (ATR)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(span=14, adjust=False).mean()
    df['norm_atr'] = df['atr_14'] / close
    
    # 10. Verified Candlestick Pattern Strategies Detection
    df = detect_candlestick_strategies(df)
    
    return df

FEATURE_COLUMNS = [
    'rsi_14', 'rsi_7', 'rsi_21', 'rsi_diff1', 'rsi_diff3', 'rsi_diff5', 'rsi_accel',
    'rsi_zscore_30', 'rsi_zscore_60', 'rsi_zone', 'cross_30_up', 'cross_50_up',
    'cross_70_down', 'cross_50_down', 'bullish_div', 'bearish_div', 'rsi_rel_pos_10',
    'rsi_higher_low', 'rsi_lower_high', 'bullish_failure_swing', 'bearish_failure_swing',
    'dist_ema_20', 'dist_ema_50', 'trend_50_200', 'norm_atr',
    'sig_three_line_strike', 'sig_three_soldiers_crows', 'sig_morning_evening_star',
    'sig_abandoned_baby', 'sig_engulfing_baseline', 'location_filter_pass', 'volume_filter_pass'
]
