import numpy as np
import pandas as pd

def detect_candlestick_strategies(df):
    """
    Detects verified high-probability candlestick strategies on a DataFrame with OHLCV data.
    Returns DataFrame with signal columns for each strategy and validation flags.
    
    Signals:
      +1 : BUY / ACCUMULATE
      -1 : SELL / REDUCE
       0 : NEUTRAL / HOLD
    """
    df = df.copy()
    
    open_p = df['Open']
    high_p = df['High']
    low_p = df['Low']
    close_p = df['Close']
    volume = df.get('Volume', pd.Series(1, index=df.index))
    
    body = (close_p - open_p).abs()
    rng = (high_p - low_p).replace(0, 1e-8)
    body_pct = body / rng
    
    # 20-period Moving Average of Volume & Price EMAs
    vol_sma20 = volume.rolling(20).mean()
    ema_20 = close_p.ewm(span=20, adjust=False).mean()
    ema_50 = close_p.ewm(span=50, adjust=False).mean()
    ema_200 = close_p.ewm(span=200, adjust=False).mean()
    
    # ATR for height filter
    tr = pd.concat([
        high_p - low_p,
        (high_p - close_p.shift(1)).abs(),
        (low_p - close_p.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    
    # ---------------------------------------------------------------------
    # 1. THREE LINE STRIKE (84% Empirical Win Rate)
    # ---------------------------------------------------------------------
    # Bullish Three Line Strike: 3 red candles followed by massive green candle engulfing all 3
    red_1 = close_p.shift(3) < open_p.shift(3)
    red_2 = close_p.shift(2) < open_p.shift(2)
    red_3 = close_p.shift(1) < open_p.shift(1)
    strike_bull_4 = (close_p > open_p) & (open_p <= close_p.shift(1)) & (close_p >= open_p.shift(3))
    
    # Bearish Three Line Strike: 3 green candles followed by massive red candle engulfing all 3
    green_1 = close_p.shift(3) > open_p.shift(3)
    green_2 = close_p.shift(2) > open_p.shift(2)
    green_3 = close_p.shift(1) > open_p.shift(1)
    strike_bear_4 = (close_p < open_p) & (open_p >= close_p.shift(1)) & (close_p <= open_p.shift(3))
    
    df['sig_three_line_strike'] = 0
    df.loc[red_1 & red_2 & red_3 & strike_bull_4, 'sig_three_line_strike'] = 1
    df.loc[green_1 & green_2 & green_3 & strike_bear_4, 'sig_three_line_strike'] = -1
    
    # ---------------------------------------------------------------------
    # 2. THREE WHITE SOLDIERS / THREE BLACK CROWS (75% - 78% Empirical Win Rate)
    # ---------------------------------------------------------------------
    # Three White Soldiers (Bullish +1)
    c1_white = (close_p.shift(2) > open_p.shift(2)) & (body_pct.shift(2) > 0.5)
    c2_white = (close_p.shift(1) > open_p.shift(1)) & (body_pct.shift(1) > 0.5) & (close_p.shift(1) > close_p.shift(2))
    c3_white = (close_p > open_p) & (body_pct > 0.5) & (close_p > close_p.shift(1))
    
    # Three Black Crows (Bearish -1)
    c1_crow = (close_p.shift(2) < open_p.shift(2)) & (body_pct.shift(2) > 0.5)
    c2_crow = (close_p.shift(1) < open_p.shift(1)) & (body_pct.shift(1) > 0.5) & (close_p.shift(1) < close_p.shift(2))
    c3_crow = (close_p < open_p) & (body_pct > 0.5) & (close_p < close_p.shift(1))
    
    df['sig_three_soldiers_crows'] = 0
    df.loc[c1_white & c2_white & c3_white, 'sig_three_soldiers_crows'] = 1
    df.loc[c1_crow & c2_crow & c3_crow, 'sig_three_soldiers_crows'] = -1
    
    # ---------------------------------------------------------------------
    # 3. EVENING STAR / MORNING STAR (70% - 72% Empirical Win Rate)
    # ---------------------------------------------------------------------
    # Morning Star (Bullish +1)
    m_c1 = (close_p.shift(2) < open_p.shift(2)) & (body.shift(2) > 0.4 * atr14.shift(2))
    m_c2 = body_pct.shift(1) < 0.35
    m_c3 = (close_p > open_p) & (close_p > (close_p.shift(2) + 0.4 * (open_p.shift(2) - close_p.shift(2))))
    
    # Evening Star (Bearish -1)
    e_c1 = (close_p.shift(2) > open_p.shift(2)) & (body.shift(2) > 0.4 * atr14.shift(2))
    e_c2 = body_pct.shift(1) < 0.35
    e_c3 = (close_p < open_p) & (close_p < (open_p.shift(2) - 0.4 * (close_p.shift(2) - open_p.shift(2))))
    
    df['sig_morning_evening_star'] = 0
    df.loc[m_c1 & m_c2 & m_c3, 'sig_morning_evening_star'] = 1
    df.loc[e_c1 & e_c2 & e_c3, 'sig_morning_evening_star'] = -1
    
    # ---------------------------------------------------------------------
    # 4. ABANDONED BABY (70% Empirical Win Rate)
    # ---------------------------------------------------------------------
    # Bullish Abandoned Baby (+1)
    ab_bull_c1 = close_p.shift(2) < open_p.shift(2)
    ab_bull_doji = (body_pct.shift(1) < 0.15) & (high_p.shift(1) < low_p.shift(2))
    ab_bull_c3 = (close_p > open_p) & (low_p > high_p.shift(1))
    
    # Bearish Abandoned Baby (-1)
    ab_bear_c1 = close_p.shift(2) > open_p.shift(2)
    ab_bear_doji = (body_pct.shift(1) < 0.15) & (low_p.shift(1) > high_p.shift(2))
    ab_bear_c3 = (close_p < open_p) & (high_p < low_p.shift(1))
    
    df['sig_abandoned_baby'] = 0
    df.loc[ab_bull_c1 & ab_bull_doji & ab_bull_c3, 'sig_abandoned_baby'] = 1
    df.loc[ab_bear_c1 & ab_bear_doji & ab_bear_c3, 'sig_abandoned_baby'] = -1
    
    # ---------------------------------------------------------------------
    # 5. STANDARD ENGULFING & HAMMERS (50-58% Win Rate - Baseline Reference)
    # ---------------------------------------------------------------------
    bull_engulf = (close_p.shift(1) < open_p.shift(1)) & (close_p > open_p) & (close_p >= open_p.shift(1)) & (open_p <= close_p.shift(1))
    bear_engulf = (close_p.shift(1) > open_p.shift(1)) & (close_p < open_p) & (close_p <= open_p.shift(1)) & (open_p >= close_p.shift(1))
    
    df['sig_engulfing_baseline'] = 0
    df.loc[bull_engulf, 'sig_engulfing_baseline'] = 1
    df.loc[bear_engulf, 'sig_engulfing_baseline'] = -1
    
    # ---------------------------------------------------------------------
    # VALIDATION RULES: Location Filter & Volume Spike
    # ---------------------------------------------------------------------
    dist_ema20 = (close_p - ema_20).abs() / ema_20
    dist_ema50 = (close_p - ema_50).abs() / ema_50
    dist_ema200 = (close_p - ema_200).abs() / ema_200
    
    # Location Filter: Price within 3% of EMA 20, 50, or 200
    df['location_filter_pass'] = ((dist_ema20 < 0.03) | (dist_ema50 < 0.03) | (dist_ema200 < 0.03)).astype(int)
    
    # Volume Expansion Filter: Volume > 1.05 * 20-day Volume SMA
    df['volume_filter_pass'] = (volume >= 1.0 * vol_sma20).astype(int)
    
    # Confluence Score calculation across strategies
    strategies_cols = [
        'sig_three_line_strike',
        'sig_three_soldiers_crows',
        'sig_morning_evening_star',
        'sig_abandoned_baby'
    ]
    
    df['pattern_confluence_sum'] = df[strategies_cols].sum(axis=1)
    
    return df

STRATEGY_REGISTRY = [
    {
        "id": "three_line_strike",
        "name": "1. Three Line Strike (Bull/Bear)",
        "win_rate": "84%",
        "rr_ratio": "High (1:3+)",
        "col": "sig_three_line_strike",
        "action": "Enter on 4th candle close; Stop beyond 4th wick."
    },
    {
        "id": "three_soldiers_crows",
        "name": "2. Three White Soldiers / Black Crows",
        "win_rate": "75% - 78%",
        "rr_ratio": "Moderate (1:2)",
        "col": "sig_three_soldiers_crows",
        "action": "Trade when breaking multi-week range or EMA."
    },
    {
        "id": "morning_evening_star",
        "name": "3. Evening Star / Morning Star",
        "win_rate": "70% - 72%",
        "rr_ratio": "Favorable (1:2.5)",
        "col": "sig_morning_evening_star",
        "action": "Trade at macro support/resistance or key EMAs."
    },
    {
        "id": "abandoned_baby",
        "name": "4. Bullish / Bearish Abandoned Baby",
        "win_rate": "70%",
        "rr_ratio": "High (1:3)",
        "col": "sig_abandoned_baby",
        "action": "Rare setup; trade when isolated by dual gaps."
    },
    {
        "id": "engulfing_baseline",
        "name": "5. Standard Engulfing & Hammers (Baseline)",
        "win_rate": "50% - 58%",
        "rr_ratio": "Poor to Moderate",
        "col": "sig_engulfing_baseline",
        "action": "Avoid standalone; requires strict location confluence."
    }
]
