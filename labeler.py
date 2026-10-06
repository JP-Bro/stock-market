import pandas as pd
import numpy as np
import config

def create_triple_barrier_labels(
    df,
    holding_period=config.BARRIER_TIME_HORIZON,
    pt_mult=config.PROFIT_TAKE_VOL_MULT,
    sl_mult=config.STOP_LOSS_VOL_MULT,
    min_vol=config.MIN_VOLATILITY_FLOOR
):
    """
    Marcos López de Prado's Triple Barrier Labeling Method for Quantitative Daily Swing Trading.
    
    Barriers:
    1. Upper Barrier (Take-Profit): Close_t * (1 + pt_mult * sigma_t)
    2. Lower Barrier (Stop-Loss):   Close_t * (1 - sl_mult * sigma_t)
    3. Vertical Barrier (Time):     t + holding_period trading days (e.g., 10 days / 2 weeks)
    
    Path-dependent outcome for each bar t:
    - +1 (BUY/PROFIT): Upper barrier touched first before stop-loss or expiration.
    - -1 (SELL/STOP-OUT): Lower barrier touched first.
    -  0 (HOLD/NEUTRAL): Neither barrier touched within holding period (or price within noise).
    """
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    n = len(df)
    
    # Calculate daily volatility: prefer ATR/Close, fallback to rolling std of log returns
    if 'norm_atr' in df.columns:
        daily_vol = df['norm_atr'].values
    elif 'atr_14' in df.columns:
        daily_vol = (df['atr_14'] / df['Close']).values
    else:
        log_ret = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        daily_vol = log_ret.rolling(14).std().fillna(0.015).values
        
    daily_vol = np.maximum(daily_vol, min_vol)
    
    labels = np.zeros(n, dtype=int)
    touch_types = []
    
    for i in range(n):
        # If not enough future bars for holding period, mark 0
        if i + 1 >= n:
            labels[i] = 0
            touch_types.append("end_of_data")
            continue
            
        p0 = close[i]
        vol = daily_vol[i]
        upper_barrier = p0 * (1.0 + pt_mult * vol)
        lower_barrier = p0 * (1.0 - sl_mult * vol)
        
        horizon_end = min(i + 1 + holding_period, n)
        event_label = 0
        touch_type = "vertical_time_expiration"
        
        for j in range(i + 1, horizon_end):
            h_j = high[j]
            l_j = low[j]
            
            upper_hit = h_j >= upper_barrier
            lower_hit = l_j <= lower_barrier
            
            if upper_hit and not lower_hit:
                event_label = 1
                touch_type = "upper_profit_barrier"
                break
            elif lower_hit and not upper_hit:
                event_label = -1
                touch_type = "lower_stop_barrier"
                break
            elif upper_hit and lower_hit:
                # Wide day hit both: check close relative to entry for conservative attribution
                if close[j] >= p0:
                    event_label = 1
                    touch_type = "upper_profit_barrier_same_day"
                else:
                    event_label = -1
                    touch_type = "lower_stop_barrier_same_day"
                break
                
        # If vertical barrier reached without horizontal hit, inspect net return
        if event_label == 0 and horizon_end > i + 1:
            p_end = close[horizon_end - 1]
            net_ret = (p_end - p0) / p0
            if net_ret > (0.6 * pt_mult * vol):
                event_label = 1
                touch_type = "vertical_positive_drift"
            elif net_ret < (-0.6 * sl_mult * vol):
                event_label = -1
                touch_type = "vertical_negative_drift"
            else:
                event_label = 0
                touch_type = "vertical_neutral_hold"
                
        labels[i] = event_label
        touch_types.append(touch_type)
        
    target_series = pd.Series(labels, index=df.index, name="target")
    return target_series

# Backward-compatible alias for existing imports
def create_quant_target_labels(df, forward_window=config.BARRIER_TIME_HORIZON, profit_mult=config.PROFIT_TAKE_VOL_MULT):
    return create_triple_barrier_labels(df, holding_period=forward_window, pt_mult=profit_mult)
