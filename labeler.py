import pandas as pd
import numpy as np
import config

def create_quant_target_labels(df, forward_window=config.FORWARD_WINDOW, profit_mult=config.PROFIT_MULTIPLIER):
    """
    Creates target labels using forward returns normalized by ATR volatility band.
    - +1 (BUY): Forward return > +profit_mult * (ATR / Close)
    - -1 (SELL): Forward return < -profit_mult * (ATR / Close)
    - 0 (HOLD/WAIT): Price movement within volatility noise range.
    """
    close = df['Close']
    atr = df['atr_14']
    
    forward_return = (close.shift(-forward_window) - close) / close
    atr_threshold = (atr / close) * profit_mult
    
    target = pd.Series(0, index=df.index)
    target[forward_return > atr_threshold] = 1   # BUY
    target[forward_return < -atr_threshold] = -1 # SELL
    
    return target
