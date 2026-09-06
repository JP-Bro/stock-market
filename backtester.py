import numpy as np
import pandas as pd

def run_quant_backtest(df_test, predictions, label_map_inv={0: -1, 1: 0, 2: 1}):
    """
    Executes financial backtest on test dataset:
    - predictions: mapped array [0 (SELL), 1 (HOLD), 2 (BUY)]
    """
    df_bt = df_test.copy()
    raw_signals = np.array([label_map_inv[p] for p in predictions])
    
    # 1-day percentage return of closing prices
    df_bt['daily_return'] = df_bt['Close'].pct_change().fillna(0)
    
    # Position: +1 for Long (BUY), -1 for Short/Cash exit (SELL), 0 for Cash/Neutral
    # Here, for equity trading: BUY (+1) = Long position, SELL (-1) = Cash (0.0), HOLD (0) = Maintain previous position
    positions = []
    curr_pos = 0.0
    for sig in raw_signals:
        if sig == 1:
            curr_pos = 1.0 # Long
        elif sig == -1:
            curr_pos = 0.0 # Cash / Exit
        # If HOLD (0), maintain curr_pos
        positions.append(curr_pos)
        
    df_bt['position'] = positions
    # Shift position by 1 day to prevent look-ahead bias
    df_bt['strategy_return'] = df_bt['position'].shift(1) * df_bt['daily_return']
    df_bt['strategy_return'] = df_bt['strategy_return'].fillna(0)
    
    # Cumulative returns
    df_bt['cum_benchmark'] = (1 + df_bt['daily_return']).cumprod()
    df_bt['cum_strategy'] = (1 + df_bt['strategy_return']).cumprod()
    
    # Performance Metrics
    total_benchmark_ret = df_bt['cum_benchmark'].iloc[-1] - 1.0
    total_strategy_ret = df_bt['cum_strategy'].iloc[-1] - 1.0
    
    trading_days = len(df_bt)
    cagr_strategy = (1 + total_strategy_ret) ** (252.0 / trading_days) - 1.0 if total_strategy_ret > -1 else -1.0
    cagr_benchmark = (1 + total_benchmark_ret) ** (252.0 / trading_days) - 1.0 if total_benchmark_ret > -1 else -1.0
    
    # Sharpe Ratio (Risk-free rate assumed 6% per annum for INR market)
    rf_daily = 0.06 / 252.0
    excess_returns = df_bt['strategy_return'] - rf_daily
    sharpe = np.sqrt(252.0) * excess_returns.mean() / (excess_returns.std() + 1e-8)
    
    # Max Drawdown
    peak = df_bt['cum_strategy'].cummax()
    drawdown = (df_bt['cum_strategy'] - peak) / peak
    max_drawdown = drawdown.min()
    
    # Win Rate
    active_trades = df_bt[df_bt['strategy_return'] != 0]['strategy_return']
    win_rate = (active_trades > 0).mean() if len(active_trades) > 0 else 0.0
    
    metrics = {
        'total_strategy_return': total_strategy_ret,
        'total_benchmark_return': total_benchmark_ret,
        'cagr_strategy': cagr_strategy,
        'cagr_benchmark': cagr_benchmark,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trading_days': trading_days
    }
    
    return df_bt, metrics
