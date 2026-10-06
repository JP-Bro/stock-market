import numpy as np
import pandas as pd
import config

def run_quant_backtest(
    df_test,
    predictions,
    label_map_inv={0: -1, 1: 0, 2: 1},
    cost_bps=config.TRANSACTION_COST_BPS
):
    """
    Executes professional institutional financial backtest on held-out test data:
    - Eliminates look-ahead bias via 1-day execution shift
    - Implements Marcos López de Prado Triple Barrier Trade Execution:
      * Enter Long on BUY (+1)
      * Intraday Take-Profit Barrier Exit (+1.5x Volatility)
      * Intraday Stop-Loss Barrier Exit (-1.0x Volatility)
      * Vertical Time-Horizon Exit (10 days)
      * Signal Reversal Exit on SELL (-1)
    - Deducts realistic transaction costs, exchange fees, STT, and slippage (cost_bps)
    - Tracks trade-level duration, Profit Factor, Sortino Ratio, Calmar Ratio, and Max Drawdown
    """
    df_bt = df_test.copy()
    raw_signals = np.array([label_map_inv[p] for p in predictions])
    n = len(df_bt)
    
    close = df_bt['Close'].values
    high = df_bt['High'].values
    low = df_bt['Low'].values
    
    # Calculate daily volatility: prefer norm_atr, fallback to 1.5%
    if 'norm_atr' in df_bt.columns:
        daily_vol = df_bt['norm_atr'].values
    else:
        daily_vol = np.full(n, 0.015)
    daily_vol = np.maximum(daily_vol, config.MIN_VOLATILITY_FLOOR)
    
    # Simulate realistic trade positions with barrier exits
    positions = np.zeros(n, dtype=float)
    in_trade = False
    entry_idx = 0
    entry_p = 0.0
    upper_barrier = 0.0
    lower_barrier = 0.0
    
    for t in range(n):
        sig = raw_signals[t]
        
        # Check active trade exit conditions if currently holding
        if in_trade:
            holding_bars = t - entry_idx
            
            # 1. Take Profit hit
            if high[t] >= upper_barrier:
                positions[t] = 0.0
                in_trade = False
            # 2. Stop Loss hit
            elif low[t] <= lower_barrier:
                positions[t] = 0.0
                in_trade = False
            # 3. Time horizon reached
            elif holding_bars >= config.BARRIER_TIME_HORIZON:
                positions[t] = 0.0
                in_trade = False
            # 4. Explicit Sell signal
            elif sig == -1:
                positions[t] = 0.0
                in_trade = False
            else:
                positions[t] = 1.0 # Maintain position
                
        # If not currently in trade, check for new BUY entry
        if not in_trade and sig == 1:
            in_trade = True
            entry_idx = t
            entry_p = close[t]
            vol_t = daily_vol[t]
            upper_barrier = entry_p * (1.0 + (config.PROFIT_TAKE_VOL_MULT * vol_t))
            lower_barrier = entry_p * (1.0 - (config.STOP_LOSS_VOL_MULT * vol_t))
            positions[t] = 1.0
            
    df_bt['position'] = positions
    df_bt['daily_return'] = df_bt['Close'].pct_change().fillna(0.0)
    
    # 1-Day Lagged Execution (Trade decision at t close earns return from t to t+1)
    df_bt['exec_position'] = df_bt['position'].shift(1).fillna(0.0)
    
    # Transaction cost charged on position transitions: |pos_t - pos_{t-1}|
    pos_changes = (df_bt['position'] - df_bt['position'].shift(1).fillna(0.0)).abs()
    cost_rate = cost_bps / 10000.0
    df_bt['tx_costs'] = pos_changes * cost_rate
    
    # Gross & Net Strategy Returns
    df_bt['gross_return'] = df_bt['exec_position'] * df_bt['daily_return']
    df_bt['strategy_return'] = df_bt['gross_return'] - df_bt['tx_costs']
    
    # Cumulative Wealth Curves (Compounded Returns)
    df_bt['cum_benchmark'] = (1.0 + df_bt['daily_return']).cumprod()
    df_bt['cum_strategy'] = (1.0 + df_bt['strategy_return']).cumprod()
    
    # Performance Metrics
    total_benchmark_ret = float(df_bt['cum_benchmark'].iloc[-1] - 1.0)
    total_strategy_ret = float(df_bt['cum_strategy'].iloc[-1] - 1.0)
    
    trading_days = len(df_bt)
    years = max(trading_days / 252.0, 0.01)
    
    cagr_strategy = (1.0 + total_strategy_ret) ** (1.0 / years) - 1.0 if total_strategy_ret > -1.0 else -1.0
    cagr_benchmark = (1.0 + total_benchmark_ret) ** (1.0 / years) - 1.0 if total_benchmark_ret > -1.0 else -1.0
    
    # Sharpe Ratio (6% p.a. risk-free rate)
    rf_daily = 0.06 / 252.0
    excess_returns = df_bt['strategy_return'] - rf_daily
    excess_std = excess_returns.std()
    sharpe = float(np.sqrt(252.0) * excess_returns.mean() / (excess_std + 1e-8)) if excess_std > 0 else 0.0
    
    # Sortino Ratio (Downside deviation only)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 1 else 1e-8
    sortino = float(np.sqrt(252.0) * excess_returns.mean() / (downside_std + 1e-8))
    
    # Maximum Drawdown & Drawdown Series
    peak = df_bt['cum_strategy'].cummax()
    drawdown = (df_bt['cum_strategy'] - peak) / peak
    max_drawdown = float(drawdown.min())
    
    # Calmar Ratio
    calmar = float(cagr_strategy / abs(max_drawdown)) if abs(max_drawdown) > 0.001 else 0.0
    
    # Trade-Level Attribution
    active_days = df_bt[df_bt['exec_position'] > 0]
    daily_win_rate = float((active_days['strategy_return'] > 0).mean()) if len(active_days) > 0 else 0.0
    
    gross_gains = df_bt[df_bt['strategy_return'] > 0]['strategy_return'].sum()
    gross_losses = abs(df_bt[df_bt['strategy_return'] < 0]['strategy_return'].sum())
    profit_factor = float(gross_gains / (gross_losses + 1e-8)) if gross_losses > 0 else (99.0 if gross_gains > 0 else 0.0)
    
    total_trades = int(pos_changes.sum() / 2.0)
    avg_hold_days = float(len(active_days) / max(total_trades, 1))
    
    metrics = {
        'total_strategy_return': total_strategy_ret,
        'total_benchmark_return': total_benchmark_ret,
        'cagr_strategy': cagr_strategy,
        'cagr_benchmark': cagr_benchmark,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'daily_win_rate': daily_win_rate,
        'profit_factor': profit_factor,
        'total_trades': total_trades,
        'avg_hold_days': avg_hold_days,
        'trading_days': trading_days,
        'cost_bps_applied': cost_bps
    }
    
    return df_bt, metrics
