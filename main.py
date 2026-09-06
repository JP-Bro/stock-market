import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

import config
from data_loader import load_abcapital_data
from feature_engineering import engineer_quant_rsi_features, FEATURE_COLUMNS
from labeler import create_quant_target_labels
from models.xgboost_model import QuantXGBoostModel
from models.deep_lstm import QuantDeepLSTMModel, create_sequences
from models.ensemble import HybridEnsembleModel
from backtester import run_quant_backtest
from predictor import generate_section30_report

def main():
    print(f"=== Starting Quant AI RSI Trading System for {config.ASSET_NAME} ({config.TICKER}) ===")
    
    # 1. Load Data
    df_raw = load_abcapital_data()
    
    # 2. Engineer Features
    print("Engineering multi-period RSI features, Z-scores, divergence, and structure...")
    df_feat = engineer_quant_rsi_features(df_raw)
    
    # 3. Create Labels
    print("Creating ATR-scaled forward target labels (BUY / HOLD / SELL)...")
    target = create_quant_target_labels(df_feat)
    df_feat['target'] = target
    
    # Map target {-1: 0 (SELL), 0: 1 (HOLD), 1: 2 (BUY)}
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {0: -1, 1: 0, 2: 1}
    df_feat['target_mapped'] = df_feat['target'].map(label_map)
    
    df_clean = df_feat.dropna().copy()
    print(f"Clean Dataset Shape: {df_clean.shape}")
    print("Target Distribution:\n", df_clean['target'].value_counts(normalize=True))
    
    X = df_clean[FEATURE_COLUMNS].values
    y = df_clean['target_mapped'].values
    dates = df_clean.index
    
    # 4. Chronological Split: 70% Train, 15% Validation, 15% Test
    n = len(df_clean)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    dates_test = dates[val_end:]
    df_test = df_clean.iloc[val_end:].copy()
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nChronological Split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Out-of-Sample Test Period: {dates[val_end].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # 5. Train XGBoost Model
    print("\n--- Training Model 1: Quant XGBoost (Class-Weighted) ---")
    xgb_model = QuantXGBoostModel()
    xgb_model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
    xgb_test_probs = xgb_model.predict_proba(X_test_scaled)
    
    # 6. Train PyTorch Deep LSTM Model
    print("\n--- Training Model 2: PyTorch 1D-CNN + Deep LSTM (Weighted Loss) ---")
    seq_len = config.SEQ_LEN
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_len)
    
    lstm_model = QuantDeepLSTMModel(input_dim=len(FEATURE_COLUMNS))
    lstm_model.fit(X_train_seq, y_train_seq)
    lstm_test_probs = lstm_model.predict_proba(X_test_seq)
    
    # Slice XGBoost probabilities to match sequence window alignment
    xgb_test_probs_seq = xgb_test_probs[seq_len:]
    y_test_seq_np = y_test_seq.numpy()
    df_test_seq = df_test.iloc[seq_len:].copy()
    
    # 7. Hybrid Ensemble Blending
    print("\n--- Evaluating Hybrid AI Ensemble (XGBoost + Deep LSTM) ---")
    ensemble = HybridEnsembleModel(xgb_weight=0.5, lstm_weight=0.5)
    ensemble_probs = ensemble.predict_proba(xgb_test_probs_seq, lstm_test_probs)
    ensemble_preds = ensemble.predict_with_thresholds(ensemble_probs, buy_threshold=0.38, sell_threshold=0.38)
    
    acc = accuracy_score(y_test_seq_np, ensemble_preds)
    macro_f1 = f1_score(y_test_seq_np, ensemble_preds, average='macro')
    print(f"Hybrid Ensemble Test Accuracy: {acc:.4f} | Macro F1-Score: {macro_f1:.4f}")
    print("\nDetailed Out-of-Sample Classification Report:")
    print(classification_report(y_test_seq_np, ensemble_preds, target_names=['SELL (-1)', 'HOLD (0)', 'BUY (+1)']))
    
    # 8. Out-of-Sample Backtest
    print("\n--- Running Financial Backtest on Held-Out Test Period (2025–2026) ---")
    df_bt, metrics = run_quant_backtest(df_test_seq, ensemble_preds, inv_label_map)
    
    print(f"Total Strategy Return:  {metrics['total_strategy_return']*100:.2f}%")
    print(f"Benchmark Buy-and-Hold: {metrics['total_benchmark_return']*100:.2f}%")
    print(f"Strategy CAGR:          {metrics['cagr_strategy']*100:.2f}%")
    print(f"Benchmark CAGR:         {metrics['cagr_benchmark']*100:.2f}%")
    print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:           {metrics['max_drawdown']*100:.2f}%")
    print(f"Strategy Win Rate:      {metrics['win_rate']*100:.2f}%")
    
    # 9. Live Inference & Section 30 Report
    print("\n================================================================================")
    print("               LIVE PREDICTION & EXPERT SECTION 30 REPORT")
    print("================================================================================")
    
    latest_X = X_test_scaled[-1:]
    latest_X_seq = X_test_seq[-1:]
    
    latest_xgb_p = xgb_model.predict_proba(latest_X)[0]
    latest_lstm_p = lstm_model.predict_proba(latest_X_seq)[0]
    latest_ens_p = 0.5 * latest_xgb_p + 0.5 * latest_lstm_p
    
    latest_pred_class = ensemble.predict_with_thresholds(np.array([latest_ens_p]), 0.38, 0.38)[0]
    decision_map = {0: "SELL / REDUCE", 1: "HOLD / WAIT", 2: "BUY / ACCUMULATE"}
    final_decision_str = decision_map[latest_pred_class]
    
    report_text = generate_section30_report(df_clean, latest_ens_p, final_decision_str)
    print(report_text)
    
    # Save Report to file
    report_file = os.path.join(config.REPORTS_DIR, "live_analysis_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nLive Analysis Report successfully saved to: {report_file}")

if __name__ == "__main__":
    main()
