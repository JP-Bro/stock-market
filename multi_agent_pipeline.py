import numpy as np
import pandas as pd
from datetime import datetime
import config
from data_loader import load_market_data
from feature_engineering import engineer_quant_rsi_features, FEATURE_COLUMNS
from labeler import create_triple_barrier_labels
from models.xgboost_model import QuantXGBoostModel
from models.deep_lstm import QuantDeepLSTMModel, create_sequences
from models.ensemble import HybridEnsembleModel
from backtester import run_quant_backtest
from candlestick_strategies import STRATEGY_REGISTRY
from sentiment_agent import SentimentAgent

class MultiAgentPipeline:
    """
    Hierarchical Multi-Agent Quant Trading Pipeline:
    |-- Agent 1: Technical & Pattern Recognition Agent (OHLCV, RSI, MACD, Candlestick Strategies, XGBoost + LSTM)
    |-- Agent 2: Real-time News & Sentiment Agent (NLP Polarity, News Momentum, Headline Catalysts)
    `-- Agent 3: Arbitrator & Fusion Agent (Mediator, Conflict Resolution, Risk Management & Execution Sizing)
    """
    def __init__(self, ticker=config.TICKER, asset_name=None, turbo_mode=config.TURBO_MODE):
        self.ticker = ticker
        self.asset_name = asset_name or config.SUPPORTED_ASSETS.get(ticker, {}).get("name", ticker)
        self.turbo_mode = turbo_mode
        self.sentiment_agent = SentimentAgent(self.ticker, self.asset_name)
        
    def run_pipeline(self):
        """
        Executes the entire multi-agent pipeline from raw data acquisition to final arbitrated trade order.
        """
        print(f"\n================================================================================")
        print(f"[*] MULTI-AGENT PIPELINE: {self.asset_name} ({self.ticker}) [Turbo Mode: {self.turbo_mode}]")
        print(f"================================================================================")
        
        # ---------------------------------------------------------------------
        # 1. AGENT 1: TECHNICAL & PATTERN RECOGNITION AGENT
        # ---------------------------------------------------------------------
        print(f"[Agent 1: Technical] Loading market data and engineering features...")
        df_raw = load_market_data(self.ticker, start_date=config.START_DATE)
        df_feat = engineer_quant_rsi_features(df_raw)
        
        # Triple Barrier target generation
        target = create_triple_barrier_labels(
            df_feat,
            holding_period=config.BARRIER_TIME_HORIZON,
            pt_mult=config.PROFIT_TAKE_VOL_MULT,
            sl_mult=config.STOP_LOSS_VOL_MULT
        )
        df_feat['target'] = target
        df_feat['target_mapped'] = df_feat['target'].map({-1: 0, 0: 1, 1: 2})
        df_clean = df_feat.dropna().copy()
        
        X = df_clean[FEATURE_COLUMNS].values
        y = df_clean['target_mapped'].values
        
        # Strict Chronological Split with Purging & Embargoing (Marcos López de Prado)
        n = len(df_clean)
        h = config.BARRIER_TIME_HORIZON
        raw_train_end = int(n * config.TRAIN_RATIO)
        train_end = max(0, raw_train_end - h)  # Purge last h bars to prevent leakage into validation
        val_start = raw_train_end + 5         # 5-bar embargo
        raw_val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
        val_end = max(val_start + 1, raw_val_end - h) # Purge last h bars to prevent leakage into test
        test_start = raw_val_end + 5          # 5-bar embargo
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]
        X_test, y_test = X[test_start:], y[test_start:]
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train / Ingest ML Models
        tune_xgb = not self.turbo_mode
        xgb_model = QuantXGBoostModel(tune_params=tune_xgb)
        xgb_model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
        
        seq_len = config.SEQ_LEN
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_len)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_len)
        
        lstm_epochs = config.LSTM_EPOCHS if not self.turbo_mode else 15
        lstm_model = QuantDeepLSTMModel(input_dim=len(FEATURE_COLUMNS))
        lstm_model.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=lstm_epochs)
        
        # Backtest
        ensemble = HybridEnsembleModel(xgb_weight=0.5, lstm_weight=0.5)
        xgb_test_probs = xgb_model.predict_proba(X_test_scaled)[seq_len:]
        lstm_test_probs = lstm_model.predict_proba(X_test_seq)
        ens_test_probs = ensemble.predict_proba(xgb_test_probs, lstm_test_probs)
        ens_test_preds = ensemble.predict_with_thresholds(
            ens_test_probs,
            buy_threshold=config.CONFIDENCE_THRESHOLD,
            sell_threshold=config.CONFIDENCE_THRESHOLD
        )
        
        df_test_seq = df_clean.iloc[test_start + seq_len:].copy()
        df_bt, bt_metrics = run_quant_backtest(df_test_seq, ens_test_preds, {0: -1, 1: 0, 2: 1})
        
        # Live Technical Inference
        X_all_scaled = scaler.transform(df_clean[FEATURE_COLUMNS].values)
        latest_X = X_all_scaled[-1:]
        latest_X_seq = create_sequences(X_all_scaled[-seq_len:])
            
        p_xgb = xgb_model.predict_proba(latest_X)[0]
        p_lstm = lstm_model.predict_proba(latest_X_seq)[0]
        p_tech = ensemble.predict_proba(np.array([p_xgb]), np.array([p_lstm]))[0] # [SELL, HOLD, BUY]
        
        latest_bar = df_clean.iloc[-1]
        patterns_detected = []
        for st in STRATEGY_REGISTRY:
            val = int(latest_bar.get(st['col'], 0))
            if val != 0:
                patterns_detected.append({
                    "name": st['name'],
                    "signal": "BUY" if val == 1 else "SELL",
                    "win_rate": st['win_rate'],
                    "action": st['action']
                })
                
        tech_decision = "BUY" if p_tech[2] >= config.CONFIDENCE_THRESHOLD and p_tech[2] > p_tech[0] else ("SELL" if p_tech[0] >= config.CONFIDENCE_THRESHOLD and p_tech[0] > p_tech[2] else "HOLD")
        
        technical_result = {
            "p_xgb": p_xgb.tolist(),
            "p_lstm": p_lstm.tolist(),
            "p_technical": p_tech.tolist(), # [p_sell, p_hold, p_buy]
            "decision": tech_decision,
            "patterns_detected": patterns_detected,
            "rsi_14": float(latest_bar['rsi_14']),
            "close_price": float(latest_bar['Close']),
            "volatility_daily": float(latest_bar.get('norm_atr', 0.02)),
            "location_pass": bool(latest_bar.get('location_filter_pass', 0)),
            "volume_pass": bool(latest_bar.get('volume_filter_pass', 0)),
            "backtest_metrics": bt_metrics
        }
        print(f"[Agent 1: Technical] Output -> Decision: {tech_decision} (BUY: {p_tech[2]*100:.1f}%, HOLD: {p_tech[1]*100:.1f}%, SELL: {p_tech[0]*100:.1f}%) | Patterns Found: {len(patterns_detected)}")
        
        # ---------------------------------------------------------------------
        # 2. AGENT 2: NEWS & SENTIMENT AGENT
        # ---------------------------------------------------------------------
        print(f"[Agent 2: Sentiment] Fetching real-time news and calculating NLP sentiment...")
        sentiment_result = self.sentiment_agent.analyze_sentiment()
        p_sent = sentiment_result['sentiment_probs'] # [p_sell, p_hold, p_buy]
        print(f"[Agent 2: Sentiment] Output -> Label: {sentiment_result['sentiment_label']} (Score: {sentiment_result['sentiment_score']}) | Articles Scored: {sentiment_result['news_count']}")
        
        # ---------------------------------------------------------------------
        # 3. AGENT 3: ARBITRATOR & RISK FUSION AGENT (THE MEDIATOR)
        # ---------------------------------------------------------------------
        print(f"[Agent 3: Arbitrator] Reconciling Technicals + Sentiment into Final Portfolio Order...")
        w_tech = config.WEIGHT_TECHNICAL_AGENT
        w_sent = config.WEIGHT_SENTIMENT_AGENT
        
        # Blended Probability: [SELL, HOLD, BUY]
        p_final = (w_tech * np.array(p_tech)) + (w_sent * np.array(p_sent))
        p_sell_f, p_hold_f, p_buy_f = float(p_final[0]), float(p_final[1]), float(p_final[2])
        
        # Conflict Detection and Arbitrator Logic
        has_tech_bull = (p_tech[2] >= config.CONFIDENCE_THRESHOLD)
        has_tech_bear = (p_tech[0] >= config.CONFIDENCE_THRESHOLD)
        has_sent_bull = (sentiment_result['sentiment_score'] >= 0.10)
        has_sent_bear = (sentiment_result['sentiment_score'] <= -0.10)
        
        confluence_state = "NEUTRAL"
        arbitrator_notes = []
        
        if has_tech_bull and has_sent_bull:
            final_signal = "STRONG BUY / ACCUMULATE"
            signal_code = 2
            confluence_state = "HIGH CONFLUENCE BULLISH"
            surety_level = "VERY HIGH (Technicals + Sentiment in Synergistic Alignment)"
            position_size_pct = "8.0% - 10.0% Allocation"
            arbitrator_notes.append("[+] Both Quantitative Technical Models and Financial News Sentiment confirm strong upside momentum.")
        elif has_tech_bear and has_sent_bear:
            final_signal = "STRONG SELL / SHORT / HEDGE"
            signal_code = 0
            confluence_state = "HIGH CONFLUENCE BEARISH"
            surety_level = "VERY HIGH (Technicals + Sentiment in Downside Alignment)"
            position_size_pct = "Exit to 100% Cash / Hedge"
            arbitrator_notes.append("[!] Downside structural break confirmed by negative headline sentiment. Strict capital preservation advised.")
        elif has_tech_bull and not has_sent_bear:
            final_signal = "MODERATE BUY (Technicals Supported)"
            signal_code = 2
            confluence_state = "MODERATE BULLISH"
            surety_level = "MODERATE (Technicals Positive, News Neutral)"
            position_size_pct = "4.0% - 6.0% Allocation"
            arbitrator_notes.append("[+] Technical patterns and neural models indicate positive expectancy; news sentiment is neutral.")
        elif has_tech_bear and not has_sent_bull:
            final_signal = "MODERATE SELL / REDUCE"
            signal_code = 0
            confluence_state = "MODERATE BEARISH"
            surety_level = "MODERATE (Technicals Negative, News Neutral)"
            position_size_pct = "Trim 50% Position / Tighten Stops"
            arbitrator_notes.append("[-] Technical resistance rejection detected; news flow neutral.")
        elif has_tech_bull and has_sent_bear:
            final_signal = "STAND ASIDE / CONFLICT DETECTED"
            signal_code = 1
            confluence_state = "DIVERGENT CONFLICT"
            surety_level = "LOW (Technical Buy vs Negative News Sentiment)"
            position_size_pct = "0% (Hold Cash)"
            arbitrator_notes.append("[!] Arbitrator Intervention: Technical models show buy signals, but adverse news sentiment creates tail risk. Standing aside to protect capital.")
        elif has_tech_bear and has_sent_bull:
            final_signal = "STAND ASIDE / CONFLICT DETECTED"
            signal_code = 1
            confluence_state = "DIVERGENT CONFLICT"
            surety_level = "LOW (Technical Sell vs Positive News Sentiment)"
            position_size_pct = "0% (Hold Cash)"
            arbitrator_notes.append("[!] Arbitrator Intervention: Positive news hype conflicting with technical breakdown. Standing aside until clear resolution.")
        else:
            final_signal = "HOLD CASH / NO HIGH-SURETY SETUP"
            signal_code = 1
            confluence_state = "BALANCED / INSUFFICIENT CONVICTION"
            surety_level = "NEUTRAL (Within Standard Statistical Noise Band)"
            position_size_pct = "0% (Maintain Cash / Existing Stops)"
            arbitrator_notes.append("[*] Market is in consolidation or neutral equilibrium. Protect capital until a high-probability asymmetric opportunity emerges.")
            
        # Triple-Barrier Price Levels
        curr_p = technical_result['close_price']
        vol = technical_result['volatility_daily']
        target_1 = curr_p * (1.0 + (1.0 * vol))
        target_2 = curr_p * (1.0 + (config.PROFIT_TAKE_VOL_MULT * vol))
        stop_loss = curr_p * (1.0 - (config.STOP_LOSS_VOL_MULT * vol))
        
        execution_plan = {
            "current_price": round(curr_p, 2),
            "stop_loss_level": round(stop_loss, 2),
            "profit_target_1": round(target_1, 2),
            "profit_target_2": round(target_2, 2),
            "risk_reward_ratio": f"1:{config.PROFIT_TAKE_VOL_MULT/config.STOP_LOSS_VOL_MULT:.1f}",
            "holding_horizon": f"{config.BARRIER_TIME_HORIZON} Trading Days (Swing)",
            "recommended_allocation": position_size_pct
        }
        
        arbitrator_result = {
            "final_signal": final_signal,
            "signal_code": signal_code,
            "confluence_state": confluence_state,
            "surety_level": surety_level,
            "arbitrated_probs": [round(p_sell_f, 3), round(p_hold_f, 3), round(p_buy_f, 3)],
            "weights_used": {"technical_agent": w_tech, "sentiment_agent": w_sent},
            "arbitrator_notes": arbitrator_notes,
            "execution_plan": execution_plan
        }
        
        print(f"[Agent 3: Arbitrator] Final Verdict: {final_signal} | Surety: {surety_level}")
        print(f"[Agent 3: Arbitrator] Execution Plan: SL @ {execution_plan['stop_loss_level']}, TP1 @ {execution_plan['profit_target_1']}, TP2 @ {execution_plan['profit_target_2']}")
        print(f"================================================================================\n")
        
        return {
            "ticker": self.ticker,
            "asset_name": self.asset_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "technical_agent": technical_result,
            "sentiment_agent": sentiment_result,
            "arbitrator_agent": arbitrator_result,
            "df_clean": df_clean
        }

if __name__ == "__main__":
    pipeline = MultiAgentPipeline(ticker="TMCV.NS", turbo_mode=True)
    results = pipeline.run_pipeline()
