import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

import config
from data_loader import load_market_data
from feature_engineering import engineer_quant_rsi_features, FEATURE_COLUMNS
from labeler import create_triple_barrier_labels
from models.xgboost_model import QuantXGBoostModel
from models.deep_lstm import QuantDeepLSTMModel, create_sequences
from models.ensemble import HybridEnsembleModel
from backtester import run_quant_backtest
from predictor import generate_section30_report, evaluate_all_strategies
from sentiment_agent import SentimentAgent
from multi_agent_pipeline import MultiAgentPipeline

def run_single_stock_pipeline(ticker=config.TICKER, asset_name=None, turbo=config.TURBO_MODE):
    pipeline = MultiAgentPipeline(ticker=ticker, asset_name=asset_name, turbo_mode=turbo)
    results = pipeline.run_pipeline()
    
    # Generate and save report
    df_clean = results['df_clean']
    arb = results['arbitrator_agent']
    sent = results['sentiment_agent']
    tech = results['technical_agent']
    
    report_text = generate_section30_report(
        df_clean,
        arb['arbitrated_probs'],
        arb['final_signal'],
        ticker=ticker,
        asset_name=results['asset_name'],
        sentiment_info=sent
    )
    
    safe_t = ticker.replace("^", "").replace(".", "_")
    report_file = os.path.join(config.REPORTS_DIR, f"{safe_t}_multi_agent_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"\nSaved Comprehensive Multi-Agent Report to: {report_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Quant Trading & Sentiment AI System")
    parser.add_argument("--ticker", type=str, default=config.TICKER, help="Ticker symbol (e.g., TMCV.NS, RELIANCE.NS, AAPL, NVDA)")
    parser.add_argument("--turbo", action="store_true", default=True, help="Enable Turbo high-speed inference mode")
    parser.add_argument("--all-stocks", action="store_true", help="Screen across all configured stocks in catalog")
    args = parser.parse_args()
    
    if args.all_stocks:
        print(f"=== Running Multi-Agent AI Screening Across {len(config.SUPPORTED_ASSETS)} Assets ===")
        summary_rows = []
        for sym, meta in config.SUPPORTED_ASSETS.items():
            try:
                print(f"\n>>> Analyzing {meta['name']} ({sym})...")
                res = run_single_stock_pipeline(ticker=sym, asset_name=meta['name'], turbo=True)
                arb = res['arbitrator_agent']
                tech = res['technical_agent']
                sent = res['sentiment_agent']
                summary_rows.append({
                    "Ticker": sym,
                    "Name": meta['name'],
                    "Sector": meta.get('sector', ''),
                    "Close": tech['close_price'],
                    "Technical Decision": tech['decision'],
                    "Sentiment": sent['sentiment_label'],
                    "Arbitrator Signal": arb['final_signal'],
                    "Surety": arb['surety_level'],
                    "Target 1": arb['execution_plan']['profit_target_1'],
                    "Stop Loss": arb['execution_plan']['stop_loss_level']
                })
            except Exception as e:
                print(f"Error processing {sym}: {e}")
                
        df_summary = pd.DataFrame(summary_rows)
        print("\n================================================================================")
        print("               MULTI-ASSET AI SCREENING SUMMARY TABLE")
        print("================================================================================")
        print(df_summary.to_string(index=False))
        summary_path = os.path.join(config.REPORTS_DIR, "market_screening_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"\nSaved Screening Summary to: {summary_path}")
    else:
        name = config.SUPPORTED_ASSETS.get(args.ticker, {}).get("name", args.ticker)
        run_single_stock_pipeline(ticker=args.ticker, asset_name=name, turbo=args.turbo)

if __name__ == "__main__":
    main()
