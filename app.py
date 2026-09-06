import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import torch
import yfinance as yf

import config
from feature_engineering import engineer_quant_rsi_features, FEATURE_COLUMNS
from labeler import create_quant_target_labels
from models.xgboost_model import QuantXGBoostModel
from models.deep_lstm import QuantDeepLSTMModel, create_sequences
from models.ensemble import HybridEnsembleModel
from predictor import generate_section30_report, evaluate_all_strategies

# Page Config
st.set_page_config(
    page_title="Multi-Asset Quant AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 28px; font-weight: bold; margin-bottom: 5px; }
    .subtitle { color: #6c757d; font-size: 14px; margin-bottom: 20px; }
    .badge-buy { background-color: #28a745; color: white; padding: 12px 20px; border-radius: 8px; font-size: 22px; font-weight: bold; text-align: center; }
    .badge-sell { background-color: #dc3545; color: white; padding: 12px 20px; border-radius: 8px; font-size: 22px; font-weight: bold; text-align: center; }
    .badge-hold { background-color: #ffc107; color: black; padding: 12px 20px; border-radius: 8px; font-size: 22px; font-weight: bold; text-align: center; }
    .badge-wait { background-color: #6c757d; color: white; padding: 12px 20px; border-radius: 8px; font-size: 22px; font-weight: bold; text-align: center; }
    .card-box { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# Sidebar Asset Selection
st.sidebar.header("🎯 Stock & Asset Selection")
ASSET_OPTIONS = {
    "Tata Steel Ltd. (TATASTEEL.NS)": ("TATASTEEL.NS", "Tata Steel Ltd.", "₹"),
    "Reliance Industries Ltd. (RELIANCE.NS)": ("RELIANCE.NS", "Reliance Industries Ltd.", "₹"),
    "JPMorgan Chase & Co. (JPM)": ("JPM", "JPMorgan Chase & Co.", "$"),
    "IDFC First Bank Ltd. (IDFCFIRSTB.NS)": ("IDFCFIRSTB.NS", "IDFC First Bank Ltd.", "₹"),
    "Aditya Birla Real Estate Ltd. (ABREL.NS)": ("ABREL.NS", "Aditya Birla Real Estate Ltd.", "₹"),
    "Aditya Birla Capital Ltd. (ABCAPITAL.NS)": ("ABCAPITAL.NS", "Aditya Birla Capital Ltd.", "₹")
}

selected_asset_label = st.sidebar.selectbox("Choose Stock to Analyze", list(ASSET_OPTIONS.keys()), index=0)
selected_ticker, selected_name, curr_symbol = ASSET_OPTIONS[selected_asset_label]

days_lookback = st.sidebar.slider("Display Window (Days)", min_value=30, max_value=365, value=120)

st.markdown(f"<div class='main-title'>📈 Multi-Strategy Quant AI — {selected_name} ({selected_ticker})</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Empirical Candlestick Strategies, Multi-Pattern Confluence & AI Neural Ensemble</div>", unsafe_allow_html=True)

@st.cache_resource(ttl=1800)
def train_and_get_ai_models_for_ticker(ticker_symbol):
    """
    Trains the XGBoost and PyTorch Deep LSTM models dynamically on historical data for ticker_symbol.
    """
    df_raw = yf.download(ticker_symbol, start=config.START_DATE, progress=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw = df_raw.xs(ticker_symbol, level=1, axis=1)
    df_raw = df_raw.dropna().copy()
    
    df_feat = engineer_quant_rsi_features(df_raw)
    target = create_quant_target_labels(df_feat)
    df_feat['target'] = target
    
    label_map = {-1: 0, 0: 1, 1: 2}
    df_feat['target_mapped'] = df_feat['target'].map(label_map)
    df_clean = df_feat.dropna().copy()
    
    X = df_clean[FEATURE_COLUMNS].values
    y = df_clean['target_mapped'].values
    
    n = len(df_clean)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train XGBoost
    xgb_model = QuantXGBoostModel()
    xgb_model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Train PyTorch LSTM
    seq_len = config.SEQ_LEN
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_len)
    lstm_model = QuantDeepLSTMModel(input_dim=len(FEATURE_COLUMNS))
    lstm_model.fit(X_train_seq, y_train_seq, epochs=30)
    
    ensemble = HybridEnsembleModel(xgb_weight=0.5, lstm_weight=0.5)
    
    return df_clean, scaler, xgb_model, lstm_model, ensemble

with st.spinner(f"Fetching live data & training AI Ensemble for {selected_name} ({selected_ticker})..."):
    df_clean, scaler, xgb_model, lstm_model, ensemble = train_and_get_ai_models_for_ticker(selected_ticker)

# Compute Live Inference using trained AI ensemble
X_all_scaled = scaler.transform(df_clean[FEATURE_COLUMNS].values)
latest_X = X_all_scaled[-1:]

seq_len = config.SEQ_LEN
X_all_seq = []
for i in range(len(X_all_scaled) - seq_len, len(X_all_scaled)):
    X_all_seq.append(X_all_scaled[i-seq_len:i])
X_all_seq_tensor = torch.tensor(np.array(X_all_seq), dtype=torch.float32)
latest_X_seq = X_all_seq_tensor[-1:]

p_xgb = xgb_model.predict_proba(latest_X)[0]
p_lstm = lstm_model.predict_proba(latest_X_seq)[0]
p_ens = ensemble.predict_proba(np.array([p_xgb]), np.array([p_lstm]))[0]

pred_class = ensemble.predict_with_thresholds(np.array([p_ens]), 0.38, 0.38)[0]
decision_map = {0: ("SELL / REDUCE", "badge-sell"), 1: ("HOLD / WAIT", "badge-hold"), 2: ("BUY / ACCUMULATE", "badge-buy")}
decision_str, badge_class = decision_map[pred_class]

latest = df_clean.iloc[-1]
rsi_val = float(latest['rsi_14'])
close_val = float(latest['Close'])
rsi_diff1 = float(latest['rsi_diff1'])

# Evaluate All Strategies
df_strategies = evaluate_all_strategies(df_clean, p_ens)

# Top Banner Summary Metrics
col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1.5])
with col1:
    st.markdown(f"**AI MODEL DECISION**")
    st.markdown(f"<div class='{badge_class}'>{decision_str}</div>", unsafe_allow_html=True)
with col2:
    st.metric("Latest Close Price", f"{curr_symbol}{close_val:.2f}", f"{latest['Close'] - df_clean.iloc[-2]['Close']:.2f}")
with col3:
    st.metric("Current RSI (14)", f"{rsi_val:.2f}", f"{rsi_diff1:+.2f}")
with col4:
    st.metric("AI Probabilities", f"BUY: {p_ens[2]*100:.0f}%", f"SELL: {p_ens[0]*100:.0f}% | HOLD: {p_ens[1]*100:.0f}%")

st.markdown("---")

# Main Multi-Section Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Section 1: Multi-Strategy Predictions & Win Rates",
    "📈 Section 2: Interactive Chart & Technical Indicators",
    "🧠 Section 3: Quantitative AI Diagnosis & Backtest"
])

with tab1:
    st.subheader(f"Verified Candlestick & ML Strategy Breakdown — {selected_name}")
    st.markdown("""
    This section evaluates live market data across **empirical candlestick patterns** (backtested on multi-decade market data) 
    alongside the **Multi-Agent XGBoost + PyTorch Neural Ensemble**.
    """)
    
    # Render Strategy Comparison Table
    st.dataframe(
        df_strategies,
        column_config={
            "Strategy Name": st.column_config.TextColumn("Strategy Name", width="medium"),
            "True Win Rate / Accuracy": st.column_config.TextColumn("Empirical Win Rate", width="small"),
            "Risk / Reward Ratio": st.column_config.TextColumn("Risk / Reward", width="small"),
            "Current Signal": st.column_config.TextColumn("Live Prediction Signal", width="medium"),
            "Action Plan": st.column_config.TextColumn("Execution Verdict & Action Plan", width="large")
        },
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("Strategy Blueprint & Validation Rules")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class='card-box'>
        <h4>✅ High-Probability Execution Rules</h4>
        <ul>
            <li><b>Rule 1: Location Filter:</b> Trade setups only when formed within 3% of EMA 20, EMA 50, or EMA 200 key moving averages.</li>
            <li><b>Rule 2: Volume Expansion:</b> Reversal candles must be backed by volume &ge; 20-period Volume SMA.</li>
            <li><b>Rule 3: Multi-Pattern Confluence:</b> Never trade single candles in isolation; require agreement across ML & candlestick patterns.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        loc_status = "✅ PASS" if int(latest.get('location_filter_pass', 0)) == 1 else "❌ FAIL"
        vol_status = "✅ PASS" if int(latest.get('volume_filter_pass', 0)) == 1 else "❌ FAIL"
        
        st.markdown(f"""
        <div class='card-box'>
        <h4>🔍 Current Validation Status</h4>
        <p><b>Location Filter (Near EMA 20/50/200):</b> {loc_status}</p>
        <p><b>Volume Expansion Filter:</b> {vol_status}</p>
        <p><b>Confluence Signal:</b> {df_strategies.iloc[-1]['Current Signal']}</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader(f"Live Price & RSI Indicator Chart — {selected_ticker}")
    df_plot = df_clean.tail(days_lookback)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(f"{selected_name} Price & Moving Averages", f"RSI (14) Momentum Indicator")
    )
    
    # Row 1: Price Chart & Moving Averages
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot['Close'], name='Close Price', line=dict(color='#007bff', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot['ema_20'], name='EMA 20', line=dict(color='#ffc107', width=1.5, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot['ema_50'], name='EMA 50', line=dict(color='#dc3545', width=1.5, dash='dot')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot['ema_200'], name='EMA 200', line=dict(color='#28a745', width=1.5)),
        row=1, col=1
    )
    
    # Row 2: RSI Chart
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot['rsi_14'], name='RSI 14', line=dict(color='#6f42c1', width=2)),
        row=2, col=1
    )
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Midpoint (50)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", row=2, col=1)
    
    fig.update_layout(
        height=550,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
        hovermode="x unified"
    )
    fig.update_yaxes(title_text=f"Price ({curr_symbol})", row=1, col=1)
    fig.update_yaxes(title_text="RSI Value", range=[0, 100], row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader(f"Quantitative AI Model Diagnosis — {selected_name}")
    
    report_text = generate_section30_report(df_clean, p_ens, decision_str)
    st.text_area("Full Section 30 Output Report", report_text, height=350)
    
    st.subheader("AI Model Architecture & Weights")
    st.write(f"• **XGBoost Classifier Weight:** 50.0% (Probabilities — BUY: {p_xgb[2]*100:.1f}%, HOLD: {p_xgb[1]*100:.1f}%, SELL: {p_xgb[0]*100:.1f}%)")
    st.write(f"• **PyTorch Deep LSTM Weight:** 50.0% (Probabilities — BUY: {p_lstm[2]*100:.1f}%, HOLD: {p_lstm[1]*100:.1f}%, SELL: {p_lstm[0]*100:.1f}%)")
