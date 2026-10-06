import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

import config
from multi_agent_pipeline import MultiAgentPipeline
from predictor import evaluate_all_strategies, generate_section30_report

# Streamlit Page Config
st.set_page_config(
    page_title="Multi-Agent AI Quant Trading & Sentiment Platform",
    page_icon="🤖",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 30px; font-weight: 800; color: #1e293b; margin-bottom: 2px; }
    .subtitle { color: #64748b; font-size: 15px; margin-bottom: 20px; }
    .agent-card { background: #f8fafc; border-radius: 12px; padding: 18px; border: 1px solid #e2e8f0; margin-bottom: 15px; }
    .badge-buy { background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 12px 18px; border-radius: 10px; font-size: 20px; font-weight: bold; text-align: center; }
    .badge-sell { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; padding: 12px 18px; border-radius: 10px; font-size: 20px; font-weight: bold; text-align: center; }
    .badge-hold { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 12px 18px; border-radius: 10px; font-size: 20px; font-weight: bold; text-align: center; }
    .news-card { background: white; border-radius: 8px; padding: 12px; border-left: 4px solid #3b82f6; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .news-bullish { border-left-color: #10b981 !important; }
    .news-bearish { border-left-color: #ef4444 !important; }
    .news-neutral { border-left-color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar Asset & Engine Selection
st.sidebar.header("🎯 Stock & Market Selection")

asset_catalog = config.SUPPORTED_ASSETS
catalog_options = [f"{sym} — {meta['name']} ({meta.get('sector', '')})" for sym, meta in asset_catalog.items()]
catalog_options.insert(0, "✨ Custom Ticker Input...")

selected_option = st.sidebar.selectbox("Select Asset from Universe", catalog_options, index=1)

if selected_option == "✨ Custom Ticker Input...":
    custom_sym = st.sidebar.text_input("Enter Yahoo Finance Ticker Symbol", value="TMCV.NS", help="Examples: TMCV.NS, RELIANCE.NS, AAPL, NVDA, TSLA, MSFT")
    custom_name = st.sidebar.text_input("Company / Asset Name", value="Custom Asset")
    selected_ticker = custom_sym.strip()
    selected_name = custom_name.strip()
    curr_symbol = "₹" if ".NS" in selected_ticker or ".BO" in selected_ticker else "$"
else:
    selected_ticker = selected_option.split(" — ")[0]
    meta = asset_catalog[selected_ticker]
    selected_name = meta["name"]
    curr_symbol = meta.get("currency", "₹")

st.sidebar.markdown("---")
st.sidebar.header("⚡ Engine Configuration")
turbo_mode = st.sidebar.checkbox("🚀 Turbo Mode (High Speed Execution)", value=True, help="Accelerates model convergence, optimizes tree estimators, and utilizes fast-path caching.")
days_lookback = st.sidebar.slider("Display Window (Trading Days)", min_value=30, max_value=365, value=120)

# Main Title Banner
st.markdown(f"<div class='main-title'>🤖 Multi-Agent Quant AI Portfolio — {selected_name}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>Ticker: <code>{selected_ticker}</code> | Architecture: Technical Agent + Sentiment Agent + Arbitrator Agent</div>", unsafe_allow_html=True)

@st.cache_resource(ttl=900)
def execute_pipeline(ticker, name, turbo):
    pipeline = MultiAgentPipeline(ticker=ticker, asset_name=name, turbo_mode=turbo)
    return pipeline.run_pipeline()

with st.spinner(f"Running Multi-Agent AI Pipeline for {selected_name} ({selected_ticker})..."):
    try:
        pipeline_data = execute_pipeline(selected_ticker, selected_name, turbo_mode)
    except Exception as e:
        st.error(f"Failed to execute pipeline for {selected_ticker}: {e}")
        st.stop()

tech_agent = pipeline_data['technical_agent']
sent_agent = pipeline_data['sentiment_agent']
arb_agent = pipeline_data['arbitrator_agent']
df_clean = pipeline_data['df_clean']

# Top Decision Banner
col_dec, col_p, col_rsi, col_sent = st.columns([2, 1.3, 1.3, 1.4])

sig_code = arb_agent['signal_code']
badge_cls = "badge-buy" if sig_code == 2 else ("badge-sell" if sig_code == 0 else "badge-hold")

with col_dec:
    st.markdown("**ARBITRATED PORTFOLIO DECISION**")
    st.markdown(f"<div class='{badge_cls}'>{arb_agent['final_signal']}</div>", unsafe_allow_html=True)
    st.caption(f"Surety: {arb_agent['surety_level']}")

with col_p:
    latest_close = tech_agent['close_price']
    prev_close = float(df_clean.iloc[-2]['Close']) if len(df_clean) >= 2 else latest_close
    delta_p = latest_close - prev_close
    st.metric("Latest Close", f"{curr_symbol}{latest_close:.2f}", f"{delta_p:+.2f}")

with col_rsi:
    rsi_v = tech_agent['rsi_14']
    st.metric("RSI (14 Momentum)", f"{rsi_v:.2f}", f"{'Oversold' if rsi_v < 30 else ('Overbought' if rsi_v > 70 else 'Neutral Zone')}")

with col_sent:
    s_score = sent_agent['sentiment_score']
    st.metric("News Sentiment Score", f"{s_score:+.2f}", f"{sent_agent['sentiment_label']}")

st.markdown("---")

# Main Multi-Section Tabs
tab_agents, tab_patterns, tab_news, tab_charts, tab_backtest, tab_report = st.tabs([
    "🤖 Agent Pipeline Architecture",
    "🕯️ Pattern Confluence & Signals",
    "📰 Live News & NLP Sentiment",
    "📈 Interactive Technical Charts",
    "📊 Institutional Backtest",
    "📄 Section 30 Diagnostic Report"
])

with tab_agents:
    st.subheader("Multi-Agent System Workflow & Consensus")
    st.markdown("""
    This quantitative trading system operates through three specialized AI agents working in synchronization:
    """)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class='agent-card'>
            <h4>⚙️ Agent 1: Technical & Pattern Agent</h4>
            <p><b>Model Verdict:</b> <code>{tech_agent['decision']}</code></p>
            <p><b>XGBoost Prob:</b> BUY {tech_agent['p_xgb'][2]*100:.1f}% | SELL {tech_agent['p_xgb'][0]*100:.1f}%</p>
            <p><b>Deep LSTM Prob:</b> BUY {tech_agent['p_lstm'][2]*100:.1f}% | SELL {tech_agent['p_lstm'][0]*100:.1f}%</p>
            <p><b>Patterns Detected:</b> {len(tech_agent['patterns_detected'])} active setups</p>
            <p><b>EMA Location Filter:</b> {'✅ Valid' if tech_agent['location_pass'] else '❌ Outside Band'}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class='agent-card'>
            <h4>📰 Agent 2: News & Sentiment Agent</h4>
            <p><b>Sentiment Stance:</b> <code>{sent_agent['sentiment_label']}</code></p>
            <p><b>Sentiment Score:</b> {sent_agent['sentiment_score']:+.3f} (-1 to +1)</p>
            <p><b>Articles Analyzed:</b> {sent_agent['news_count']} live sources</p>
            <p><b>Bullish / Bearish Ratio:</b> {sent_agent['bullish_articles']} Bullish / {sent_agent['bearish_articles']} Bearish</p>
            <p><b>News Momentum:</b> {'High Velocity' if sent_agent['news_count'] >= 5 else 'Moderate Flow'}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        ep = arb_agent['execution_plan']
        st.markdown(f"""
        <div class='agent-card'>
            <h4>⚖️ Agent 3: Arbitrator & Fusion Agent</h4>
            <p><b>Arbitrated Action:</b> <code>{arb_agent['final_signal']}</code></p>
            <p><b>Recommended Sizing:</b> {ep['recommended_allocation']}</p>
            <p><b>Target 1 (1.0x Vol):</b> {curr_symbol}{ep['profit_target_1']}</p>
            <p><b>Target 2 (1.5x Vol):</b> {curr_symbol}{ep['profit_target_2']}</p>
            <p><b>Stop Loss (-1.0x Vol):</b> {curr_symbol}{ep['stop_loss_level']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Arbitrator Deliberation Notes & Risk Analysis")
    for note in arb_agent['arbitrator_notes']:
        st.info(note)

with tab_patterns:
    st.subheader(f"Empirical Candlestick & Chart Pattern Scanner — {selected_name}")
    
    df_strategies = evaluate_all_strategies(df_clean, tech_agent['p_technical'], asset_name=selected_name)
    st.dataframe(
        df_strategies,
        column_config={
            "Strategy Name": st.column_config.TextColumn("Strategy Name", width="medium"),
            "True Win Rate / Accuracy": st.column_config.TextColumn("Empirical Win Rate", width="small"),
            "Risk / Reward Ratio": st.column_config.TextColumn("Risk / Reward", width="small"),
            "Current Signal": st.column_config.TextColumn("Live Pattern Signal", width="medium"),
            "Action Plan": st.column_config.TextColumn("Execution Protocol", width="large")
        },
        use_container_width=True,
        hide_index=True
    )
    
    if tech_agent['patterns_detected']:
        st.markdown("### 🔥 Active Patterns Detected Today")
        for p in tech_agent['patterns_detected']:
            st.success(f"**{p['name']}**: {p['signal']} Signal (Win Rate: {p['win_rate']}) — {p['action']}")
    else:
        st.warning("No standalone single-candle patterns active on today's closing bar. ML ensemble and multi-timeframe structure are utilized.")

with tab_news:
    st.subheader(f"Real-Time Financial News & NLP Sentiment Feed — {selected_name}")
    st.markdown(f"**Aggregate NLP Sentiment Score:** `{sent_agent['sentiment_score']:+.3f}` | **Status:** `{sent_agent['sentiment_label']}`")
    
    for art in sent_agent['articles']:
        lbl = art['label']
        cls_name = "news-bullish" if lbl == "BULLISH" else ("news-bearish" if lbl == "BEARISH" else "news-neutral")
        badge_emoji = "🟢" if lbl == "BULLISH" else ("🔴" if lbl == "BEARISH" else "⚪")
        
        keywords_str = f" | Keywords: {', '.join(art['matched_keywords'])}" if art.get('matched_keywords') else ""
        
        st.markdown(f"""
        <div class='news-card {cls_name}'>
            <b>{badge_emoji} [{lbl} | Score: {art['sentiment_score']:+.2f}]</b> <a href="{art['link']}" target="_blank" style="text-decoration:none; color:#0f172a; font-weight:600;">{art['title']}</a>
            <div style="font-size:12px; color:#64748b; margin-top:4px;">Source: {art['publisher']} • {art['timestamp']}{keywords_str}</div>
        </div>
        """, unsafe_allow_html=True)

with tab_charts:
    st.subheader(f"Interactive Technical & Volatility Chart — {selected_ticker}")
    df_plot = df_clean.tail(days_lookback)
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(f"{selected_name} Price, EMAs & Bollinger Bands", "RSI (14) Momentum", "MACD Indicator")
    )
    
    # Row 1: Price & Bands
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], name='Close Price', line=dict(color='#0284c7', width=2)), row=1, col=1)
    if 'bb_upper' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['bb_upper'], name='Upper BB (20,2)', line=dict(color='#cbd5e1', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['bb_lower'], name='Lower BB (20,2)', line=dict(color='#cbd5e1', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(241, 245, 249, 0.3)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ema_20'], name='EMA 20', line=dict(color='#eab308', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ema_50'], name='EMA 50', line=dict(color='#f97316', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ema_200'], name='EMA 200', line=dict(color='#10b981', width=1.5)), row=1, col=1)
    
    # Row 2: RSI
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['rsi_14'], name='RSI 14', line=dict(color='#8b5cf6', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Row 3: MACD
    if 'macd_line' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['macd_line'], name='MACD Line', line=dict(color='#3b82f6', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['macd_signal'], name='Signal Line', line=dict(color='#ef4444', width=1.5)), row=3, col=1)
        colors = ['#10b981' if val >= 0 else '#ef4444' for val in df_plot['macd_hist']]
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['macd_hist'], name='MACD Hist', marker_color=colors), row=3, col=1)
        
    fig.update_layout(
        height=700,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_white",
        hovermode="x unified"
    )
    fig.update_yaxes(title_text=f"Price ({curr_symbol})", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

with tab_backtest:
    st.subheader(f"Institutional Out-of-Sample Financial Backtest — {selected_name}")
    bt = tech_agent['backtest_metrics']
    
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Net Strategy Return", f"{bt['total_strategy_return']*100:.1f}%", f"Benchmark: {bt['total_benchmark_return']*100:.1f}%")
        st.metric("Strategy CAGR", f"{bt['cagr_strategy']*100:.1f}%", f"Benchmark: {bt['cagr_benchmark']*100:.1f}%")
    with b2:
        st.metric("Annualized Sharpe Ratio", f"{bt['sharpe_ratio']:.2f}", "Risk-Free: 6.0%")
        st.metric("Sortino Ratio", f"{bt['sortino_ratio']:.2f}", "Downside Deviations")
    with b3:
        st.metric("Profit Factor", f"{bt['profit_factor']:.2f}", "Gross Gains / Losses")
        st.metric("Max Drawdown", f"{bt['max_drawdown']*100:.1f}%", "Peak-to-Trough")
    with b4:
        st.metric("Executed Trades", f"{bt['total_trades']}", f"Avg Duration: {bt['avg_hold_days']:.1f} Days")
        st.metric("Frictions Deducted", f"{bt['cost_bps_applied']} bps", "Round-trip STT + Slippage")

with tab_report:
    st.subheader("Section 30 Quantitative AI Diagnosis Report")
    report_text = generate_section30_report(
        df_clean,
        arb_agent['arbitrated_probs'],
        arb_agent['final_signal'],
        ticker=selected_ticker,
        asset_name=selected_name,
        sentiment_info=sent_agent
    )
    st.text_area("Full Diagnostic Report Content", report_text, height=400)
    st.download_button(
        label="📥 Download Diagnostic Report (.txt)",
        data=report_text,
        file_name=f"{selected_ticker}_multi_agent_report.txt",
        mime="text/plain"
    )
