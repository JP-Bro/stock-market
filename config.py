import os
import torch
import numpy as np
import random
from datetime import datetime

# Reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Turbo Mode Setting (Fast inference, optimized epochs, instant caching)
TURBO_MODE = True

# Asset & Target Config (Target: Tata Motors TMCV from Jan 2021 to Sept 2026)
TICKER = "TMCV.NS"
ASSET_NAME = "Tata Motors Limited (TMCV)"
START_DATE = "2021-01-01"
END_DATE = "2026-09-26"
BACKTEST_START_DATE = "2025-06-01"  # Explicit held-out test split

# Supported Multi-Asset Catalog (Diverse Sectors & Markets)
SUPPORTED_ASSETS = {
    # Indian Large Caps
    "TMCV.NS": {"name": "Tata Motors Limited (TMCV)", "currency": "₹", "sector": "Automotive"},
    "RELIANCE.NS": {"name": "Reliance Industries Ltd.", "currency": "₹", "sector": "Energy & Conglomerate"},
    "TATASTEEL.NS": {"name": "Tata Steel Ltd.", "currency": "₹", "sector": "Metals & Mining"},
    "HDFCBANK.NS": {"name": "HDFC Bank Ltd.", "currency": "₹", "sector": "Banking & Financials"},
    "ICICIBANK.NS": {"name": "ICICI Bank Ltd.", "currency": "₹", "sector": "Banking & Financials"},
    "SBIN.NS": {"name": "State Bank of India", "currency": "₹", "sector": "PSU Banking"},
    "INFY.NS": {"name": "Infosys Ltd.", "currency": "₹", "sector": "Information Technology"},
    "TCS.NS": {"name": "Tata Consultancy Services", "currency": "₹", "sector": "Information Technology"},
    "SUNPHARMA.NS": {"name": "Sun Pharmaceutical Ltd.", "currency": "₹", "sector": "Healthcare & Pharma"},
    "ITC.NS": {"name": "ITC Limited", "currency": "₹", "sector": "FMCG & Consumer"},
    "LT.NS": {"name": "Larsen & Toubro Ltd.", "currency": "₹", "sector": "Infrastructure & Capital Goods"},
    "ABREL.NS": {"name": "Aditya Birla Real Estate Ltd.", "currency": "₹", "sector": "Real Estate"},
    "ABCAPITAL.NS": {"name": "Aditya Birla Capital Ltd.", "currency": "₹", "sector": "Financial Services"},
    "IDFCFIRSTB.NS": {"name": "IDFC First Bank Ltd.", "currency": "₹", "sector": "Banking"},
    # US & Global Equities
    "AAPL": {"name": "Apple Inc.", "currency": "$", "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corporation", "currency": "$", "sector": "Technology"},
    "NVDA": {"name": "NVIDIA Corporation", "currency": "$", "sector": "Semiconductors"},
    "GOOGL": {"name": "Alphabet Inc. (Google)", "currency": "$", "sector": "Communication Services"},
    "AMZN": {"name": "Amazon.com Inc.", "currency": "$", "sector": "Consumer Discretionary"},
    "TSLA": {"name": "Tesla Inc.", "currency": "$", "sector": "Electric Vehicles & Energy"},
    "JPM": {"name": "JPMorgan Chase & Co.", "currency": "$", "sector": "Banking & Investment"}
}

# Marcos López de Prado Triple Barrier Method Configuration
BARRIER_TIME_HORIZON = 10     # Max holding period in trading days (2 weeks)
PROFIT_TAKE_VOL_MULT = 1.5    # Upper Barrier: +1.5x daily volatility band
STOP_LOSS_VOL_MULT = 1.0      # Lower Barrier: -1.0x daily volatility band (1.5:1 Risk-Reward Ratio)
MIN_VOLATILITY_FLOOR = 0.005  # Minimum daily volatility floor (0.5%)

# Time Series Split (Fallback if date split not used)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Sequence Model Window (Daily Lookback)
SEQ_LEN = 15

# XGBoost Hyperparameters (Optimized for Turbo / Full execution)
XGB_N_ESTIMATORS = 180 if TURBO_MODE else 250
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.03

# PyTorch Deep LSTM Hyperparameters
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_EPOCHS = 25 if TURBO_MODE else 50
LSTM_LR = 0.001
LSTM_BATCH_SIZE = 32
LSTM_EARLY_STOPPING_PATIENCE = 5 if TURBO_MODE else 7

# Multi-Agent Pipeline Weights
WEIGHT_TECHNICAL_AGENT = 0.60
WEIGHT_SENTIMENT_AGENT = 0.40

# Decision & Confidence Filtering (Only trade when statistical surety is high)
CONFIDENCE_THRESHOLD = 0.46   # Calibrated probability floor
TRANSACTION_COST_BPS = 15.0   # 0.15% (15 bps) round-trip commission + STT + slippage

# Dynamic Project Directories (No hardcoded user paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)
