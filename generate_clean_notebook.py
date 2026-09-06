import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Markdown
cell1 = nbf.v4.new_markdown_cell("""# 📈 RSI Quant AI Trading Model — Experiment & Tuning Notebook

Use this notebook to test different parameter values (forward window, ATR threshold, XGBoost parameters, LSTM sequence length) and evaluate 3-class accuracy, F1-scores, and backtest returns yourself.""")

# Cell 2: Markdown
cell2 = nbf.v4.new_markdown_cell("""## 1. Parameter Settings (Tweak These Values)""")

# Cell 3: Code
cell3_code = """# ==========================================================================
# ⚙️ EXPERIMENTAL HYPERPARAMETERS — CHANGE THESE AND RERUN THE NOTEBOOK!
# ==========================================================================

TICKER = 'ABCAPITAL.NS'
START_DATE = '2017-09-01'

# Labeling Parameters
FORWARD_WINDOW = 5        # Forward return horizon (in trading days)
PROFIT_MULTIPLIER = 0.8   # ATR volatility threshold (e.g. 0.5, 0.8, 1.0, 1.2)

# Model Split Ratios
TRAIN_RATIO = 0.70        # 70% Train
VAL_RATIO = 0.15          # 15% Validation
# 15% Test (Held-Out)

# XGBoost Hyperparameters
XGB_N_ESTIMATORS = 150
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.03
XGB_MIN_CHILD_WEIGHT = 1

# Deep LSTM Hyperparameters
SEQ_LEN = 15              # Temporal sequence window (10, 15, 20 days)
LSTM_EPOCHS = 30          # Training epochs (20, 30, 50)
LSTM_HIDDEN_DIM = 64
LSTM_LR = 0.001

RANDOM_SEED = 42
print('✅ Hyperparameters loaded. Change values above to experiment!')"""
cell3 = nbf.v4.new_code_cell(cell3_code)

# Cell 4: Markdown
cell4 = nbf.v4.new_markdown_cell("""## 2. Ingest Live Data & Feature Engineering""")

# Cell 5: Code
cell5_code = """import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f'Fetching live market data for {TICKER}...')
df_raw = yf.download(TICKER, start=START_DATE, progress=False)
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw = df_raw.xs(TICKER, level=1, axis=1)
df_raw = df_raw.dropna().copy()
print(f'Loaded {len(df_raw)} rows up to {df_raw.index[-1].strftime("%Y-%m-%d")}.')

# Compute Wilder RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).copy()
    loss = (-delta.where(delta < 0, 0)).copy()
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

# Feature Engineering
df = df_raw.copy()
close = df['Close']
high, low = df['High'], df['Low']

df['rsi_14'] = compute_rsi(close, 14)
df['rsi_7'] = compute_rsi(close, 7)
df['rsi_21'] = compute_rsi(close, 21)
df['rsi_diff1'] = df['rsi_14'].diff(1)
df['rsi_diff3'] = df['rsi_14'].diff(3)
df['rsi_diff5'] = df['rsi_14'].diff(5)
df['rsi_accel'] = df['rsi_diff1'].diff(1)

mean_30 = df['rsi_14'].rolling(30).mean()
std_30 = df['rsi_14'].rolling(30).std() + 1e-8
df['rsi_zscore_30'] = (df['rsi_14'] - mean_30) / std_30

df['rsi_zone'] = pd.cut(df['rsi_14'], bins=[-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf], labels=[0,1,2,3,4,5,6,7]).astype(float)
df['cross_30_up'] = ((df['rsi_14'].shift(1) < 30) & (df['rsi_14'] >= 30)).astype(int)
df['cross_50_up'] = ((df['rsi_14'].shift(1) < 50) & (df['rsi_14'] >= 50)).astype(int)
df['cross_70_down'] = ((df['rsi_14'].shift(1) > 70) & (df['rsi_14'] <= 70)).astype(int)
df['cross_50_down'] = ((df['rsi_14'].shift(1) > 50) & (df['rsi_14'] <= 50)).astype(int)

def calc_slope(s):
    if len(s) < 5: return 0.0
    return np.polyfit(np.arange(len(s)), s, 1)[0]

p_slope = close.rolling(10).apply(calc_slope, raw=True)
r_slope = df['rsi_14'].rolling(10).apply(calc_slope, raw=True)
df['bullish_div'] = ((p_slope < 0) & (r_slope > 0)).astype(int)
df['bearish_div'] = ((p_slope > 0) & (r_slope < 0)).astype(int)

df['ema_20'] = close.ewm(span=20, adjust=False).mean()
df['ema_50'] = close.ewm(span=50, adjust=False).mean()
df['ema_200'] = close.ewm(span=200, adjust=False).mean()
df['dist_ema_20'] = (close - df['ema_20']) / df['ema_20']
df['dist_ema_50'] = (close - df['ema_50']) / df['ema_50']
df['trend_50_200'] = (df['ema_50'] > df['ema_200']).astype(int)

tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
df['atr_14'] = tr.ewm(span=14, adjust=False).mean()
df['norm_atr'] = df['atr_14'] / close

# Create ATR-Scaled Target Labels
fwd_ret = (close.shift(-FORWARD_WINDOW) - close) / close
atr_thresh = (df['atr_14'] / close) * PROFIT_MULTIPLIER
target = pd.Series(0, index=df.index)
target[fwd_ret > atr_thresh] = 1
target[fwd_ret < -atr_thresh] = -1
df['target'] = target
df['target_mapped'] = df['target'].map({-1: 0, 0: 1, 1: 2})

df_clean = df.dropna().copy()
feature_cols = ['rsi_14', 'rsi_7', 'rsi_21', 'rsi_diff1', 'rsi_diff3', 'rsi_diff5', 'rsi_accel', 'rsi_zscore_30', 'rsi_zone', 'cross_30_up', 'cross_50_up', 'cross_70_down', 'cross_50_down', 'bullish_div', 'bearish_div', 'dist_ema_20', 'dist_ema_50', 'trend_50_200', 'norm_atr']

print('Features and Targets ready!')
print('Target Distribution:\\n', df_clean['target'].value_counts(normalize=True))"""
cell5 = nbf.v4.new_code_cell(cell5_code)

# Cell 6: Markdown
cell6 = nbf.v4.new_markdown_cell("""## 3. Train Models & Evaluate Accuracy""")

# Cell 7: Code
cell7_code = """X = df_clean[feature_cols].values
y = df_clean['target_mapped'].values

n = len(df_clean)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]
df_test = df_clean.iloc[val_end:].copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 1. Train XGBoost
sample_weights = compute_sample_weight('balanced', y_train)
clf_xgb = xgb.XGBClassifier(
    n_estimators=XGB_N_ESTIMATORS,
    max_depth=XGB_MAX_DEPTH,
    learning_rate=XGB_LEARNING_RATE,
    min_child_weight=XGB_MIN_CHILD_WEIGHT,
    subsample=0.8, colsample_bytree=0.8, gamma=0.1,
    random_state=RANDOM_SEED
)
clf_xgb.fit(X_train_scaled, y_train, sample_weight=sample_weights, verbose=False)
p_xgb = clf_xgb.predict_proba(X_test_scaled)

# 2. Train PyTorch Deep LSTM
class DeepLSTM(nn.Module):
    def __init__(self, in_dim):
        super(DeepLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, LSTM_HIDDEN_DIM, num_layers=2, batch_first=True)
        self.fc = nn.Linear(LSTM_HIDDEN_DIM, 3)
    def forward(self, x):
        x = self.relu(self.conv1(x.transpose(1, 2))).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def make_seqs(X_d, y_d, s_len):
    Xs, ys = [], []
    for i in range(len(X_d) - s_len):
        Xs.append(X_d[i:i+s_len])
        ys.append(y_d[i+s_len])
    return torch.tensor(np.array(Xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.long)

X_tr_s, y_tr_s = make_seqs(X_train_scaled, y_train, SEQ_LEN)
X_te_s, y_te_s = make_seqs(X_test_scaled, y_test, SEQ_LEN)

model_lstm = DeepLSTM(len(feature_cols))
c_weights = compute_class_weight('balanced', classes=np.unique(y_tr_s.numpy()), y=y_tr_s.numpy())
criterion = nn.CrossEntropyLoss(weight=torch.tensor(c_weights, dtype=torch.float32))
opt = torch.optim.Adam(model_lstm.parameters(), lr=LSTM_LR)

model_lstm.train()
for epoch in range(LSTM_EPOCHS):
    opt.zero_grad()
    out = model_lstm(X_tr_s)
    loss = criterion(out, y_tr_s)
    loss.backward()
    opt.step()

model_lstm.eval()
with torch.no_grad():
    p_lstm = torch.softmax(model_lstm(X_te_s), dim=1).numpy()

# Blended Ensemble Predictions
p_ens = 0.5 * p_xgb[SEQ_LEN:] + 0.5 * p_lstm
preds = np.argmax(p_ens, axis=1)
y_true = y_te_s.numpy()

acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds, average='macro')

print('===========================================================')
print(f'📊 RESULTS ON HELD-OUT TEST DATA (FORWARD WINDOW = {FORWARD_WINDOW} Days)')
print('===========================================================')
print(f'Overall Out-of-Sample Accuracy: {acc*100:.2f}%')
print(f'Macro F1-Score:                 {f1:.4f}')
print('\\nDetailed Classification Report:')
print(classification_report(y_true, preds, target_names=['SELL (-1)', 'HOLD (0)', 'BUY (+1)']))
print('Confusion Matrix (Rows=Actual, Cols=Predicted):')
print(confusion_matrix(y_true, preds))"""
cell7 = nbf.v4.new_code_cell(cell7_code)

# Cell 8: Markdown
cell8 = nbf.v4.new_markdown_cell("""## 4. Backtest Strategy Return vs Buy-and-Hold Benchmark""")

# Cell 9: Code
cell9_code = """df_bt = df_test.iloc[SEQ_LEN:].copy()
inv_map = {0: -1, 1: 0, 2: 1}
raw_sigs = [inv_map[p] for p in preds]

df_bt['daily_return'] = df_bt['Close'].pct_change().fillna(0)
pos = []
curr = 0.0
for s in raw_sigs:
    if s == 1: curr = 1.0
    elif s == -1: curr = 0.0
    pos.append(curr)
df_bt['position'] = pos
df_bt['strategy_return'] = df_bt['position'].shift(1) * df_bt['daily_return']

cum_bench = (1 + df_bt['daily_return']).cumprod() - 1
cum_strat = (1 + df_bt['strategy_return'].fillna(0)).cumprod() - 1

print(f'Total Strategy Return:  {cum_strat.iloc[-1]*100:.2f}%')
print(f'Benchmark Buy-and-Hold: {cum_bench.iloc[-1]*100:.2f}%')"""
cell9 = nbf.v4.new_code_cell(cell9_code)

nb['cells'] = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9]

target_file = r"C:\Users\BhansaLi\Desktop\Multi_Agent\rsi_trading_experiment.ipynb"
with open(target_file, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Clean Jupyter Notebook written to {target_file}")
