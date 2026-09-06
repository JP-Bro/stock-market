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

# Asset & Target Config
TICKER = "ABREL.NS"
ASSET_NAME = "Aditya Birla Real Estate Limited"
START_DATE = "2017-09-01"
# END_DATE dynamically defaults to today's current date
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Target Labeling Config
FORWARD_WINDOW = 5
PROFIT_MULTIPLIER = 0.8

# Time Series Split (70% Train, 15% Val, 15% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Sequence Model Window
SEQ_LEN = 15

# Hyperparameters
XGB_N_ESTIMATORS = 250
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.03

LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_EPOCHS = 50
LSTM_LR = 0.001
LSTM_BATCH_SIZE = 32

# Paths
BASE_DIR = r"C:\Users\BhansaLi\Desktop\Multi_Agent"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)
