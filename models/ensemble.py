import numpy as np

class HybridEnsembleModel:
    """
    Hybrid Quant AI Ensemble combining XGBoost non-linear tabular trees and
    PyTorch 1D-CNN + LSTM Deep Neural Network sequential predictions.
    """
    def __init__(self, xgb_weight=0.5, lstm_weight=0.5):
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        
    def predict_proba(self, xgb_probs, lstm_probs):
        """
        Blends probability distributions from XGBoost and PyTorch Deep LSTM.
        """
        blended_probs = self.xgb_weight * xgb_probs + self.lstm_weight * lstm_probs
        return blended_probs
        
    def predict_with_thresholds(self, probs, buy_threshold=0.38, sell_threshold=0.38):
        """
        Applies calibrated probability thresholds:
        - Class 0: SELL (-1)
        - Class 1: HOLD (0)
        - Class 2: BUY (+1)
        """
        preds = []
        for p in probs:
            p_sell, p_hold, p_buy = p[0], p[1], p[2]
            if p_buy > buy_threshold and p_buy > p_sell:
                preds.append(2) # BUY
            elif p_sell > sell_threshold and p_sell > p_buy:
                preds.append(0) # SELL
            else:
                preds.append(1) # HOLD
        return np.array(preds)
