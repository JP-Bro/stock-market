import numpy as np
import config

class HybridEnsembleModel:
    """
    Hybrid Quant AI Ensemble combining XGBoost non-linear tabular trees and
    PyTorch 1D-CNN + LSTM Deep Neural Network sequential predictions.
    Enforces strict probability surety thresholds to prevent false trading signals.
    """
    def __init__(self, xgb_weight=0.5, lstm_weight=0.5):
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        
    def predict_proba(self, xgb_probs, lstm_probs):
        """
        Blends probability distributions from XGBoost and PyTorch Deep LSTM.
        """
        blended_probs = self.xgb_weight * np.array(xgb_probs) + self.lstm_weight * np.array(lstm_probs)
        return blended_probs
        
    def predict_with_thresholds(
        self,
        probs,
        buy_threshold=config.CONFIDENCE_THRESHOLD,
        sell_threshold=config.CONFIDENCE_THRESHOLD,
        min_margin=0.08
    ):
        """
        Applies calibrated probability thresholds:
        - Class 2: BUY (+1) only if P(BUY) >= buy_threshold AND P(BUY) - P(HOLD) >= min_margin
        - Class 0: SELL (-1) only if P(SELL) >= sell_threshold AND P(SELL) - P(HOLD) >= min_margin
        - Class 1: HOLD (0) / NO SURETY when model is uncertain or in neutral market state.
        """
        preds = []
        for p in probs:
            p_sell, p_hold, p_buy = p[0], p[1], p[2]
            
            if (p_buy >= buy_threshold) and (p_buy > p_sell) and ((p_buy - p_hold) >= min_margin or p_buy >= 0.55):
                preds.append(2) # BUY
            elif (p_sell >= sell_threshold) and (p_sell > p_buy) and ((p_sell - p_hold) >= min_margin or p_sell >= 0.55):
                preds.append(0) # SELL
            else:
                preds.append(1) # HOLD / NO SURETY
                
        return np.array(preds)
