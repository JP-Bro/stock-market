import xgboost as xgb
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
import config

class QuantXGBoostModel:
    def __init__(self, tune_params=True):
        self.tune_params = tune_params
        self.best_params = {
            'n_estimators': config.XGB_N_ESTIMATORS,
            'max_depth': config.XGB_MAX_DEPTH,
            'learning_rate': config.XGB_LEARNING_RATE,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 1
        }
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        sample_weights = compute_sample_weight('balanced', y_train)

        if self.tune_params and X_val is not None and y_val is not None:
            print("[XGBoost] Hyperparameter tuning in progress...")
            best_score = -1.0
            
            # Grid search candidates
            depth_list = [3, 4, 5]
            lr_list = [0.01, 0.03, 0.05]
            est_list = [150, 250, 350]
            mcw_list = [1, 3]

            for d in depth_list:
                for lr in lr_list:
                    for n_est in est_list:
                        for mcw in mcw_list:
                            clf = xgb.XGBClassifier(
                                n_estimators=n_est,
                                max_depth=d,
                                learning_rate=lr,
                                min_child_weight=mcw,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                gamma=0.1,
                                reg_alpha=0.1,
                                reg_lambda=1.0,
                                random_state=config.RANDOM_SEED
                            )
                            clf.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
                            val_preds = clf.predict(X_val)
                            score = f1_score(y_val, val_preds, average='macro')
                            
                            if score > best_score:
                                best_score = score
                                self.best_params.update({
                                    'n_estimators': n_est,
                                    'max_depth': d,
                                    'learning_rate': lr,
                                    'min_child_weight': mcw
                                })
            print(f"[XGBoost] Optimal parameters found: {self.best_params} (Validation Macro F1: {best_score:.4f})")

        self.model = xgb.XGBClassifier(
            **self.best_params,
            random_state=config.RANDOM_SEED
        )
        
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else []
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False
        )
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importances(self, feature_names):
        importances = self.model.feature_importances_
        return sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
