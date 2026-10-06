import torch
import torch.nn as nn
import numpy as np
import copy
from sklearn.utils.class_weight import compute_class_weight
import config

torch.manual_seed(config.RANDOM_SEED)

class TradingDeepLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=config.LSTM_HIDDEN_DIM, num_layers=config.LSTM_NUM_LAYERS, num_classes=3):
        super(TradingDeepLSTM, self).__init__()
        torch.manual_seed(config.RANDOM_SEED)
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.25)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Input_Dim] -> transpose for Conv1d: [Batch, Input_Dim, Seq_Len]
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits

def create_sequences(X_data, y_data=None, seq_len=config.SEQ_LEN):
    """
    Creates temporal sequence windows of shape [N - seq_len, seq_len, num_features].
    When X_data has length == seq_len and y_data is None, returns a single sequence [1, seq_len, num_features]
    containing the exact latest window up to the current bar.
    """
    if len(X_data) == seq_len and y_data is None:
        return torch.tensor(np.array([X_data]), dtype=torch.float32)
        
    Xs, ys = [], []
    for i in range(len(X_data) - seq_len):
        Xs.append(X_data[i:i+seq_len])
        if y_data is not None:
            ys.append(y_data[i+seq_len])
            
    Xs_tensor = torch.tensor(np.array(Xs), dtype=torch.float32)
    if y_data is not None:
        ys_tensor = torch.tensor(np.array(ys), dtype=torch.long)
        return Xs_tensor, ys_tensor
    return Xs_tensor

class QuantDeepLSTMModel:
    def __init__(self, input_dim, hidden_dim=config.LSTM_HIDDEN_DIM, num_layers=config.LSTM_NUM_LAYERS):
        torch.manual_seed(config.RANDOM_SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradingDeepLSTM(input_dim, hidden_dim, num_layers).to(self.device)
        self.input_dim = input_dim
        
    def fit(
        self,
        X_train_seq,
        y_train_seq,
        X_val_seq=None,
        y_val_seq=None,
        epochs=config.LSTM_EPOCHS,
        lr=config.LSTM_LR,
        patience=config.LSTM_EARLY_STOPPING_PATIENCE
    ):
        torch.manual_seed(config.RANDOM_SEED)
        y_train_np = y_train_seq.numpy() if isinstance(y_train_seq, torch.Tensor) else np.array(y_train_seq)
        classes = np.unique(y_train_np)
        weights = compute_class_weight('balanced', classes=classes, y=y_train_np)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        dataset = torch.utils.data.TensorDataset(X_train_seq, y_train_seq)
        g = torch.Generator()
        g.manual_seed(config.RANDOM_SEED)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.LSTM_BATCH_SIZE, shuffle=True, generator=g)
        
        best_val_loss = float('inf')
        best_model_weights = copy.deepcopy(self.model.state_dict())
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_y)
                
            avg_train_loss = train_loss / len(dataset)
            
            # Validation evaluation if validation set provided
            if X_val_seq is not None and y_val_seq is not None and len(X_val_seq) > 0 and len(X_val_seq.shape) == 3:
                self.model.eval()
                with torch.no_grad():
                    val_X = X_val_seq.to(self.device)
                    val_y = y_val_seq.to(self.device)
                    val_outputs = self.model(val_X)
                    val_loss = criterion(val_outputs, val_y).item()
                    
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        # Restore best weights and stop early
                        self.model.load_state_dict(best_model_weights)
                        break
            else:
                scheduler.step(avg_train_loss)
                
        if X_val_seq is not None and y_val_seq is not None:
            self.model.load_state_dict(best_model_weights)
            
        return self
        
    def predict_proba(self, X_seq):
        self.model.eval()
        with torch.no_grad():
            X_seq_tensor = X_seq.to(self.device) if isinstance(X_seq, torch.Tensor) else torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            logits = self.model(X_seq_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
