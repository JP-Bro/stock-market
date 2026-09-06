import torch
import torch.nn as nn
import numpy as np
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
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits

def create_sequences(X_data, y_data, seq_len=config.SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X_data) - seq_len):
        Xs.append(X_data[i:i+seq_len])
        ys.append(y_data[i+seq_len])
    return torch.tensor(np.array(Xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.long)

class QuantDeepLSTMModel:
    def __init__(self, input_dim, hidden_dim=config.LSTM_HIDDEN_DIM, num_layers=config.LSTM_NUM_LAYERS):
        torch.manual_seed(config.RANDOM_SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradingDeepLSTM(input_dim, hidden_dim, num_layers).to(self.device)
        self.input_dim = input_dim
        
    def fit(self, X_train_seq, y_train_seq, epochs=config.LSTM_EPOCHS, lr=config.LSTM_LR):
        torch.manual_seed(config.RANDOM_SEED)
        classes = np.unique(y_train_seq.numpy())
        weights = compute_class_weight('balanced', classes=classes, y=y_train_seq.numpy())
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        dataset = torch.utils.data.TensorDataset(X_train_seq, y_train_seq)
        g = torch.Generator()
        g.manual_seed(config.RANDOM_SEED)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.LSTM_BATCH_SIZE, shuffle=True, generator=g)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step(total_loss / len(dataloader))
            
        return self
        
    def predict_proba(self, X_seq):
        self.model.eval()
        with torch.no_grad():
            X_seq_tensor = X_seq.to(self.device)
            logits = self.model(X_seq_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
