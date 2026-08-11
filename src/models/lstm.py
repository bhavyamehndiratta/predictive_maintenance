import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CMAPSSDataset(Dataset):
    def __init__(self, df, useful_sensors, window_size=30, rul_clip=125):
        self.sequences = []
        self.labels = []
        for engine_id, group in df.groupby('engine_id'):
            group = group.sort_values('cycle')
            sensor_data = group[useful_sensors].values
            rul_data = group['RUL'].values
            for i in range(len(group)):
                start = max(0, i - window_size + 1)
                seq = sensor_data[start:i+1]
                # Pad if sequence shorter than window
                if len(seq) < window_size:
                    pad = np.zeros((window_size - len(seq), seq.shape[1]))
                    seq = np.vstack([pad, seq])
                self.sequences.append(seq.astype(np.float32))
                self.labels.append(rul_data[i].astype(np.float32))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last timestep
        out = self.fc(out)
        return out.squeeze(-1)


def train_lstm(model, train_loader, epochs=50, lr=0.001, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}')

    return model
