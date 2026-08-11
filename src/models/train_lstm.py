import sys
import numpy as np
import torch
sys.path.insert(0, '.')

from src.data.loader import load_train, load_test
from src.features.engineer import (
    drop_low_variance_sensors, add_rolling_features,
    add_trend_slopes, fit_scaler
)
from src.models.lstm import CMAPSSDataset, LSTMModel, train_lstm
from src.models.random_forest import evaluate, phm_score
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 30
BATCH_SIZE  = 256
EPOCHS      = 50
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

# Load
train = load_train('FD001')
test, rul_test = load_test('FD001')

# Feature engineering
useful_sensors = drop_low_variance_sensors(train)
train = add_rolling_features(train, useful_sensors)
test  = add_rolling_features(test,  useful_sensors)
train = add_trend_slopes(train, useful_sensors)
test  = add_trend_slopes(test,  useful_sensors)

# Normalize raw sensors only (LSTM uses raw sensor window, not engineered features)
scaler = StandardScaler()
train[useful_sensors] = scaler.fit_transform(train[useful_sensors])
test[useful_sensors]  = scaler.transform(test[useful_sensors])

# Build datasets
train_dataset = CMAPSSDataset(train, useful_sensors, window_size=WINDOW_SIZE)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Train
model = LSTMModel(input_size=len(useful_sensors), hidden_size=128, num_layers=2, dropout=0.2)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
model = train_lstm(model, train_loader, epochs=EPOCHS, device=DEVICE)

# Evaluate on test set — take last window per engine
model.eval()
preds = []
with torch.no_grad():
    for engine_id, group in test.groupby('engine_id'):
        group = group.sort_values('cycle')
        sensor_data = group[useful_sensors].values
        seq = sensor_data[-WINDOW_SIZE:]
        if len(seq) < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE - len(seq), len(useful_sensors)))
            seq = np.vstack([pad, seq])
        x = torch.tensor(seq.astype(np.float32)).unsqueeze(0).to(DEVICE)
        pred = model(x).item()
        preds.append(pred)

preds = np.array(preds)
evaluate(rul_test, preds, label='LSTM on Test')

# Save
torch.save(model.state_dict(), 'outputs/models/lstm_model.pt')
np.save('outputs/models/lstm_test_preds.npy', preds)
print('\nLSTM model and predictions saved.')
