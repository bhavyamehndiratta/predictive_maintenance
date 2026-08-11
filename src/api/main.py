import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pathlib import Path

from src.data.loader import load_train, load_test
from src.features.engineer import (
    drop_low_variance_sensors, add_rolling_features,
    add_trend_slopes, build_feature_matrix
)
from src.models.lstm import LSTMModel
from src.models.random_forest import phm_score
from sklearn.preprocessing import StandardScaler

app = FastAPI(title='Predictive Maintenance API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── Load everything once at startup ──────────────────────────────────────────
print('Loading data and models...')

train_df = load_train('FD001')
test_df, rul_true = load_test('FD001')

useful_sensors = drop_low_variance_sensors(train_df)

# Feature-engineered versions for RF
train_fe = add_rolling_features(train_df, useful_sensors)
test_fe  = add_rolling_features(test_df,  useful_sensors)
train_fe = add_trend_slopes(train_fe, useful_sensors)
test_fe  = add_trend_slopes(test_fe,  useful_sensors)

X_train, y_train, feature_cols = build_feature_matrix(train_fe, useful_sensors)
X_test,  _,       _            = build_feature_matrix(test_fe,  useful_sensors)

rf_scaler = joblib.load('outputs/models/scaler.pkl')
X_train_scaled = rf_scaler.transform(X_train)
X_test_scaled  = rf_scaler.transform(X_test)

rf_model = joblib.load('outputs/models/rf_model.pkl')

# LSTM scaler (sensor-only)
lstm_scaler = StandardScaler()
train_sensors = train_df[useful_sensors].values
lstm_scaler.fit(train_sensors)

test_df_scaled = test_df.copy()
test_df_scaled[useful_sensors] = lstm_scaler.transform(test_df[useful_sensors])

lstm_model = LSTMModel(input_size=len(useful_sensors), hidden_size=128, num_layers=2, dropout=0.2)
lstm_model.load_state_dict(torch.load('outputs/models/lstm_model.pt', map_location='cpu'))
lstm_model.eval()

rf_preds   = np.load('outputs/models/rf_test_preds.npy')
lstm_preds = np.load('outputs/models/lstm_test_preds.npy')

WINDOW_SIZE = 30
engine_ids  = sorted(test_df['engine_id'].unique().tolist())
print('API ready.')


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get('/engines')
def list_engines():
    return {'engine_ids': engine_ids}


@app.get('/engine/{engine_id}')
def get_engine(engine_id: int):
    if engine_id not in engine_ids:
        raise HTTPException(status_code=404, detail=f'Engine {engine_id} not found')

    idx = engine_ids.index(engine_id)
    group = test_df[test_df['engine_id'] == engine_id].sort_values('cycle')

    # Sensor trajectories
    sensor_data = {
        col: group[col].tolist()
        for col in useful_sensors
    }

    # RF predicted RUL (last row prediction only — single value for test)
    rf_pred = float(rf_preds[idx])

    # LSTM: predict RUL at every cycle for the trajectory
    lstm_cycle_preds = []
    sensor_vals = test_df_scaled[test_df_scaled['engine_id'] == engine_id].sort_values('cycle')[useful_sensors].values
    with torch.no_grad():
        for i in range(len(sensor_vals)):
            start = max(0, i - WINDOW_SIZE + 1)
            seq = sensor_vals[start:i+1]
            if len(seq) < WINDOW_SIZE:
                pad = np.zeros((WINDOW_SIZE - len(seq), len(useful_sensors)))
                seq = np.vstack([pad, seq])
            x = torch.tensor(seq.astype(np.float32)).unsqueeze(0)
            pred = lstm_model(x).item()
            lstm_cycle_preds.append(round(pred, 2))

    cycles = group['cycle'].tolist()
    true_rul_curve = list(range(len(cycles) + int(rul_true[idx]), int(rul_true[idx]) - 1, -1))
    true_rul_curve = true_rul_curve[:len(cycles)]

    return {
        'engine_id':        engine_id,
        'cycles':           cycles,
        'true_rul':         rul_true[idx].item(),
        'true_rul_curve':   true_rul_curve,
        'rf_pred':          rf_pred,
        'lstm_pred_curve':  lstm_cycle_preds,
        'lstm_final_pred':  lstm_cycle_preds[-1] if lstm_cycle_preds else None,
        'sensors':          sensor_data,
    }


@app.get('/metrics')
def get_metrics():
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    def metrics(preds, true, label):
        rmse = float(np.sqrt(mean_squared_error(true, preds)))
        mae  = float(mean_absolute_error(true, preds))
        phm  = phm_score(true, preds)
        return {'model': label, 'rmse': round(rmse, 4), 'mae': round(mae, 4), 'phm': round(phm, 4)}

    return {
        'results': [
            metrics(rf_preds,   rul_true, 'Random Forest'),
            metrics(lstm_preds, rul_true, 'LSTM'),
        ]
    }


@app.get('/errors')
def get_errors():
    import pandas as pd
    path = Path('outputs/results/per_engine_errors.csv')
    if not path.exists():
        raise HTTPException(status_code=404, detail='Run evaluation first')
    df = pd.read_csv(path)
    return df.to_dict(orient='records')
