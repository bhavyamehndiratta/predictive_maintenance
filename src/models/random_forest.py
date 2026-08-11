import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = '') -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    phm  = phm_score(y_true, y_pred)
    print(f'\n--- {label} ---')
    print(f'RMSE:      {rmse:.4f}')
    print(f'MAE:       {mae:.4f}')
    print(f'PHM Score: {phm:.4f}')
    return {'rmse': rmse, 'mae': mae, 'phm': phm}


def save_model(model: RandomForestRegressor, path: str = 'outputs/models/rf_model.pkl') -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f'Model saved to {path}')


def load_model(path: str = 'outputs/models/rf_model.pkl') -> RandomForestRegressor:
    return joblib.load(path)
