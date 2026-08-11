import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

SENSORS_TO_DROP = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']

SENSOR_COLS = [f's{i}' for i in range(1, 22)]
OP_COLS = ['op_setting_1', 'op_setting_2', 'op_setting_3']

WINDOW_SIZE = 30


def drop_low_variance_sensors(df: pd.DataFrame, threshold: float = 0.01) -> list[str]:
    stds = df[SENSOR_COLS].std()
    useful = stds[stds > threshold].index.tolist()
    return useful


def add_rolling_features(df: pd.DataFrame, useful_sensors: list[str], window: int = WINDOW_SIZE) -> pd.DataFrame:
    df = df.copy()
    for col in useful_sensors:
        rolled = df.groupby('engine_id')[col].rolling(window=window, min_periods=1)
        df[f'{col}_rmean'] = rolled.mean().reset_index(level=0, drop=True)
        df[f'{col}_rstd']  = rolled.std().reset_index(level=0, drop=True).fillna(0)
        df[f'{col}_rmin']  = rolled.min().reset_index(level=0, drop=True)
        df[f'{col}_rmax']  = rolled.max().reset_index(level=0, drop=True)
    return df


def add_trend_slopes(df: pd.DataFrame, useful_sensors: list[str], window: int = WINDOW_SIZE) -> pd.DataFrame:
    df = df.copy()
    x = np.arange(window)

    def slope(y):
        if len(y) < 2:
            return 0.0
        xi = x[-len(y):]
        xm, ym = xi.mean(), y.mean()
        denom = ((xi - xm) ** 2).sum()
        if denom == 0:
            return 0.0
        return ((xi - xm) * (y - ym)).sum() / denom

    for col in useful_sensors:
        df[f'{col}_slope'] = (
            df.groupby('engine_id')[col]
            .rolling(window=window, min_periods=2)
            .apply(slope, raw=True)
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
    return df


def build_feature_matrix(df: pd.DataFrame, useful_sensors: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_cols = (
        useful_sensors +
        [f'{s}_{stat}' for s in useful_sensors for stat in ['rmean', 'rstd', 'rmin', 'rmax', 'slope']]
    )
    X = df[feature_cols].values
    y = df['RUL'].values if 'RUL' in df.columns else None
    return X, y, feature_cols


def fit_scaler(X: np.ndarray, save_path: str = 'outputs/models/scaler.pkl') -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_path)
    return scaler


def load_scaler(path: str = 'outputs/models/scaler.pkl') -> StandardScaler:
    return joblib.load(path)
