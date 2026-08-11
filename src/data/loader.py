import pandas as pd
import numpy as np
from pathlib import Path

COLUMNS = [
    'engine_id', 'cycle',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'
]

MAX_RUL = 125


def load_train(fd: str = 'FD001', data_dir: str = 'data/raw') -> pd.DataFrame:
    path = Path(data_dir) / f'train_{fd}.txt'
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)
    max_cycles = df.groupby('engine_id')['cycle'].max().rename('max_cycle')
    df = df.join(max_cycles, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns='max_cycle', inplace=True)
    df['RUL'] = df['RUL'].clip(upper=MAX_RUL)
    return df


def load_test(fd: str = 'FD001', data_dir: str = 'data/raw') -> tuple[pd.DataFrame, np.ndarray]:
    path = Path(data_dir) / f'test_{fd}.txt'
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)
    rul_path = Path(data_dir) / f'RUL_{fd}.txt'
    rul = pd.read_csv(rul_path, header=None, names=['RUL']).values.flatten()
    return df, rul


def summarise(df: pd.DataFrame, label: str = 'train') -> None:
    print(f'\n--- {label} ---')
    print(f'Engines:   {df["engine_id"].nunique()}')
    print(f'Rows:      {len(df)}')
    print(f'Cycles:    min={df["cycle"].min()}  max={df["cycle"].max()}')
    if 'RUL' in df.columns:
        print(f'RUL:       min={df["RUL"].min()}  max={df["RUL"].max()}  mean={df["RUL"].mean():.1f}')
