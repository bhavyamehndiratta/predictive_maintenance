import sys
sys.path.insert(0, '.')
from src.data.loader import load_train, load_test
from src.features.engineer import (
    drop_low_variance_sensors, add_rolling_features,
    add_trend_slopes, build_feature_matrix, fit_scaler, SENSOR_COLS
)

train = load_train('FD001')
test, rul_test = load_test('FD001')

useful_sensors = drop_low_variance_sensors(train)
print(f'Useful sensors ({len(useful_sensors)}): {useful_sensors}')
dropped = [s for s in SENSOR_COLS if s not in useful_sensors]
print(f'Dropped sensors ({len(dropped)}): {dropped}')

train = add_rolling_features(train, useful_sensors)
test  = add_rolling_features(test,  useful_sensors)

train = add_trend_slopes(train, useful_sensors)
test  = add_trend_slopes(test,  useful_sensors)

X_train, y_train, feature_cols = build_feature_matrix(train, useful_sensors)
X_test,  _,       _            = build_feature_matrix(test,  useful_sensors)

print(f'\nX_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape:  {X_test.shape}')
print(f'Total features: {len(feature_cols)}')
print(f'Feature names sample: {feature_cols[:6]}')

scaler  = fit_scaler(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

print(f'\nX_train mean (should be ~0): {X_train.mean():.4f}')
print(f'X_train std  (should be ~1): {X_train.std():.4f}')
print('\nFeature engineering complete.')
