import sys
import numpy as np
sys.path.insert(0, '.')

from src.data.loader import load_train, load_test
from src.features.engineer import (
    drop_low_variance_sensors, add_rolling_features,
    add_trend_slopes, build_feature_matrix, fit_scaler
)
from src.models.random_forest import train_rf, evaluate, save_model

train = load_train('FD001')
test, rul_test = load_test('FD001')

useful_sensors = drop_low_variance_sensors(train)
train = add_rolling_features(train, useful_sensors)
test  = add_rolling_features(test,  useful_sensors)
train = add_trend_slopes(train, useful_sensors)
test  = add_trend_slopes(test,  useful_sensors)

X_train, y_train, feature_cols = build_feature_matrix(train, useful_sensors)
X_test,  _,       _            = build_feature_matrix(test,  useful_sensors)

scaler  = fit_scaler(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

print('Training Random Forest...')
model = train_rf(X_train, y_train)

y_train_pred = model.predict(X_train)
evaluate(y_train, y_train_pred, label='RF on Train')

test['idx'] = range(len(test))
last_rows = test.groupby('engine_id')['idx'].max().values
X_test_last = X_test[last_rows]
y_test_pred = model.predict(X_test_last)
evaluate(rul_test, y_test_pred, label='RF on Test')

save_model(model)
np.save('outputs/models/rf_test_preds.npy', y_test_pred)
print('\nPredictions saved.')
