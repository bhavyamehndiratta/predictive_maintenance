import sys
import numpy as np
sys.path.insert(0, '.')

from src.data.loader import load_test
from src.evaluation.analysis import (
    paired_ttest, failure_mode_analysis,
    calibration_analysis, prognostic_horizon
)

# Load predictions and ground truth
rf_preds   = np.load('outputs/models/rf_test_preds.npy')
lstm_preds = np.load('outputs/models/lstm_test_preds.npy')
test, rul_true = load_test('FD001')

# 1. Statistical comparison
paired_ttest(rf_preds, lstm_preds, rul_true)

# 2. Failure mode analysis
error_df = failure_mode_analysis(rf_preds, lstm_preds, rul_true, test)

# 3. Calibration
calibration_analysis(rf_preds,   rul_true, label='Random Forest')
calibration_analysis(lstm_preds, rul_true, label='LSTM')

# 4. Prognostic horizon
prognostic_horizon(rf_preds,   rul_true, test, threshold=20, label='Random Forest')
prognostic_horizon(lstm_preds, rul_true, test, threshold=20, label='LSTM')
