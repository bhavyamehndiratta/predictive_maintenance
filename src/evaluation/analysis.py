import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path('outputs/results').mkdir(parents=True, exist_ok=True)


def paired_ttest(rf_preds, lstm_preds, rul_true):
    """
    Paired t-test on absolute errors across 100 test engines.
    Tests whether LSTM errors are significantly lower than RF errors.
    """
    rf_errors   = np.abs(rf_preds - rul_true)
    lstm_errors = np.abs(lstm_preds - rul_true)
    t_stat, p_value = stats.ttest_rel(rf_errors, lstm_errors)
    print('\n--- Paired T-Test (RF vs LSTM absolute errors) ---')
    print(f'RF   mean abs error:   {rf_errors.mean():.4f}')
    print(f'LSTM mean abs error:   {lstm_errors.mean():.4f}')
    print(f'T-statistic:           {t_stat:.4f}')
    print(f'P-value:               {p_value:.6f}')
    if p_value < 0.05:
        print('Result: LSTM improvement is statistically significant (p < 0.05)')
    else:
        print('Result: Difference is NOT statistically significant (p >= 0.05)')
    return t_stat, p_value


def failure_mode_analysis(rf_preds, lstm_preds, rul_true, test_df):
    """
    Break down errors by engine. Identify which engines have worst predictions and why.
    """
    rf_errors   = np.abs(rf_preds - rul_true)
    lstm_errors = np.abs(lstm_preds - rul_true)

    engine_ids = test_df.groupby('engine_id')['cycle'].max().index.values
    max_cycles = test_df.groupby('engine_id')['cycle'].max().values

    df = pd.DataFrame({
        'engine_id':   engine_ids,
        'max_cycle':   max_cycles,
        'rul_true':    rul_true,
        'rf_pred':     rf_preds,
        'lstm_pred':   lstm_preds,
        'rf_error':    rf_errors,
        'lstm_error':  lstm_errors,
    })

    print('\n--- Failure Mode Analysis ---')
    print('Top 10 highest-error engines (LSTM):')
    print(df.nlargest(10, 'lstm_error')[['engine_id','rul_true','lstm_pred','lstm_error','max_cycle']].to_string(index=False))

    print('\nError by trajectory length bucket:')
    df['length_bucket'] = pd.cut(df['max_cycle'], bins=[0,100,200,300,400], labels=['<100','100-200','200-300','>300'])
    print(df.groupby('length_bucket', observed=True)[['rf_error','lstm_error']].mean().round(2))

    df.to_csv('outputs/results/per_engine_errors.csv', index=False)
    print('\nSaved to outputs/results/per_engine_errors.csv')
    return df


def calibration_analysis(preds, rul_true, label='Model'):
    """
    Calibration plot: how well do predicted RUL values match actual RUL values across bins.
    A well-calibrated model's points fall close to the diagonal.
    """
    bins = np.linspace(0, 125, 11)
    bin_centers, mean_pred, mean_true = [], [], []

    for i in range(len(bins)-1):
        mask = (rul_true >= bins[i]) & (rul_true < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            mean_pred.append(preds[mask].mean())
            mean_true.append(rul_true[mask].mean())

    plt.figure(figsize=(7, 6))
    plt.plot([0, 125], [0, 125], 'k--', label='Perfect calibration')
    plt.scatter(mean_true, mean_pred, s=80, zorder=5)
    plt.xlabel('Mean Actual RUL')
    plt.ylabel('Mean Predicted RUL')
    plt.title(f'Calibration Plot — {label}')
    plt.legend()
    plt.tight_layout()
    path = f'outputs/results/calibration_{label.lower().replace(" ","_")}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'\nCalibration plot saved to {path}')


def prognostic_horizon(preds, rul_true, test_df, threshold=20, label='Model'):
    """
    Prognostic horizon: at how many cycles before failure does the model
    start predicting within +/- threshold cycles of true RUL?
    Higher is better — means the model is reliable further from failure.
    """
    engine_ids = test_df.groupby('engine_id')['engine_id'].first().values
    horizons = []

    for i, engine_id in enumerate(engine_ids):
        group = test_df[test_df['engine_id'] == engine_id].sort_values('cycle')
        true_rul = rul_true[i]
        pred_rul = preds[i]
        error = abs(pred_rul - true_rul)
        # How many cycles before end-of-life is this prediction?
        cycles_before_failure = true_rul
        if error <= threshold:
            horizons.append(cycles_before_failure)

    horizons = np.array(horizons)
    print(f'\n--- Prognostic Horizon ({label}) ---')
    print(f'Threshold: ±{threshold} cycles')
    print(f'Engines within threshold: {len(horizons)}/{len(engine_ids)} ({100*len(horizons)/len(engine_ids):.1f}%)')
    if len(horizons) > 0:
        print(f'Mean cycles before failure when reliable: {horizons.mean():.1f}')
        print(f'Max  cycles before failure when reliable: {horizons.max():.1f}')
    return horizons
