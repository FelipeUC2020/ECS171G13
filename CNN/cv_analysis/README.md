# Cross-Validation Analysis

This folder contains `analysis.py`, a script to reproduce cross-validation (CV) performance and compare against a baseline single train/val split for the CNN energy forecasting task (72h -> 24h).

## What It Does
1. Loads windowed data via `DataProcessor` (3 days input -> 1 day output, 4 sub-metering channels).
2. Builds time-aware folds with `TimeSeriesSplit`.
3. Runs `cross_validate` saving best fold checkpoints to `cv_ckpts/`.
4. Plots per-fold train/val loss curves and final validation losses.
5. Trains a baseline (non-CV) model and saves its best checkpoint.
6. Evaluates best CV model and baseline on test set, computing:
   - Overall test MSE across all horizons.
   - Per-horizon MSE.
   - Daily sum MSE (aggregated 24h consumption).
7. Saves illustrative forecast plots for sample test windows.
8. Writes consolidated metrics to `metrics.json`.

## Generated Artifacts
```
cv_analysis/
  analysis.py
  figs/
    cv_loss_curves.png
    cv_final_val_losses.png
    sample_0_forecast.png
    sample_1_forecast.png
  cv_ckpts/
    fold_1/fold1_best.pt ... (etc.)
  baseline_ckpt/
    baseline_best.pt
  metrics.json
```

## Usage
Activate your environment and run:
```bash
python cv_analysis/analysis.py
```
Figures will be located under `cv_analysis/figs/`. Include them in the LaTeX report with `\includegraphics`.

## Extending
- Increase `epochs` when calling `run_cv` or baseline training for more stable metrics.
- Adjust `CHANNEL_INDICES` or window sizes in the script if upstream preprocessing changes.
- Add additional metrics (e.g., MAE) by extending `evaluate_model`.
