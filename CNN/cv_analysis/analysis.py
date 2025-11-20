"""Cross-validation analysis and visualization for CNN energy forecasting.
Generates figures saved under cv_analysis/figs/.

Steps:
1. Load data via DataProcessor (72h -> 24h windows).
2. Build TimeSeriesSplit folds.
3. Run cross_validate to collect per-fold histories.
4. Plot per-fold train/val loss curves, and final val loss comparison.
5. Train baseline single model on full training set (train+val combined optional) or original train/val split.
6. Evaluate best CV model and baseline on test set; compute MSE per horizon and aggregated daily sum error.
7. Output summary metrics to console and write metrics.json.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# Relative path adjustments (assuming script executed from CNN/ or repository root)
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from data_cleanup import DataProcessor  # type: ignore
from cnn_model_yin import CNN, cross_validate, train

INPUT_STEPS = 24 * 3  # 72h
OUTPUT_STEPS = 24     # 24h
CHANNEL_INDICES = [4,5,6,7]
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

FIG_DIR = os.path.join(SCRIPT_DIR, 'figs')
os.makedirs(FIG_DIR, exist_ok=True)


def build_data():
    processor = DataProcessor(input_steps=INPUT_STEPS, output_steps=OUTPUT_STEPS, local_raw_path='../raw-consumption-data.zip')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.load_and_process_data()
    # Select channels
    X_train_ch = X_train[:, :, CHANNEL_INDICES]
    X_val_ch = X_val[:, :, CHANNEL_INDICES]
    X_test_ch = X_test[:, :, CHANNEL_INDICES]
    return processor, (X_train_ch, y_train), (X_val_ch, y_val), (X_test_ch, y_test)


def make_model():
    return CNN(in_channels=4, input_length=INPUT_STEPS, output_steps=OUTPUT_STEPS)


def run_cv(X_train_ch, y_train, n_splits=5, epochs=5, batch_size=32):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, val_idx in tscv.split(X_train_ch):
        folds.append(((X_train_ch[train_idx], y_train[train_idx]), (X_train_ch[val_idx], y_train[val_idx])))
    histories, val_losses, best_model = cross_validate(
        make_model,
        folds,
        device=None,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        verbose=True,
        checkpoint_dir=os.path.join(SCRIPT_DIR, 'cv_ckpts'),
        save_best_only=True
    )
    return histories, val_losses, best_model


def plot_losses(histories, val_losses):
    # Per-fold curves
    plt.figure(figsize=(10,6))
    for i, hist in enumerate(histories, start=1):
        plt.plot(hist['train_loss'], label=f'Fold {i} Train', alpha=0.6)
        if 'val_loss' in hist:
            plt.plot(hist['val_loss'], label=f'Fold {i} Val', linestyle='--', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Per-fold Train/Validation Loss Curves')
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3)
    path_curves = os.path.join(FIG_DIR, 'cv_loss_curves.png')
    plt.tight_layout()
    plt.savefig(path_curves, dpi=150)
    plt.close()

    # Final val losses bar
    plt.figure(figsize=(6,4))
    plt.bar(range(1, len(val_losses)+1), val_losses, color='teal')
    plt.xlabel('Fold')
    plt.ylabel('Final Validation Loss (MSE)')
    plt.title('Final Validation Loss per Fold')
    for i, v in enumerate(val_losses, start=1):
        plt.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    path_bar = os.path.join(FIG_DIR, 'cv_final_val_losses.png')
    plt.tight_layout()
    plt.savefig(path_bar, dpi=150)
    plt.close()
    return {'curves': path_curves, 'bar': path_bar}


def evaluate_model(model, X_test_ch, y_test, processor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test_ch, dtype=torch.float32).to(device)
        if Xt.ndim == 3 and Xt.shape[2] == 4 and Xt.shape[1] != 4:
            Xt = Xt.permute(0, 2, 1)
        preds = model(Xt)
        yt = torch.tensor(y_test, dtype=torch.float32).to(device)
        preds_np = preds.cpu().numpy()
        yt_np = yt.cpu().numpy()
    # Inverse transform stepwise
    preds_un = np.zeros_like(preds_np)
    yt_un = np.zeros_like(yt_np)
    for t in range(preds_np.shape[1]):
        try:
            preds_un[:, t] = processor.inverse_transform_predictions(preds_np[:, t])
            yt_un[:, t] = processor.inverse_transform_predictions(yt_np[:, t])
        except Exception:
            preds_un[:, t] = preds_np[:, t]
            yt_un[:, t] = yt_np[:, t]
    mse_per_horizon = np.mean((preds_un - yt_un)**2, axis=0)
    mse_all = float(np.mean((preds_un - yt_un)**2))
    # Daily sum error (aggregate 24h predicted vs actual sum)
    pred_sum = preds_un.sum(axis=1)
    actual_sum = yt_un.sum(axis=1)
    mse_sum = float(np.mean((pred_sum - actual_sum)**2))
    return {
        'mse_all': mse_all,
        'mse_per_horizon': mse_per_horizon.tolist(),
        'mse_daily_sum': mse_sum,
        'preds_un': preds_un,
        'yt_un': yt_un
    }


def plot_sample(preds_un, yt_un, sample_idx=0):
    plt.figure(figsize=(8,3))
    plt.plot(yt_un[sample_idx], marker='o', label='Actual')
    plt.plot(preds_un[sample_idx], marker='x', label='Predicted')
    plt.title(f'Sample {sample_idx} 24h Horizon')
    plt.xlabel('Hour Ahead')
    plt.ylabel('Load (unscaled)')
    plt.legend(); plt.grid(True, alpha=0.3)
    path = os.path.join(FIG_DIR, f'sample_{sample_idx}_forecast.png')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


def baseline_training(X_train_ch, y_train, X_val_ch, y_val):
    model = make_model()
    hist = train(model, X_train_ch, y_train, X_val=X_val_ch, y_val=y_val,
                 device=None, epochs=5, batch_size=32, lr=1e-3, verbose=True,
                 checkpoint_dir=os.path.join(SCRIPT_DIR, 'baseline_ckpt'), checkpoint_prefix='baseline', save_best_only=True)
    return model, hist


def main():
    processor, (X_train_ch, y_train), (X_val_ch, y_val), (X_test_ch, y_test) = build_data()
    print('Data loaded. Train:', X_train_ch.shape, 'Val:', X_val_ch.shape, 'Test:', X_test_ch.shape)

    print('\nRunning cross-validation...')
    histories, val_losses, best_cv_model = run_cv(X_train_ch, y_train, n_splits=5, epochs=5)
    fig_paths = plot_losses(histories, val_losses)
    print('Saved CV loss figures:', fig_paths)

    print('\nTraining baseline single model...')
    baseline_model, baseline_hist = baseline_training(X_train_ch, y_train, X_val_ch, y_val)

    print('\nEvaluating CV best model on test set...')
    cv_metrics = evaluate_model(best_cv_model, X_test_ch, y_test, processor)
    cv_sample = plot_sample(cv_metrics['preds_un'], cv_metrics['yt_un'], sample_idx=0)

    print('Evaluating baseline model on test set...')
    base_metrics = evaluate_model(baseline_model, X_test_ch, y_test, processor)
    base_sample = plot_sample(base_metrics['preds_un'], base_metrics['yt_un'], sample_idx=1)

    summary = {
        'cv_final_val_losses': val_losses,
        'cv_test_mse_all': cv_metrics['mse_all'],
        'cv_test_mse_daily_sum': cv_metrics['mse_daily_sum'],
        'baseline_last_val_loss': baseline_hist['val_loss'][-1] if 'val_loss' in baseline_hist else None,
        'baseline_test_mse_all': base_metrics['mse_all'],
        'baseline_test_mse_daily_sum': base_metrics['mse_daily_sum'],
        'figures': {
            'cv_loss_curves': fig_paths['curves'],
            'cv_final_val_losses': fig_paths['bar'],
            'cv_sample_forecast': cv_sample,
            'baseline_sample_forecast': base_sample
        }
    }
    metrics_path = os.path.join(SCRIPT_DIR, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('\nSummary metrics written to', metrics_path)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
