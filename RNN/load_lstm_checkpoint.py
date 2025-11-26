import os
import sys

# Ensure we can import project modules
sys.path.append(os.path.abspath('..'))  # add parent directory to sys.path

from tensorflow.keras.models import load_model
import numpy as np

from data_cleanup import DataProcessor

# Paths
CHECKPOINT_PATH = os.path.join('checkpoints', 'lstm_v2.keras')


def load_checkpoint_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
    model = load_model(path)
    return model


def model_predict(model, processor, data):

    # Predict on a small batch from the test set
    preds_scaled = model.predict(data)  # shape: (8, 24)

    # Inverse scale to original kW units (index 0 corresponds to 'Global_active_power')
    scaler = processor.scaler
    scale_factor = scaler.scale_[0]
    min_val = scaler.min_[0]

    preds_unscaled = (preds_scaled - min_val) / scale_factor

    return preds_unscaled