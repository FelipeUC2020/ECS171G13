# model_loader.py
# This module handles loading and managing different prediction models.
# To add a new model:
# 1. Define a function to load the model (e.g., load_cnn_model).
# 2. Add it to the MODELS dictionary with a key (e.g., 'cnn').
# 3. In main.py, select the model by key and call the loader.

import torch
import os
from .model.cv_cnn.cnn_model_yin import CNN  # Example import for CNN model

device = torch.device('cpu')  # Use CPU for simplicity

def load_cnn_model(model_rel_path):
    """Load the CNN model for 24-hour energy forecasting."""
    model_path = os.path.join(os.path.dirname(__file__), "model", model_rel_path)
    model = CNN(in_channels=4, input_length=72, output_steps=24)
    if os.path.isfile(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'Loaded model from {model_path}')
    else:
        print(f'Checkpoint not found: {model_path}')
    model.to(device)
    model.eval()
    return model

# TODO: Add load_rnn_model, load_transformer_model, etc.

# Dictionary of available models
# Add new models here: MODELS['new_key'] = load_new_model_function
MODELS = {
    'cv72-24': load_cnn_model("cv_cnn/cv72-24.pt"),
    # Example for adding another model:
    # 'rnn': load_rnn_model,  # Uncomment and define load_rnn_model
}

# TODO: Implement load_rnn_model for RNN/LSTM support
# def load_rnn_model(model_rel_path):
#     # Load RNN/LSTM model
#     pass

def get_model(model_key='cv72-24', **kwargs):
    """Get a loaded model by key."""
    if model_key not in MODELS:
        raise ValueError(f"Model '{model_key}' not found. Available: {list(MODELS.keys())}")
    return MODELS[model_key]