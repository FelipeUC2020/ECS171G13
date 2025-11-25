import torch
import CNN.cnn_model_yin as CNN
import numpy as np
from tensorflow.keras.models import load_model as load_lstm_model

class ModelLoader:
    def __init__(self, type: str = "CNN"):
        self.type = type
        self.model = None

        self.load_model()

    def load_model(self):
        if self.type == "CNN":
            checkpoint_path = "CNN/checkpoint/cv_24to24/fold_5/fold5_best.pt"
            self.model = CNN.CNN(8, 24, 24, kernel_size=3, pool_kernel=0, padding=False)
            if checkpoint_path:
                ckpt = torch.load(checkpoint_path)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    self.model.load_state_dict(ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(ckpt)
            self.model.eval()

        elif self.type == "LSTM":
            model_path = "RNN/checkpoints/lstm_v2.keras"
            self.model = load_lstm_model(model_path)
            
            
    def get_predictions(self, data):
        # should return a simple array with the global active pwer consumption for each hour 
        if self.type == "CNN":
            if isinstance(data, np.ndarray):
                x = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, torch.Tensor):
                x = data.to(torch.float32)
            else:
                raise TypeError("data must be a numpy array or torch tensor")

            if x.ndim == 2:
                x = x.unsqueeze(0)
            with torch.no_grad():
                return self.model(x)

        elif self.type == "LSTM": 
            # Ensure numpy float32 array and add batch dimension if a single sample is provided
            if isinstance(data, np.ndarray):
                x = data.astype(np.float32, copy=False)
            else:
                x = np.array(data, dtype=np.float32)
            if x.ndim == 2:
                x = np.expand_dims(x, axis=0)
            return self.model.predict(x)
