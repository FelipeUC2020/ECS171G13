import torch
import CNN.cnn_model_yin as CNN
import numpy as np

class ModelLoader:
    def __init__(self, type: str = "CNN"):
        self.type = type
        self.model = None

        self.load_model()

    def load_model(self):
        if self.type == "CNN":
            checkpoint_path = "CNN/checkpoint/cv_demo/fold_5/fold5_best.pt"
            self.model = CNN.CNN()
            if checkpoint_path:
                ckpt = torch.load(checkpoint_path)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    self.model.load_state_dict(ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(ckpt)
            self.model.eval()

        # TODO: Add LSTM model loading
        elif self.type == "LSTM":
            raise NotImplementedError("LSTM model not implemented yet")
            

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

            # out_np = out.detach().numpy()
            # if out_np.shape[0] == 1:
            #     return out_np[0]
            # return out_np
        
        # TODO: Add LSTM model prediction
        elif self.type == "LSTM": 
            raise NotImplementedError("LSTM model not implemented yet")

