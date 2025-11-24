import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Callable, List, Tuple
import os
from torch.utils.data import TensorDataset, DataLoader


class CNN(nn.Module):
    """CNN for multistep forecasting from multichannel time-series.

    Input shape expected for training data: (batch, time, channels).
    This model internally expects (batch, channels, time) for Conv1d.

    By default this class is configured for:
      - in_channels = 4 (sub_metering_1/2/3/rest)
      - input_length = 72 (3 days * 24 hours)
      - output_steps = 24 (1 day * 24 hours)
    """

    def __init__(self,
                 in_channels: int = 4,
                 input_length: int = 24 * 3,
                 output_steps: int = 24,
                 conv_channels: List[int] = [32, 64, 128],
                 kernel_size: int = 3,
                 pool_kernel: int = 2,
                 fc_hidden: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.input_length = input_length
        self.output_steps = output_steps

        # build conv layers
        convs = []
        prev_ch = in_channels
        for ch in conv_channels:
            convs.append(nn.Conv1d(prev_ch, ch, kernel_size=kernel_size, padding=kernel_size//2))
            convs.append(nn.ReLU())
            convs.append(nn.MaxPool1d(kernel_size=pool_kernel))
            prev_ch = ch
        self.feature_extractor = nn.Sequential(*convs)

        # compute flattened size by passing a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            feat = self.feature_extractor(dummy)
            flattened = int(feat.numel() // feat.shape[0])

        self.fc1 = nn.Linear(flattened, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, output_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: tensor shape (batch, channels, time) or (batch, time, channels) - code calling should ensure format.
        Returns:
            preds: tensor shape (batch, output_steps)
        """
        # assume input may be (batch, time, channels); if so, detect and permute
        if x.ndim == 3 and x.shape[1] != self.in_channels and x.shape[2] == self.in_channels:
            # x is (batch, time, channels) -> permute
            x = x.permute(0, 2, 1)

        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        h = F.relu(self.fc1(features))
        out = self.fc2(h)
        return out


def train(model: nn.Module,
          X_train,
          y_train,
          X_val=None,
          y_val=None,
          device: Optional[torch.device] = None,
          epochs: int = 10,
          batch_size: int = 32,
          lr: float = 1e-3,
          optimizer_cls: Callable = optim.Adam,
          optimizer_kwargs: Optional[dict] = None,
          criterion: Callable = nn.MSELoss(),
          shuffle: bool = False,
          verbose: bool = True,
          checkpoint_dir: Optional[str] = None,
          checkpoint_prefix: str = "model",
          save_best_only: bool = True) -> dict:
    """Train helper accepting numpy arrays (n, time, channels) and multi-step targets (n, out_steps).

    Returns history dict with train_loss and optionally val_loss.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer_kwargs = optimizer_kwargs or {'lr': lr}
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    # convert to tensors
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float32)
    if X_val is not None and not isinstance(X_val, torch.Tensor):
        X_val = torch.tensor(X_val, dtype=torch.float32)
    if y_val is not None and not isinstance(y_val, torch.Tensor):
        y_val = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

    # history dict: lists for loss tracking; scalar metadata kept separately
    history: dict = {'train_loss': []}
    if X_val is not None and y_val is not None:
        history['val_loss'] = []
    # store metadata under distinct keys as scalars
    history['best_val_loss'] = float('inf')
    history['checkpoint_path'] = ""

    best_val = float('inf')
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            # xb: (batch, time, channels) -> permute to (batch, channels, time)
            if xb.shape[1] == model.in_channels:
                # already (batch, channels, time)
                xb_t = xb.to(device)
            else:
                xb_t = xb.to(device).permute(0, 2, 1)
            yb_t = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb_t)
            loss = criterion(preds, yb_t)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(1, n_batches)
        history['train_loss'].append(avg_train_loss)

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                Xv = X_val.to(device) if isinstance(X_val, torch.Tensor) else torch.tensor(X_val, dtype=torch.float32).to(device)
                if Xv.shape[1] != model.in_channels:
                    Xv = Xv.permute(0, 2, 1)
                yv = y_val.to(device) if isinstance(y_val, torch.Tensor) else torch.tensor(y_val, dtype=torch.float32).to(device)
                vpreds = model(Xv)
                val_loss = float(criterion(vpreds, yv).item())
            history['val_loss'].append(val_loss)
            if verbose:
                print(f"Epoch {epoch}/{epochs} - train_loss: {avg_train_loss:.6f} - val_loss: {val_loss:.6f}")
            # checkpoint saving logic
            if checkpoint_dir:
                improved = val_loss < best_val
                if improved:
                    best_val = val_loss
                    history['best_val_loss'] = best_val
                    if save_best_only:
                        ckpt_name = f"{checkpoint_prefix}_best.pt"
                    else:
                        ckpt_name = f"{checkpoint_prefix}_epoch{epoch}.pt"
                    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val
                    }, ckpt_path)
                    history['checkpoint_path'] = ckpt_path
                elif (checkpoint_dir and not save_best_only):
                    ckpt_name = f"{checkpoint_prefix}_epoch{epoch}.pt"
                    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val if best_val != float('inf') else None
                    }, ckpt_path)
                    history['checkpoint_path'] = ckpt_path
        else:
            if verbose:
                print(f"Epoch {epoch}/{epochs} - train_loss: {avg_train_loss:.6f}")

    return history


def cross_validate(model_cls: Callable[..., nn.Module],
                   folds: List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]],
                   device: Optional[torch.device] = None,
                   epochs: int = 5,
                   batch_size: int = 32,
                   lr: float = 1e-3,
                   optimizer_cls: Callable = optim.Adam,
                   optimizer_kwargs: Optional[dict] = None,
                   criterion: Callable = nn.MSELoss(),
                   shuffle: bool = False,
                   verbose: bool = True,
                   checkpoint_dir: Optional[str] = None,
                   save_best_only: bool = True) -> Tuple[List[dict], List[Optional[float]], Optional[str]]:
    """Run cross-validation over folds. Each fold is ((X_tr,y_tr),(X_val,y_val)).

    Returns (histories, val_losses, best_checkpoint_path) where best_checkpoint_path is the path to the model with lowest final val_loss.
    """
    histories = []
    val_losses = []
    best_checkpoint_path = None
    best_loss = float('inf')

    for i, ((X_tr, y_tr), (X_val, y_val)) in enumerate(folds, start=1):
        if verbose:
            print(f"Starting fold {i}/{len(folds)}: train {X_tr.shape} | val {X_val.shape}")

        model = model_cls()
        fold_ckpt_dir = None
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            fold_ckpt_dir = os.path.join(checkpoint_dir, f"fold_{i}")
        hist = train(model, X_tr, y_tr, X_val=X_val, y_val=y_val,
                     device=device, epochs=epochs, batch_size=batch_size, lr=lr,
                     optimizer_cls=optimizer_cls, optimizer_kwargs=optimizer_kwargs,
                     criterion=criterion, shuffle=shuffle, verbose=verbose,
                     checkpoint_dir=fold_ckpt_dir, checkpoint_prefix=f"fold{i}", save_best_only=save_best_only)

        histories.append(hist)
        if 'val_loss' in hist and len(hist['val_loss']) > 0:
            final_val_loss = hist['val_loss'][-1]
            val_losses.append(final_val_loss)
            if final_val_loss < best_loss:
                best_loss = final_val_loss
                best_checkpoint_path = hist.get('checkpoint_path')
        else:
            val_losses.append(None)

    return histories, val_losses, best_checkpoint_path

    
