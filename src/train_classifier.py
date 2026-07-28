"""PyTorch classification model for market regime prediction."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd


class RegimeClassifier(nn.Module):
    """MLP classifier."""

    def __init__(self, input_dim: int, n_classes: int, hidden_dims: list[int] = None,
                 dropout: float = 0.3, use_bn: bool = True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        out = self.backbone(x)
        return self.head(out)


class RegimeLSTM(nn.Module):
    """LSTM classifier for sequential factor data."""

    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        n_dirs = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_input = hidden_dim * n_dirs
        self.fc = nn.Sequential(
            nn.Linear(fc_input, fc_input),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input, fc_input // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input // 2, n_classes),
        )

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h = h_n[-1]
        return self.fc(h)


class RegimeTransformer(nn.Module):
    """Transformer classifier for sequential factor data."""

    def __init__(self, input_dim: int, n_classes: int, d_model: int = 64,
                 nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.pos_drop(x)
        x = self.encoder(x)
        # Global average pooling over sequence
        x = x.mean(dim=1)
        return self.fc(x)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""

    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class RegimeTrainer:
    """Train/evaluate regime classifier."""

    def __init__(self, model, n_classes: int, lr: float = 1e-3, weight_decay: float = 1e-4,
                 use_focal: bool = True, class_weights: np.ndarray = None):
        self.model = model
        self.n_classes = n_classes
        self.use_focal = use_focal

        # Class weights for imbalance
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        if use_focal:
            self.criterion = FocalLoss(alpha=self.class_weights, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights, label_smoothing=0.1
            )

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        total_samples = 0
        for X_batch, y_batch in loader:
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
        return total_loss / total_samples

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        for X_batch, y_batch in loader:
            logits = self.model(X_batch)
            loss = nn.CrossEntropyLoss(reduction='sum')(logits, y_batch)
            total_loss += loss.item()  # reduction='sum' already sums over batch
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
        avg_loss = total_loss / total
        acc = correct / total
        return avg_loss, acc

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 200, patience: int = 30):
        best_val_acc = 0
        best_state = None
        wait = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={self.scheduler.get_last_lr()[0]:.6f}")

            if wait >= patience:
                print(f"  Early stop at epoch {epoch}, best val_acc={best_val_acc:.4f}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_val_acc


def predict_proba(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
    return np.vstack(all_probs), np.concatenate(all_preds)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


if __name__ == "__main__":
    import time
    t0 = time.time()

    data = np.load("data/processed/classification_data.npz", allow_pickle=True)
    X_raw = data["X"].astype(np.float64)
    y_raw = data["y"].astype(np.int64)
    dates_raw = data["dates"]
    factor_cols = data["factor_cols"].tolist()
    n_classes = len(np.unique(y_raw))
    input_dim = X_raw.shape[1]

    print(f"Raw dataset: {X_raw.shape}, classes={n_classes}")

    # Reshape to sequence
    LOOKBACK = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    lookback = LOOKBACK
    n_factors = input_dim // lookback
    X_seq = X_raw.reshape(-1, lookback, n_factors)

    # Remove bad values
    valid = ~(np.isnan(X_seq).any(axis=(1, 2)) | np.isinf(X_seq).any(axis=(1, 2)))
    print(f"Valid rows: {valid.sum()}/{len(valid)} (removed {(~valid).sum()} bad rows)")
    X = X_seq[valid]
    y = y_raw[valid]
    dates_arr = dates_raw[valid]

    # Time split before every fitted preprocessing step.
    dates = pd.to_datetime(dates_arr)
    train_mask = dates < "2018-01-01"
    val_mask = (dates >= "2018-01-01") & (dates < "2022-01-01")
    test_mask = dates >= "2022-01-01"

    # Clip using training-period thresholds only.
    for col in range(n_factors):
        train_flat = X[train_mask, :, col].reshape(-1)
        q1, q3 = np.percentile(train_flat, [1, 99])
        X[:, :, col] = np.clip(X[:, :, col], q1, q3)

    # Normalize using ONLY training set statistics (prevents data leakage)
    X_train = X[train_mask]
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    # Compute class weights (inverse frequency)
    y_train = y[train_mask]
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = len(y_train) / (n_classes * class_counts)
    class_weights = class_weights / class_weights.sum() * n_classes  # normalize
    print(f"\nClass distribution (train): {class_counts}")
    print(f"Class weights: {class_weights}")

    # Create datasets
    train_ds = TensorDataset(torch.tensor(X_norm[train_mask], dtype=torch.float32),
                              torch.tensor(y[train_mask], dtype=torch.int64))
    val_ds = TensorDataset(torch.tensor(X_norm[val_mask], dtype=torch.float32),
                            torch.tensor(y[val_mask], dtype=torch.int64))
    test_ds = TensorDataset(torch.tensor(X_norm[test_mask], dtype=torch.float32),
                             torch.tensor(y[test_mask], dtype=torch.int64))

    # Weighted sampler to oversample minority classes
    sample_weights = torch.zeros(len(train_ds))
    for c in range(n_classes):
        mask = y[train_mask] == c
        sample_weights[mask] = class_weights[c]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Input shape: ({lookback}, {n_factors}) per sample")

    # Train multiple models and compare
    torch.manual_seed(42)

    models = {
        "MLP": RegimeClassifier(input_dim=input_dim, n_classes=n_classes),
        "LSTM": RegimeLSTM(input_dim=n_factors, n_classes=n_classes, hidden_dim=128,
                            num_layers=2, dropout=0.3, bidirectional=False),
        "Transformer": RegimeTransformer(input_dim=n_factors, n_classes=n_classes, d_model=64,
                                         nhead=8, num_layers=3, dim_feedforward=256, dropout=0.2),
    }

    results = {}
    for name, mdl in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name} (params: {sum(p.numel() for p in mdl.parameters()):,})")
        print(f"{'='*50}")

        trainer = RegimeTrainer(
            mdl, n_classes=n_classes, lr=3e-4, weight_decay=1e-4,
            use_focal=True, class_weights=class_weights
        )
        best_val_acc = trainer.train(train_loader, val_loader, epochs=200, patience=50)

        test_loss, test_acc = trainer.evaluate(test_loader)
        probs, preds = predict_proba(mdl, test_loader)
        y_test = y[test_mask]
        cm = confusion_matrix(y_test, preds, n_classes)

        precisions, recalls, f1s = [], [], []
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        results[name] = {
            "test_acc": test_acc,
            "macro_f1": np.mean(f1s),
            "per_class_f1": f1s,
            "cm": cm,
            "model": mdl,
        }

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"COMPARISON (lookback={lookback}, seq_shape=({lookback}, {n_factors}))")
    print(f"{'='*80}")
    header = f"{'Model':<15} {'Acc':>6} {'Macro-F1':>8}"
    for i in range(n_classes):
        header += f" R{i}-F1".rjust(8)
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        row = f"{name:<15} {res['test_acc']:>6.3f} {res['macro_f1']:>8.3f}"
        for i in range(n_classes):
            row += f" {res['per_class_f1'][i]:>7.3f}"
        print(row)

    # Save all models
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    for name, res in results.items():
        torch.save({
            "model_name": name,
            "model_state": res["model"].state_dict(),
            "mean": mean.flatten().astype(np.float32),
            "std": std.flatten().astype(np.float32),
            "factor_cols": factor_cols,
            "n_classes": n_classes,
            "n_factors": n_factors,
            "lookback": lookback,
            "test_acc": res["test_acc"],
            "macro_f1": res["macro_f1"],
        }, f"data/processed/regime_model_{name}.pt")
        print(f"Saved: regime_model_{name}.pt (Acc={res['test_acc']:.3f}, Macro-F1={res['macro_f1']:.3f})")

    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} (Macro-F1={results[best_name]['macro_f1']:.3f})")

    # Save confusion matrix of best model
    print(f"\nBest model ({best_name}) confusion matrix:")
    cm = results[best_name]["cm"]
    for i in range(n_classes):
        row_str = f"  Regime {i}: "
        for j in range(n_classes):
            row_str += f"{cm[i,j]:5d} "
        print(row_str)

    print(f"\nTotal time: {time.time()-t0:.1f}s")
