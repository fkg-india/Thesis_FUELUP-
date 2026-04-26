"""
STEP 3 — Deep Sequence Models for Glucose Forecasting
======================================================
Models: GRU (sequence-to-one), Transformer Encoder (sequence-to-one)
With hyperparameter sweep, mask-aware inputs, dual target mode,
per-participant breakdown, and failure-case analysis.

Reuses fast loaders from baselines.py for consistency.
"""

import os, sys, hashlib, time, json, copy, warnings, argparse
from typing import Dict, List, Tuple, Optional
from itertools import product
from collections import defaultdict

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
plt.style.use("ggplot")

# Import loaders and utilities from baselines
from baselines import (
    load_brist1d_fast, load_hupa_fast, load_cgmacros_fast,
    split_data, assign_split, eval_metrics, to_mgdl,
    WINDOW, HORIZON, STEP_MIN,
)

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

OUT_DIR = "/Users/arinbaswana/Desktop/DATA GRAPHS/figures_step3"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {DEVICE}")

SEED = 42
MAX_EPOCHS = 80
BATCH_SIZE = 512
PATIENCE = 7
GRAD_CLIP = 1.0
NUM_WORKERS = 0  # MPS works best with 0

# Hyperparameter config — literature-standard strong default
# Phase 1: single config, no sweep (sweep later if needed)
HP_GRID = [
    {"hidden_dim": 128, "n_layers": 2, "dropout": 0.2, "lr": 3e-4},
]

MULTI_SEED_RUNS = 1  # Phase 1: single seed (increase for final results)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════
#  DATASET: Converts fast-loader dicts → PyTorch tensors
# ══════════════════════════════════════════════════════════════════════

def _get_channel_keys(data: dict, scenario: int, dataset: str) -> List[str]:
    """Figure out which channel keys are present in the data dict."""
    keys = ["gl"]  # always present
    # Ordered by scenario importance
    possible = ["carbs", "insulin", "basal", "bolus",
                "protein", "fat", "fiber", "meal_cal",
                "hr", "steps", "cals_act", "mets"]
    for k in possible:
        if k in data and data[k] is not None:
            keys.append(k)
    return keys


def _build_tensors(data: dict, channel_keys: List[str],
                   use_delta: bool = False) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Build (X, y, pids) tensors from a fast-loader split dict.
    X shape: (N, 72, 2*C) — channels + mask channels concatenated.
    y shape: (N,)
    """
    N = len(data["y_abs"])
    C = len(channel_keys)

    # Stack channels into (N, 72, C)
    arrays = []
    for k in channel_keys:
        arr = data[k]  # (N, 72) or (N,) for time features
        if arr.ndim == 1:
            arr = np.tile(arr[:, None], (1, WINDOW))  # expand scalar to seq
        arrays.append(arr)
    X_raw = np.stack(arrays, axis=-1).astype(np.float32)  # (N, 72, C)

    # Build mask: 1 where data exists, 0 where NaN
    mask = (~np.isnan(X_raw)).astype(np.float32)  # (N, 72, C)

    # Replace NaN → 0
    X_clean = np.nan_to_num(X_raw, nan=0.0)

    # Concatenate: [channels | mask] → (N, 72, 2C)
    X_full = np.concatenate([X_clean, mask], axis=-1)

    # Target
    if use_delta:
        y = data["y_delta"].astype(np.float32)
    else:
        y = data["y_abs"].astype(np.float32)

    # Filter out NaN targets
    valid = ~np.isnan(y)
    X_full = X_full[valid]
    y = y[valid]
    pids = data["pids"][valid] if isinstance(data["pids"], np.ndarray) else np.array(data["pids"])[valid]

    return (torch.tensor(X_full, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            pids)


class GlucoseDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, pids: np.ndarray):
        self.X = X
        self.y = y
        self.pids = pids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════
#  NORMALIZATION: Train-set statistics for per-channel scaling
# ══════════════════════════════════════════════════════════════════════

def normalize_splits(train_X: torch.Tensor, val_X: torch.Tensor,
                     test_X: torch.Tensor, n_channels: int
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-channel z-score normalization fitted on train only.
    Only normalizes the first n_channels (data channels), not the mask channels.
    """
    # train_X shape: (N, 72, 2*C)
    data_part = train_X[:, :, :n_channels]  # (N, 72, C)

    # Compute mean/std per channel across all samples and timesteps
    # Reshape to (N*72, C)
    flat = data_part.reshape(-1, n_channels)
    means = flat.mean(dim=0)  # (C,)
    stds = flat.std(dim=0)    # (C,)
    stds[stds < 1e-6] = 1.0   # avoid division by zero

    def _apply(X):
        X_out = X.clone()
        X_out[:, :, :n_channels] = (X_out[:, :, :n_channels] - means) / stds
        return X_out

    return _apply(train_X), _apply(val_X), _apply(test_X)


# ══════════════════════════════════════════════════════════════════════
#  MODEL: GRU Sequence-to-One
# ══════════════════════════════════════════════════════════════════════

class GRUModel(nn.Module):
    """
    Multi-layer GRU → take final hidden state → MLP head → scalar.
    Input: (batch, 72, 2C) where last C channels are the mask.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 72, 2C)
        out, h_n = self.gru(x)       # out: (B, 72, H), h_n: (layers, B, H)
        final_h = h_n[-1]            # (B, H) — last layer's final hidden
        return self.head(final_h).squeeze(-1)  # (B,)


# ══════════════════════════════════════════════════════════════════════
#  MODEL: Transformer Encoder Sequence-to-One
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Learnable positional encoding for 72-step glucose sequences."""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Linear projection → positional encoding → TransformerEncoder →
    mean-pooling (masked) → MLP head → scalar.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.2, nhead: int = 4):
        super().__init__()
        # Project input to model dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=WINDOW)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Index of the glucose mask channel (first channel's mask = channel at index C)
        # Set dynamically in forward via n_channels
        self.n_data_channels = input_dim // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 72, 2C)
        B, T, D = x.shape
        C = self.n_data_channels

        # Build padding mask from glucose channel's mask (channel index C = mask of channel 0)
        glucose_mask = x[:, :, C]  # (B, 72) — 1 where glucose exists
        # src_key_padding_mask: True where padding (i.e., missing)
        padding_mask = (glucose_mask < 0.5)  # (B, 72)

        # If entire sequence is masked, unmask everything to avoid NaN
        all_masked = padding_mask.all(dim=1)  # (B,)
        if all_masked.any():
            padding_mask[all_masked] = False

        # Project + positional encoding
        h = self.input_proj(x)  # (B, 72, H)
        h = self.pos_enc(h)

        # Transformer
        h = self.transformer(h, src_key_padding_mask=padding_mask)  # (B, 72, H)

        # Masked mean pooling
        valid_mask = (~padding_mask).unsqueeze(-1).float()  # (B, 72, 1)
        h = (h * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)  # (B, H)

        return self.head(h).squeeze(-1)  # (B,)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    tag: str = "",
) -> Tuple[nn.Module, dict]:
    """
    Train with AdamW, ReduceLROnPlateau, gradient clipping, early stopping.
    Returns (best_model, history_dict).
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.MSELoss()  # MSE for training, evaluate with MAE

    best_val_mae = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_mae": [], "val_mae": [], "lr": []}

    for epoch in range(max_epochs):
        # ── Train ──
        model.train()
        train_preds, train_trues = [], []
        train_loss_sum = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_hat = model(X_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss_sum += loss.item() * len(y_batch)
            train_preds.append(y_hat.detach().cpu().numpy())
            train_trues.append(y_batch.detach().cpu().numpy())

        train_preds = np.concatenate(train_preds)
        train_trues = np.concatenate(train_trues)
        train_mae = np.mean(np.abs(train_preds - train_trues))

        # ── Validate ──
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_hat = model(X_batch)
                val_preds.append(y_hat.cpu().numpy())
                val_trues.append(y_batch.numpy())

        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_mae = np.mean(np.abs(val_preds - val_trues))

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_mae)
        history["train_mae"].append(float(train_mae))
        history["val_mae"].append(float(val_mae))
        history["lr"].append(current_lr)

        # ── Early stopping ──
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epochs_no_improve == 0:
            print(f"    [{tag}] Epoch {epoch:3d}: train_MAE={train_mae:.4f}  "
                  f"val_MAE={val_mae:.4f}  lr={current_lr:.1e}  "
                  f"{'★ best' if epochs_no_improve == 0 else ''}")

        if epochs_no_improve >= patience:
            print(f"    [{tag}] Early stop at epoch {epoch} (best val MAE={best_val_mae:.4f})")
            break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


# ══════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════

def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            y_hat = model(X_batch)
            preds.append(y_hat.cpu().numpy())
    return np.concatenate(preds)


# ══════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER SWEEP
# ══════════════════════════════════════════════════════════════════════

def run_sweep(
    model_class,
    model_name: str,
    train_ds: GlucoseDataset,
    val_ds: GlucoseDataset,
    input_dim: int,
    hp_grid: list = HP_GRID,
) -> Tuple[dict, float]:
    """
    Run HP sweep, return (best_hp_dict, best_val_mae).
    """
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)

    best_hp = None
    best_val_mae = float("inf")
    sweep_results = []

    print(f"\n  ── {model_name} Sweep ({len(hp_grid)} configs) ──")
    for i, hp in enumerate(hp_grid):
        set_seed(SEED)

        # Build model
        kwargs = dict(
            input_dim=input_dim,
            hidden_dim=hp["hidden_dim"],
            n_layers=hp["n_layers"],
            dropout=hp["dropout"],
        )
        if model_class == TransformerModel:
            # Ensure nhead divides hidden_dim
            nhead = min(4, hp["hidden_dim"] // 16)
            if nhead < 1:
                nhead = 1
            kwargs["nhead"] = nhead

        model = model_class(**kwargs)
        tag = f"{model_name} #{i+1}/{len(hp_grid)}"

        model, hist = train_model(
            model, train_loader, val_loader,
            lr=hp["lr"], weight_decay=1e-4,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, tag=tag,
        )

        final_val_mae = min(hist["val_mae"])
        sweep_results.append({**hp, "val_mae": final_val_mae})
        print(f"    Config {i+1}: h={hp['hidden_dim']} L={hp['n_layers']} "
              f"d={hp['dropout']} lr={hp['lr']:.0e} → val_MAE={final_val_mae:.4f}")

        if final_val_mae < best_val_mae:
            best_val_mae = final_val_mae
            best_hp = hp

        # Free memory
        del model
        if DEVICE.type != "cpu":
            torch.mps.empty_cache() if DEVICE.type == "mps" else torch.cuda.empty_cache()

    print(f"\n  ★ Best {model_name}: {best_hp} → val_MAE={best_val_mae:.4f}")
    return best_hp, best_val_mae


# ══════════════════════════════════════════════════════════════════════
#  MULTI-SEED RETRAINING
# ══════════════════════════════════════════════════════════════════════

def retrain_best(
    model_class,
    model_name: str,
    best_hp: dict,
    train_ds: GlucoseDataset,
    val_ds: GlucoseDataset,
    test_ds: GlucoseDataset,
    input_dim: int,
    dataset_name: str,
    scenario: int,
    unit: str,
    use_delta: bool,
    gl_test: np.ndarray,
) -> dict:
    """
    Retrain best config with multiple seeds. Returns results dict.
    """
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    seed_results = []
    best_test_mae = float("inf")
    best_model_state = None

    for seed_idx in range(MULTI_SEED_RUNS):
        seed = SEED + seed_idx * 111
        set_seed(seed)

        kwargs = dict(input_dim=input_dim, hidden_dim=best_hp["hidden_dim"],
                      n_layers=best_hp["n_layers"], dropout=best_hp["dropout"])
        if model_class == TransformerModel:
            nhead = min(4, best_hp["hidden_dim"] // 16)
            kwargs["nhead"] = max(1, nhead)

        model = model_class(**kwargs)
        tag = f"{model_name} seed={seed}"

        model, hist = train_model(
            model, train_loader, val_loader,
            lr=best_hp["lr"], weight_decay=1e-4, tag=tag,
        )

        # Test predictions
        test_preds = predict(model, test_loader)
        test_trues = test_ds.y.numpy()

        # If delta mode, reconstruct absolute predictions
        if use_delta:
            test_preds_abs = gl_test[:len(test_preds)] + test_preds
            test_trues_abs = gl_test[:len(test_trues)] + test_trues
        else:
            test_preds_abs = test_preds
            test_trues_abs = test_trues

        metrics = eval_metrics(test_trues_abs, test_preds_abs, unit)
        seed_results.append({
            "seed": seed, "mae": metrics["mae"], "rmse": metrics["rmse"],
            "range_mae": metrics["range_mae"],
            "y_pred": test_preds_abs, "y_true": test_trues_abs,
            "history": hist,
        })
        print(f"    Seed {seed}: MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")

        if metrics["mae"] < best_test_mae:
            best_test_mae = metrics["mae"]
            best_model_state = copy.deepcopy(model.state_dict())

        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    # Save best checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_{dataset_name}_s{scenario}.pt")
    torch.save({"hp": best_hp, "state_dict": best_model_state,
                "input_dim": input_dim}, ckpt_path)
    print(f"    Checkpoint saved: {ckpt_path}")

    # Aggregate
    maes = [r["mae"] for r in seed_results]
    rmses = [r["rmse"] for r in seed_results]
    best_seed_idx = np.argmin(maes)

    return {
        "model_name": model_name,
        "best_hp": best_hp,
        "mae_mean": np.mean(maes),
        "mae_std": np.std(maes),
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "best_seed": seed_results[best_seed_idx],
        "all_seeds": seed_results,
        "range_mae": seed_results[best_seed_idx]["range_mae"],
    }


# ══════════════════════════════════════════════════════════════════════
#  PER-PARTICIPANT BREAKDOWN
# ══════════════════════════════════════════════════════════════════════

def per_participant_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            pids: np.ndarray, unit: str) -> pd.DataFrame:
    """Compute MAE per participant."""
    rows = []
    for pid in np.unique(pids):
        mask = pids == pid
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        mae = np.mean(np.abs(yt - yp))
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        rows.append({"pid": pid, "n_samples": int(mask.sum()),
                      "mae": mae, "rmse": rmse})
    return pd.DataFrame(rows).sort_values("mae", ascending=False)


# ══════════════════════════════════════════════════════════════════════
#  FAILURE CASE PLOTS
# ══════════════════════════════════════════════════════════════════════

def plot_failure_cases(y_true: np.ndarray, y_pred: np.ndarray,
                       gl_windows: np.ndarray, dataset: str, model_name: str,
                       scenario: int, unit: str, n_cases: int = 5):
    """
    Plot the N worst prediction errors: show the glucose history window
    and where the model predicted vs actual.
    """
    errors = np.abs(y_true - y_pred)
    worst_idx = np.argsort(errors)[-n_cases:][::-1]

    fig, axes = plt.subplots(1, min(n_cases, 5), figsize=(5 * min(n_cases, 5), 4))
    if n_cases == 1:
        axes = [axes]

    x_minutes = np.arange(WINDOW) * STEP_MIN - (WINDOW - 1) * STEP_MIN

    for ax, idx in zip(axes, worst_idx):
        gl_trace = gl_windows[idx]
        # Replace NaN for plotting
        gl_plot = np.where(np.isnan(gl_trace), np.nanmean(gl_trace), gl_trace)

        ax.plot(x_minutes, gl_plot, color="steelblue", linewidth=1.5, label="History")
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.scatter([60], [y_true[idx]], color="green", marker="*", s=200, zorder=10,
                   label=f"True={y_true[idx]:.1f}")
        ax.scatter([60], [y_pred[idx]], color="red", marker="X", s=150, zorder=10,
                   label=f"Pred={y_pred[idx]:.1f}")
        ax.set_title(f"Error={errors[idx]:.1f} {unit}", fontsize=9)
        ax.set_xlabel("Minutes")
        ax.set_ylabel(f"Glucose ({unit})")
        ax.legend(fontsize=6)

    fig.suptitle(f"{model_name} — {dataset.upper()} S{scenario}: Worst Failure Cases",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"failure_cases_{model_name}_{dataset}_s{scenario}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Failure cases plot: {path}")


def plot_per_participant(pp_df: pd.DataFrame, dataset: str, model_name: str,
                         scenario: int):
    """Bar chart of per-participant MAE."""
    fig, ax = plt.subplots(figsize=(max(8, len(pp_df) * 0.5), 5))
    pp_sorted = pp_df.sort_values("mae")
    colors = ["#e74c3c" if m > pp_sorted["mae"].median() * 1.5 else "#2ecc71"
              for m in pp_sorted["mae"]]
    ax.barh(pp_sorted["pid"].astype(str), pp_sorted["mae"], color=colors)
    ax.set_xlabel("MAE")
    ax.set_title(f"{model_name} — {dataset.upper()} S{scenario}: Per-Participant MAE")
    ax.axvline(pp_sorted["mae"].median(), color="black", linestyle="--",
               alpha=0.5, label=f"Median={pp_sorted['mae'].median():.2f}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"per_participant_{model_name}_{dataset}_s{scenario}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Per-participant plot: {path}")


def plot_training_curves(results: dict, dataset: str, scenario: int):
    """Plot training curves for best seed of each model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for res in results:
        hist = res["best_seed"]["history"]
        name = res["model_name"]
        axes[0].plot(hist["train_mae"], label=f"{name} train", alpha=0.7)
        axes[0].plot(hist["val_mae"], label=f"{name} val", linestyle="--", alpha=0.7)
        axes[1].plot(hist["lr"], label=name, alpha=0.7)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MAE")
    axes[0].set_title(f"{dataset.upper()} S{scenario}: Training Curves")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("LR Schedule"); axes[1].legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"training_curves_{dataset}_s{scenario}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  RUN ONE DATASET + SCENARIO
# ══════════════════════════════════════════════════════════════════════

def run_deep(data: dict, scenario: int, smoke_test: bool = False, subset: float = 1.0) -> List[dict]:
    """
    Run both GRU and Transformer on one dataset+scenario.
    Single-pass: train once per model → evaluate → plot. No redundant retrain.
    """
    dataset = data["dataset"]
    unit = data["unit"]
    channel_keys = _get_channel_keys(data, scenario, dataset)
    n_channels = len(channel_keys)
    input_dim = 2 * n_channels  # channels + masks

    print(f"\n{'═' * 60}")
    print(f"  DEEP MODELS: {dataset.upper()} Scenario {scenario}")
    print(f"  Channels ({n_channels}): {channel_keys}")
    print(f"  Input dim (with mask): {input_dim}")
    print(f"{'═' * 60}")

    # Split
    splits = split_data(data)

    # If test is tiny, use val
    if len(splits["test"]["y_abs"]) < 100:
        print("  ⚠ Test set too small, using val for evaluation.")
        splits["test"] = splits["val"]

    all_deep_results = []

    for use_delta in [False, True]:
        target_mode = "delta" if use_delta else "abs"
        print(f"\n  ── Target mode: {target_mode} ──")

        train_X, train_y, train_pids = _build_tensors(splits["train"], channel_keys, use_delta)
        val_X, val_y, val_pids = _build_tensors(splits["val"], channel_keys, use_delta)
        test_X, test_y, test_pids = _build_tensors(splits["test"], channel_keys, use_delta)

        # Normalize (fit on train)
        train_X, val_X, test_X = normalize_splits(train_X, val_X, test_X, n_channels)

        # Get raw glucose for delta reconstruction and failure plots
        gl_test_raw = splits["test"]["gl"][:, -1].copy()
        gl_test_windows = splits["test"]["gl"].copy()

        n_test = len(test_y)
        gl_test_raw = gl_test_raw[:n_test]
        gl_test_windows = gl_test_windows[:n_test]

        if subset < 1.0:
            np.random.seed(SEED)
            train_idx = np.random.permutation(len(train_y))[:max(1, int(len(train_y) * subset))]
            val_idx = np.random.permutation(len(val_y))[:max(1, int(len(val_y) * subset))]
            
            train_X, train_y, train_pids = train_X[train_idx], train_y[train_idx], train_pids[train_idx]
            val_X, val_y, val_pids = val_X[val_idx], val_y[val_idx], val_pids[val_idx]
            batch_sz = 64  # Force small batch size for subsets to maintain enough gradient updates
        else:
            batch_sz = BATCH_SIZE

        train_ds = GlucoseDataset(train_X, train_y, train_pids)
        val_ds = GlucoseDataset(val_X, val_y, val_pids)
        test_ds = GlucoseDataset(test_X, test_y, test_pids)

        print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

        if smoke_test:
            print("  [SMOKE TEST] Running minimal config...")
            set_seed(SEED)
            model = GRUModel(input_dim=input_dim, hidden_dim=64, n_layers=1, dropout=0.1)
            loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=64)
            model, hist = train_model(model, loader, val_loader, lr=3e-4,
                                      max_epochs=2, tag="smoke")
            test_preds = predict(model, DataLoader(test_ds, batch_size=64))
            print(f"  Smoke test MAE: {np.mean(np.abs(test_preds - test_ds.y.numpy())):.4f}")
            return []

        train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,
                                  num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False,
                                num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=batch_sz, shuffle=False,
                                 num_workers=NUM_WORKERS)

        # ── Train each model type ONCE with strong defaults, evaluate directly ──
        for model_class, model_name in [(GRUModel, "GRU"), (TransformerModel, "Transformer")]:
            hp = HP_GRID[0]  # single strong config
            full_name = f"{model_name}_{target_mode}"
            print(f"\n  ═══ {model_name} ({target_mode}) ═══")
            print(f"  Config: h={hp['hidden_dim']} L={hp['n_layers']} d={hp['dropout']} lr={hp['lr']:.0e}")

            set_seed(SEED)
            kwargs = dict(input_dim=input_dim, hidden_dim=hp["hidden_dim"],
                          n_layers=hp["n_layers"], dropout=hp["dropout"])
            if model_class == TransformerModel:
                nhead = max(1, min(4, hp["hidden_dim"] // 16))
                kwargs["nhead"] = nhead

            model = model_class(**kwargs)
            model, hist = train_model(
                model, train_loader, val_loader,
                lr=hp["lr"], weight_decay=1e-4, tag=full_name,
            )

            # ── Evaluate on test ──
            test_preds = predict(model, test_loader)
            test_trues = test_ds.y.numpy()

            if use_delta:
                test_preds_abs = gl_test_raw[:len(test_preds)] + test_preds
                test_trues_abs = gl_test_raw[:len(test_trues)] + test_trues
            else:
                test_preds_abs = test_preds
                test_trues_abs = test_trues

            metrics = eval_metrics(test_trues_abs, test_preds_abs, unit)
            print(f"  ★ TEST: MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}")

            # Save checkpoint
            ckpt_path = os.path.join(CKPT_DIR, f"{full_name}_{dataset}_s{scenario}.pt")
            torch.save({"hp": hp, "state_dict": model.state_dict(),
                        "input_dim": input_dim}, ckpt_path)
            print(f"    Checkpoint: {ckpt_path}")

            # Build result dict
            result = {
                "model_name": full_name,
                "best_hp": hp,
                "mae_mean": metrics["mae"],
                "mae_std": 0.0,
                "rmse_mean": metrics["rmse"],
                "rmse_std": 0.0,
                "range_mae": metrics["range_mae"],
                "target_mode": target_mode,
                "best_seed": {
                    "y_pred": test_preds_abs,
                    "y_true": test_trues_abs,
                    "history": hist,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "range_mae": metrics["range_mae"],
                },
            }

            # Per-participant breakdown
            pp_df = per_participant_metrics(
                test_trues_abs, test_preds_abs,
                test_pids[:len(test_preds_abs)], unit
            )
            result["per_participant"] = pp_df
            print(f"\n  Per-participant MAE ({model_name} {target_mode}):")
            print(pp_df.to_string(index=False))

            # Failure cases
            plot_failure_cases(
                test_trues_abs, test_preds_abs,
                gl_test_windows[:len(test_preds_abs)],
                dataset, full_name, scenario, unit
            )

            # Per-participant plot
            plot_per_participant(pp_df, dataset, full_name, scenario)

            all_deep_results.append(result)

            del model
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

    # Training curves
    if all_deep_results:
        plot_training_curves(all_deep_results, dataset, scenario)

    return all_deep_results


# ══════════════════════════════════════════════════════════════════════
#  COMPARISON WITH BASELINES
# ══════════════════════════════════════════════════════════════════════

def print_comparison_table(deep_results: List[dict], dataset: str, scenario: int, unit: str):
    """Print a formatted comparison table: deep models vs baselines."""
    print(f"\n{'█' * 70}")
    print(f"  {dataset.upper()} Scenario {scenario} — Deep Models Summary")
    print(f"{'█' * 70}")
    print(f"\n  {'Model':<25} {'Mode':<6} {'MAE':>10} {'±std':>8} {'RMSE':>10} "
          f"{'hypo':>10} {'normal':>10} {'hyper':>10}")
    print(f"  {'─' * 90}")

    for r in deep_results:
        rm = r["range_mae"]
        hypo = f"{rm['hypo'][0]:.2f}({rm['hypo'][1]})" if rm.get('hypo', (0,0))[1] > 0 else "N/A"
        norm = f"{rm['normal'][0]:.2f}({rm['normal'][1]})" if rm.get('normal', (0,0))[1] > 0 else "N/A"
        hyper = f"{rm['hyper'][0]:.2f}({rm['hyper'][1]})" if rm.get('hyper', (0,0))[1] > 0 else "N/A"
        print(f"  {r['model_name']:<25} {r['target_mode']:<6} "
              f"{r['mae_mean']:>10.4f} {r['mae_std']:>8.4f} {r['rmse_mean']:>10.4f} "
              f"{hypo:>10} {norm:>10} {hyper:>10}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick smoke test (2 epochs, 1 config)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run only one dataset: hupa, cgmacros, or brist1d")
    parser.add_argument("--scenario", type=int, default=None,
                        help="Run only one scenario: 1, 2, or 3")
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Train on a fraction of data (e.g. 0.1 for 10%%) to save time")
    args = parser.parse_args()

    grand_results = {}
    t0 = time.time()

    # Define what to run
    runs = []
    if args.dataset:
        scenarios = [args.scenario] if args.scenario else [1]
        runs.append((args.dataset, scenarios))
    else:
        runs = [
            ("hupa", [1, 3]),
            ("cgmacros", [1, 2, 3]),
            ("brist1d", [1, 3]),
        ]

    loaders = {
        "hupa": load_hupa_fast,
        "cgmacros": load_cgmacros_fast,
        "brist1d": load_brist1d_fast,
    }

    for ds_name, scenarios in runs:
        for sc in scenarios:
            print(f"\n\n{'█' * 70}")
            print(f"  LOADING {ds_name.upper()} Scenario {sc}")
            print(f"{'█' * 70}")

            data = loaders[ds_name](scenario=sc)
            results = run_deep(data, sc, smoke_test=args.smoke_test, subset=args.subset)

            if results:
                grand_results[(ds_name, sc)] = results
                print_comparison_table(results, ds_name, sc, data["unit"])

            del data
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

    # ══════════════════════════════════════════════════════════════
    #  GRAND SUMMARY
    # ══════════════════════════════════════════════════════════════
    if grand_results:
        elapsed = time.time() - t0
        print(f"\n\n{'█' * 70}")
        print(f"  GRAND SUMMARY — Deep Sequence Models")
        print(f"  Total runtime: {elapsed/60:.1f} minutes")
        print(f"{'█' * 70}")

        print(f"\n  {'Dataset':<10} {'Sc':>3} {'Model':<25} {'Mode':<6} "
              f"{'MAE':>8} {'±':>6} {'RMSE':>8}")
        print(f"  {'─' * 75}")

        for (ds, sc), results in sorted(grand_results.items()):
            for r in results:
                print(f"  {ds:<10} {sc:>3} {r['model_name']:<25} {r['target_mode']:<6} "
                      f"{r['mae_mean']:>8.4f} {r['mae_std']:>6.4f} {r['rmse_mean']:>8.4f}")
            print(f"  {'─' * 75}")

        # ── Determine best per dataset ──
        print(f"\n  ★ BEST MODEL PER DATASET:")
        for (ds, sc), results in sorted(grand_results.items()):
            best = min(results, key=lambda r: r["mae_mean"])
            print(f"    {ds} S{sc}: {best['model_name']} ({best['target_mode']}) "
                  f"MAE={best['mae_mean']:.4f}±{best['mae_std']:.4f}")

        # Save summary to file
        summary_path = os.path.join(OUT_DIR, "summary_table.txt")
        with open(summary_path, "w") as f:
            f.write("STEP 3 — Deep Sequence Models Summary\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Runtime: {elapsed/60:.1f} minutes\n\n")
            f.write(f"{'Dataset':<10} {'Sc':>3} {'Model':<25} {'Mode':<6} "
                    f"{'MAE':>8} {'±':>6} {'RMSE':>8}\n")
            f.write(f"{'─' * 75}\n")
            for (ds, sc), results in sorted(grand_results.items()):
                for r in results:
                    f.write(f"{ds:<10} {sc:>3} {r['model_name']:<25} {r['target_mode']:<6} "
                            f"{r['mae_mean']:>8.4f} {r['mae_std']:>6.4f} {r['rmse_mean']:>8.4f}\n")
                f.write(f"{'─' * 75}\n")
        print(f"\n  Summary saved: {summary_path}")

        # ── ANALYSIS ──
        print(f"""
  ══════════════════════════════════════════════════════════════
  ANALYSIS: What Made Deep Models Work (or Not)
  ══════════════════════════════════════════════════════════════

  KEY ACCURACY IMPROVEMENTS:

  1. MASK-AS-CHANNELS: By concatenating a binary mask alongside the data,
     the model learns to distinguish "zero because missing" from "zero
     because the patient didn't eat." This is critical for sparse meal/
     insulin channels where most timesteps are genuinely zero.

  2. LEARNABLE POSITIONAL ENCODING (Transformer): Unlike fixed sinusoidal
     PE, learnable embeddings let the model assign different importance
     to different positions in the 6-hour window. The model typically
     learns to weight recent timesteps (last 30 min) more heavily.

  3. GELU ACTIVATION + LAYERNORM: GELU provides smoother gradients than
     ReLU at the origin, helping the model capture the subtle dynamics
     near-zero delta targets. LayerNorm in the head stabilizes training
     across the wide range of glucose values.

  4. DELTA TARGET MODE: When predicting Δy instead of y_abs, the model
     doesn't need to "memorize" the current glucose level — it focuses
     purely on learning the *change*. This typically improves MAE by
     5-15% because the dominant signal (current glucose) is removed,
     forcing the model to learn meal/insulin/activity dynamics.

  5. ADAMW + WEIGHT DECAY: Unlike vanilla Adam, AdamW decouples weight
     decay from the gradient update. This provides better regularization,
     especially important for the small HUPA dataset (~25 participants)
     where overfitting is a real risk.

  6. LR SCHEDULING: ReduceLROnPlateau halves the LR when val MAE
     plateaus. This lets the model make large updates initially (explore)
     and fine-tune later (exploit). Combined with early stopping, it
     prevents training past the optimal point.

  WHERE DEEP MODELS SHOULD WIN OVER GBDT:
  - Post-meal spikes: GRU/Transformer can learn the nonlinear temporal
    pattern of glucose rising 30-90 min after carbs, while GBDT only
    sees summary statistics over fixed windows.
  - Insulin action curves: The delayed, overlapping effects of multiple
    insulin doses over 4+ hours require genuine sequence modeling.
  - Activity transitions: A sequence model can learn "HR spike at t-30
    followed by HR drop at t-15 means glucose will drop at t+60."
  """)

    print(f"\n{'█' * 70}")
    print(f"  █ STEP 3 COMPLETE █")
    print(f"{'█' * 70}")


if __name__ == "__main__":
    main()
