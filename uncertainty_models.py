import os
import copy
import time
import argparse
from typing import Tuple, List

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import baseline + deep model resources
from baselines import load_hupa_fast, load_cgmacros_fast, load_brist1d_fast, split_data
import deep_models
from deep_models import (
    DEVICE, SEED, set_seed, GlucoseDataset, _build_tensors, normalize_splits,
    TransformerModel, _get_channel_keys
)

WINDOW = 72

# ══════════════════════════════════════════════════════════════════════
#  LOSS & ARCHITECTURE: Quantile Regression
# ══════════════════════════════════════════════════════════════════════

def pinball_loss(preds: torch.Tensor, targets: torch.Tensor, quantiles=[0.1, 0.5, 0.9]) -> torch.Tensor:
    """
    Computes pinball (quantile) loss.
    preds: (Batch, 3) 
    targets: (Batch,)
    """
    loss = 0.0
    for i, q in enumerate(quantiles):
        error = targets - preds[:, i]
        loss += torch.max(q * error, (q - 1) * error).mean()
    return loss / len(quantiles)

class QuantileTransformerModel(TransformerModel):
    """Overrides pure Transformer head to output 3 quantiles instead of scalar."""
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.2, nhead: int = 4):
        super().__init__(input_dim, hidden_dim, n_layers, dropout, nhead)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 3), # q10, q50, q90
        )

# ══════════════════════════════════════════════════════════════════════
#  TRAINING: Quantile Model
# ══════════════════════════════════════════════════════════════════════

def train_quantile_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                         lr: float = 3e-4, max_epochs: int = 40, patience: int = 7):
    """Training loop specifically using pinball loss."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = pinball_loss(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = pinball_loss(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        lr_curr = optimizer.param_groups[0]['lr']
        
        tag_best = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            tag_best = " ★ best"
        else:
            epochs_no_improve += 1
            
        print(f"    [Quantile] Epoch {epoch:>3}: train_Pinball={train_loss:.4f}  "
              f"val_Pinball={val_loss:.4f}  lr={lr_curr:.1e}{tag_best}")
        
        if epochs_no_improve >= patience:
            print(f"    [Quantile] Early stop at epoch {epoch} (best val loss={best_val_loss:.4f})")
            break

    model.load_state_dict(best_weights)
    return model

def predict_quantile(model: nn.Module, loader: DataLoader):
    """Returns q10, q50, q90."""
    model.eval()
    q10_list, q50_list, q90_list = [], [], []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds = model(X_batch.to(DEVICE)).cpu().numpy()
            q10_list.append(preds[:, 0])
            q50_list.append(preds[:, 1])
            q90_list.append(preds[:, 2])
    return np.concatenate(q10_list), np.concatenate(q50_list), np.concatenate(q90_list)

# ══════════════════════════════════════════════════════════════════════
#  INFERENCE: MC-Dropout
# ══════════════════════════════════════════════════════════════════════

def predict_mcdropout(model: nn.Module, loader: DataLoader, K: int = 30):
    """
    Keep dropout active (.train()) and run K forward passes.
    Returns q10, q50, q90 bounds based on the dropout variance.
    """
    model.train() # Critical: this keeps dropout ON
    all_preds_K = []
    
    with torch.no_grad():
        for _ in range(K):
            preds_curr = []
            for X_batch, _ in loader:
                p = model(X_batch.to(DEVICE)).cpu().numpy()
                preds_curr.append(p)
            all_preds_K.append(np.concatenate(preds_curr))
            
    # all_preds_K is shape (K, N)
    all_stacked = np.array(all_preds_K)
    q10 = np.percentile(all_stacked, 10, axis=0)
    q50 = np.mean(all_stacked, axis=0)  # Use mean as deterministic midpoint
    q90 = np.percentile(all_stacked, 90, axis=0)
    
    return q10, q50, q90

# ══════════════════════════════════════════════════════════════════════
#  EVALUATION & PLOTTING
# ══════════════════════════════════════════════════════════════════════

def plot_uncertainty_case(y_hist, y_true_target, q10, q50, q90, save_path, title):
    """
    y_hist: (72,) historical glucose raw
    y_true_target: scalar
    """
    plt.figure(figsize=(8, 4))
    t_hist = np.arange(-WINDOW, 0) * 5
    plt.plot(t_hist, y_hist, label="History", color="dodgerblue", marker='.')
    
    t_pred = 60 # 60 min ahead
    plt.scatter([t_pred], [y_true_target], color="green", marker='*', s=150, zorder=5, label="True Future")
    
    # Plot bounds
    plt.scatter([t_pred], [q50], color="red", marker='X', s=100, zorder=4, label="Prediction (q50/mean)")
    plt.plot([t_pred, t_pred], [q10, q90], color='red', linestyle='-', lw=2, alpha=0.5, label="80% Interval")
    
    # Shade interval roughly 
    plt.axhspan(q10, q90, color='red', alpha=0.1)
    
    plt.title(title)
    plt.xlabel("Time (minutes, 0 = now)")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def eval_uncertainty(name: str, y_true: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray,
                     gl_raw: np.ndarray, gl_windows: np.ndarray, ds_name: str, sc: int):
    # Bruteforce cleanup of any tensor remnants
    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().detach().numpy()
    y_true = np.array(y_true)
    q10 = np.array(q10)
    q50 = np.array(q50)
    q90 = np.array(q90)
        
    mae = np.mean(np.abs(y_true - q50))
    rmse = np.sqrt(np.mean((y_true - q50)**2))
    
    coverage = np.mean((y_true >= q10) & (y_true <= q90)) * 100.0
    width = np.mean(q90 - q10)
    
    print(f"\n  [{name}] MAE: {mae:.2f}  RMSE: {rmse:.2f}")
    print(f"  [{name}] 80% CI Coverage: {coverage:.1f}%")
    print(f"  [{name}] Mean Interval Width: {width:.1f} mg/dL")
    
    # Plot top 3 biggest mistakes where target was heavily missed
    errors = np.abs(y_true - q50)
    worst_idx = np.argsort(errors)[::-1][:3]
    
    for i, idx in enumerate(worst_idx):
        path = f"figures_step4/failure_uncert_{name}_{ds_name}_s{sc}_err{i}.png"
        title = f"{name} Worst #{i+1} | MAE={errors[idx]:.1f} | Bounds: [{q10[idx]:.1f}, {q90[idx]:.1f}]"
        plot_uncertainty_case(gl_windows[idx, :], y_true[idx], 
                              q10[idx], q50[idx], q90[idx], path, title)
        
    return {"mae": mae, "rmse": rmse, "coverage": coverage, "width": width}

# ══════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

def run_uncertainty(data, ds_name, sc, subset=0.1):
    splits = split_data(data)
    channel_keys = _get_channel_keys(data, sc, ds_name)
    n_channels = len(channel_keys)
    
    train_X, train_y, train_pids = _build_tensors(splits["train"], channel_keys, use_delta=False)
    val_X, val_y, val_pids = _build_tensors(splits["val"], channel_keys, use_delta=False)
    if "test" not in splits:
        splits["test"] = splits["val"]
    test_X, test_y, test_pids = _build_tensors(splits["test"], channel_keys, use_delta=False)
    
    train_X, val_X, test_X = normalize_splits(train_X, val_X, test_X, n_channels)
    
    gl_test_raw = splits["test"]["gl"][:, -1].copy()
    gl_test_windows = splits["test"]["gl"].copy()
    
    n_test = len(test_y)
    gl_test_raw = gl_test_raw[:n_test]
    gl_test_windows = gl_test_windows[:n_test]
    
    if subset < 1.0:
        np.random.seed(SEED)
        train_idx = np.random.permutation(len(train_y))[:max(1, int(len(train_y) * subset))]
        val_idx = np.random.permutation(len(val_y))[:max(1, int(len(val_y) * subset))]
        test_idx = np.random.permutation(len(test_y))[:max(1, int(len(test_y) * subset))]
        train_X, train_y = train_X[train_idx], train_y[train_idx]
        val_X, val_y = val_X[val_idx], val_y[val_idx]
        test_X, test_y = test_X[test_idx], test_y[test_idx]
        gl_test_raw = gl_test_raw[test_idx]
        gl_test_windows = gl_test_windows[test_idx]
        batch_sz = 64
    else:
        batch_sz = 512
        
    train_ds = GlucoseDataset(train_X, train_y, ["p"]*len(train_y))
    val_ds = GlucoseDataset(val_X, val_y, ["p"]*len(val_y))
    test_ds = GlucoseDataset(test_X, test_y, ["p"]*len(test_y))
    
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    
    input_dim = train_X.shape[-1]
    
    print(f"\n{'='*60}")
    print(f"  Training Standard Transformer (for MC-Dropout base)")
    print(f"{'='*60}")
    set_seed(SEED)
    base_model = TransformerModel(input_dim=input_dim, hidden_dim=128, n_layers=2, dropout=0.2)
    # Using deep_models native train_model
    base_model, _ = deep_models.train_model(
        base_model, train_loader, val_loader, lr=3e-4, max_epochs=40, patience=7, tag="Standard"
    )
    
    print(f"\n  Running MC-Dropout Inference...")
    t0 = time.time()
    mc_q10, mc_q50, mc_q90 = predict_mcdropout(base_model, test_loader, K=30)
    print(f"  MC-Dropout completed in {time.time()-t0:.1f}s")
    
    mc_res = eval_uncertainty("MC_Dropout", test_y, mc_q10, mc_q50, mc_q90, 
                              gl_test_raw, gl_test_windows, ds_name, sc)
    
    print(f"\n{'='*60}")
    print(f"  Training Quantile Transformer (Pinball)")
    print(f"{'='*60}")
    set_seed(SEED)
    quant_model = QuantileTransformerModel(input_dim=input_dim, hidden_dim=128, n_layers=2, dropout=0.2)
    quant_model = train_quantile_model(quant_model, train_loader, val_loader, lr=3e-4, max_epochs=60, patience=7)
    
    print(f"\n  Running Quantile Inference...")
    t0 = time.time()
    q_q10, q_q50, q_q90 = predict_quantile(quant_model, test_loader)
    print(f"  Quantile inference completed in {time.time()-t0:.1f}s")
    
    q_res = eval_uncertainty("Quantile", test_y, q_q10, q_q50, q_q90, 
                             gl_test_raw, gl_test_windows, ds_name, sc)
                             
    return {"MC": mc_res, "Quantile": q_res}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=float, default=0.1)
    args = parser.parse_args()
    
    SUBSET = args.subset
    os.makedirs("figures_step4", exist_ok=True)
    
    # Run a fast sweep on HUPA Scenario 1 and BrisT1D Scenario 3 
    # to compare both approaches on 10% data
    print(f"Running Uncertanity Sweep (subset={SUBSET})...")
    
    data_hupa = load_hupa_fast(scenario=1)
    res_hupa = run_uncertainty(data_hupa, "hupa", 1, subset=SUBSET)
    
    data_bris = load_brist1d_fast(scenario=3)
    res_bris = run_uncertainty(data_bris, "brist1d", 3, subset=SUBSET)
    
    print(f"\n\n{'█'*60}")
    print(f"  UNCERTAINTY SUMMARY")
    print(f"{'█'*60}")
    print(f"\n  Dataset       Method       MAE     RMSE    Cov%   Width")
    print(f"  {'-'*60}")
    for ds, sc, res in [("HUPA S1", 1, res_hupa), ("BrisT1d S3", 3, res_bris)]:
        mc = res["MC"]
        qu = res["Quantile"]
        print(f"  {ds:<13} MC-Drop      {mc['mae']:<7.2f} {mc['rmse']:<7.2f} "
              f"{mc['coverage']:<6.1f} {mc['width']:<6.1f}")
        print(f"  {ds:<13} Quantile     {qu['mae']:<7.2f} {qu['rmse']:<7.2f} "
              f"{qu['coverage']:<6.1f} {qu['width']:<6.1f}")
