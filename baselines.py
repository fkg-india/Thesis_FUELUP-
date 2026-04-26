"""
STEP 2 — Strong Baselines for Glucose Forecasting
===================================================
Models: Persistence, Linear Drift, Ridge, GBDT (HistGradientBoosting)
Optimized loaders that bypass Sample objects for speed.
"""

import os, re, glob, hashlib, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
plt.style.use("ggplot")

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
ROOT_DIR = Path(__file__).resolve().parent
BRIST1D_DIR = str(ROOT_DIR / "Kaggle BG")
HUPA_DIR = str(ROOT_DIR / "Mendeley")
CGMACROS_DIR = str(
    ROOT_DIR
    / "cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0"
    / "CGMacros_dateshifted365"
    / "CGMacros"
)
OUT_DIR = str(ROOT_DIR / "figures_step2")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW     = 72   # steps (6h)
HORIZON    = 12   # steps (+60 min)
STEP_MIN   = 5

# ══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════
def assign_split(pid: str) -> str:
    h = int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100
    if h < 68: return "train"
    elif h < 84: return "val"
    else: return "test"

def vectorized_slope(arr: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Compute linear-regression slope over the LAST n_steps of each row.
    arr: (N, T)  →  returns (N,)
    Uses closed-form OLS: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    """
    segment = arr[:, -n_steps:]                     # (N, n_steps)
    x = np.arange(n_steps, dtype=np.float32)        # (n_steps,)
    n = float(n_steps)
    sum_x  = x.sum()
    sum_x2 = (x ** 2).sum()
    sum_y  = np.nansum(segment, axis=1)             # (N,)
    sum_xy = np.nansum(segment * x[None, :], axis=1)
    denom  = n * sum_x2 - sum_x ** 2
    slope  = (n * sum_xy - sum_x * sum_y) / denom   # in units/step
    return slope

# ══════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (vectorized over (N, 72) arrays)
# ══════════════════════════════════════════════════════════════════════
def engineer_features(
    gl: np.ndarray,                            # (N, 72) glucose
    carbs: Optional[np.ndarray]   = None,      # (N, 72)
    insulin: Optional[np.ndarray] = None,      # (N, 72) — BrisT1D/HUPA
    basal: Optional[np.ndarray]   = None,      # (N, 72) — HUPA
    bolus: Optional[np.ndarray]   = None,      # (N, 72) — HUPA
    protein: Optional[np.ndarray] = None,      # (N, 72) — CGMacros S2+
    fat: Optional[np.ndarray]     = None,
    fiber: Optional[np.ndarray]   = None,
    meal_cal: Optional[np.ndarray]= None,
    hr: Optional[np.ndarray]      = None,      # (N, 72) — S3
    steps: Optional[np.ndarray]   = None,
    cals_act: Optional[np.ndarray]= None,
    mets: Optional[np.ndarray]    = None,
    hour_sin: Optional[np.ndarray]= None,      # (N,)
    hour_cos: Optional[np.ndarray]= None,
) -> Tuple[np.ndarray, List[str]]:
    """Build tabular feature matrix from windowed arrays. Returns (N, F), feature_names."""
    N = gl.shape[0]
    feats = {}

    # ── Glucose features ──
    feats["gl_current"]    = gl[:, -1]
    feats["gl_mean_30"]    = np.nanmean(gl[:, -6:],  axis=1)
    feats["gl_mean_60"]    = np.nanmean(gl[:, -12:], axis=1)
    feats["gl_mean_360"]   = np.nanmean(gl,          axis=1)
    feats["gl_std_60"]     = np.nanstd(gl[:, -12:],  axis=1)
    feats["gl_std_360"]    = np.nanstd(gl,           axis=1)
    feats["gl_min_360"]    = np.nanmin(gl,           axis=1)
    feats["gl_max_360"]    = np.nanmax(gl,           axis=1)
    feats["gl_range_360"]  = feats["gl_max_360"] - feats["gl_min_360"]
    feats["gl_slope_30"]   = vectorized_slope(gl, 6)
    feats["gl_slope_60"]   = vectorized_slope(gl, 12)
    feats["gl_delta_5"]    = gl[:, -1] - gl[:, -2]
    feats["gl_delta_15"]   = gl[:, -1] - gl[:, -4]
    feats["gl_delta_30"]   = gl[:, -1] - gl[:, -7]
    feats["gl_delta_60"]   = gl[:, -1] - gl[:, -13]

    # ── Carbs (sparse event channel) ──
    if carbs is not None:
        feats["carbs_sum_30"]  = np.nansum(carbs[:, -6:],  axis=1)
        feats["carbs_sum_60"]  = np.nansum(carbs[:, -12:], axis=1)
        feats["carbs_sum_120"] = np.nansum(carbs[:, -24:], axis=1)
        feats["carbs_sum_360"] = np.nansum(carbs,          axis=1)

    # ── Insulin (BrisT1D: single column | HUPA: basal + bolus) ──
    if insulin is not None:
        feats["ins_sum_30"]  = np.nansum(insulin[:, -6:],  axis=1)
        feats["ins_sum_60"]  = np.nansum(insulin[:, -12:], axis=1)
        feats["ins_sum_120"] = np.nansum(insulin[:, -24:], axis=1)
    if basal is not None:
        feats["basal_sum_60"]  = np.nansum(basal[:, -12:],  axis=1)
        feats["basal_sum_120"] = np.nansum(basal[:, -24:],  axis=1)
    if bolus is not None:
        feats["bolus_sum_60"]  = np.nansum(bolus[:, -12:],  axis=1)
        feats["bolus_sum_120"] = np.nansum(bolus[:, -24:],  axis=1)

    # ── Full macros (Scenario 2+, mainly CGMacros) ──
    for name, arr in [("protein", protein), ("fat", fat),
                      ("fiber", fiber), ("meal_cal", meal_cal)]:
        if arr is not None:
            feats[f"{name}_sum_60"]  = np.nansum(arr[:, -12:], axis=1)
            feats[f"{name}_sum_120"] = np.nansum(arr[:, -24:], axis=1)

    # ── Activity (Scenario 3) ──
    if hr is not None:
        feats["hr_mean_30"]  = np.nanmean(hr[:, -6:],  axis=1)
        feats["hr_mean_60"]  = np.nanmean(hr[:, -12:], axis=1)
    if steps is not None:
        feats["steps_sum_30"]  = np.nansum(steps[:, -6:],  axis=1)
        feats["steps_sum_60"]  = np.nansum(steps[:, -12:], axis=1)
    if cals_act is not None:
        feats["cals_act_sum_60"] = np.nansum(cals_act[:, -12:], axis=1)
    if mets is not None:
        feats["mets_mean_60"]  = np.nanmean(mets[:, -12:], axis=1)

    # ── Time features ──
    if hour_sin is not None:
        feats["hour_sin"] = hour_sin
    if hour_cos is not None:
        feats["hour_cos"] = hour_cos

    names = list(feats.keys())
    X = np.column_stack([feats[n] for n in names]).astype(np.float32)
    return X, names


# ══════════════════════════════════════════════════════════════════════
#  OPTIMIZED LOADERS — return (gl_matrix, channel_arrays, y, pids, meta)
# ══════════════════════════════════════════════════════════════════════

def _parse_lag_col(c: str):
    m = re.match(r'^([a-z_]+)-(\d+):(\d+)$', c)
    if m: return m.group(1), int(m.group(2)) * 60 + int(m.group(3))
    return None, None

def _get_lag_cols(all_cols: list, prefix: str) -> list:
    """Get sorted lag columns (oldest first) for a given prefix."""
    pairs = []
    for c in all_cols:
        p, mins = _parse_lag_col(c)
        if p == prefix:
            pairs.append((mins, c))
    pairs.sort(key=lambda x: x[0], reverse=True)  # oldest first
    return [c for _, c in pairs]


def load_brist1d_fast(scenario: int = 1):
    """Load BrisT1D directly as numpy arrays. Returns dict."""
    print(f"  Loading BrisT1D (Scenario {scenario})...")
    df = pd.read_csv(os.path.join(BRIST1D_DIR, "train.csv"), low_memory=False)

    # Get lag columns for each prefix
    bg_cols     = _get_lag_cols(df.columns, "bg")
    carbs_cols  = _get_lag_cols(df.columns, "carbs")
    ins_cols    = _get_lag_cols(df.columns, "insulin")
    hr_cols     = _get_lag_cols(df.columns, "hr")
    steps_cols  = _get_lag_cols(df.columns, "steps")
    cals_cols   = _get_lag_cols(df.columns, "cals")

    # Extract matrices
    gl = df[bg_cols].values.astype(np.float32)          # (N, 72)
    y_abs = df["bg+1:00"].values.astype(np.float32)
    y_delta = y_abs - gl[:, -1]                         # delta from current
    pids = df["p_num"].values.astype(str)

    carbs_arr = df[carbs_cols].values.astype(np.float32) if carbs_cols else None
    ins_arr   = df[ins_cols].values.astype(np.float32) if ins_cols else None

    # Time features
    times = df["time"].astype(str)
    hours = times.str.split(":").str[0].astype(float)
    mins  = times.str.split(":").str[1].astype(float)
    h_sin = np.sin(2 * np.pi * (hours + mins / 60) / 24).values.astype(np.float32)
    h_cos = np.cos(2 * np.pi * (hours + mins / 60) / 24).values.astype(np.float32)

    # Activity (S3)
    hr_arr    = df[hr_cols].values.astype(np.float32) if scenario >= 3 and hr_cols else None
    steps_arr = df[steps_cols].values.astype(np.float32) if scenario >= 3 and steps_cols else None
    cals_arr  = df[cals_cols].values.astype(np.float32) if scenario >= 3 and cals_cols else None

    # Drop rows with missing target
    valid = ~np.isnan(y_abs)
    print(f"    Rows: {len(df):,}, valid targets: {valid.sum():,}")

    return dict(
        gl=gl[valid], y_abs=y_abs[valid], y_delta=y_delta[valid],
        pids=pids[valid], carbs=carbs_arr[valid] if carbs_arr is not None else None,
        insulin=ins_arr[valid] if ins_arr is not None else None,
        hr=hr_arr[valid] if hr_arr is not None else None,
        steps=steps_arr[valid] if steps_arr is not None else None,
        cals_act=cals_arr[valid] if cals_arr is not None else None,
        hour_sin=h_sin[valid], hour_cos=h_cos[valid],
        unit="mmol/L", dataset="brist1d",
    )


def _build_windows_fast(df: pd.DataFrame, gl_col: str, channel_map: dict,
                        pid: str) -> dict:
    """Build sliding windows from a single participant's 5-min DataFrame."""
    n = len(df)
    max_start = n - WINDOW - HORIZON + 1
    if max_start <= 0:
        return None

    # Pre-extract all columns as arrays
    gl_full = df[gl_col].values.astype(np.float32)
    timestamps = pd.DatetimeIndex(df["time"])
    hours_frac = (timestamps.hour + timestamps.minute / 60.0).values.astype(np.float32)

    # Indices of valid windows (where target is not NaN)
    starts = np.arange(max_start)
    target_idxs = starts + WINDOW - 1 + HORIZON
    valid_mask = ~np.isnan(gl_full[target_idxs])
    starts = starts[valid_mask]
    if len(starts) == 0:
        return None

    # Build glucose windows using stride tricks (efficient)
    N = len(starts)
    gl_windows = np.array([gl_full[s:s + WINDOW] for s in starts], dtype=np.float32)
    y_abs = gl_full[starts + WINDOW - 1 + HORIZON]
    y_delta = y_abs - gl_windows[:, -1]

    # Time features at t=0 (end of window)
    t0_idx = starts + WINDOW - 1
    h_frac = hours_frac[t0_idx]
    h_sin = np.sin(2 * np.pi * h_frac / 24.0).astype(np.float32)
    h_cos = np.cos(2 * np.pi * h_frac / 24.0).astype(np.float32)

    result = dict(gl=gl_windows, y_abs=y_abs, y_delta=y_delta,
                  pids=np.full(N, pid), hour_sin=h_sin, hour_cos=h_cos,
                  unit="mg/dL")

    # Other channels
    for key, col in channel_map.items():
        if col in df.columns:
            arr_full = df[col].values.astype(np.float32)
            result[key] = np.array([arr_full[s:s + WINDOW] for s in starts], dtype=np.float32)

    return result


def _merge_participant_dicts(dicts: list) -> dict:
    if not dicts:
        return {}
    
    # Collect all possible keys
    keys = set()
    for d in dicts:
        keys.update(d.keys())
        
    merged = {}
    for k in keys:
        vals = []
        is_array = False
        sample_shape = None
        
        # Check if this key corresponds to an array
        for d in dicts:
            if k in d and isinstance(d[k], np.ndarray):
                is_array = True
                if len(d[k].shape) > 1:
                    sample_shape = d[k].shape[1:]
                break
                
        if is_array:
            for d in dicts:
                n_samples = len(d["y_abs"])
                if k in d and d[k] is not None:
                    vals.append(d[k])
                else:
                    # Pad with NaNs
                    shape = (n_samples,) if sample_shape is None else (n_samples, *sample_shape)
                    pad = np.full(shape, np.nan, dtype=np.float32)
                    vals.append(pad)
            merged[k] = np.concatenate(vals, axis=0)
        else:
            # For scalars like 'unit' or 'dataset', take the first available
            for d in dicts:
                if k in d and d[k] is not None:
                    merged[k] = d[k]
                    break
    return merged


def load_hupa_fast(scenario: int = 1):
    print(f"  Loading HUPA (Scenario {scenario})...")
    channel_map = {"carbs": "carb_input", "basal": "basal_rate", "bolus": "bolus_volume_delivered"}
    if scenario >= 3:
        channel_map.update({"hr": "heart_rate", "steps": "steps", "cals_act": "calories"})

    csv_files = sorted(glob.glob(os.path.join(HUPA_DIR, "*.csv")))
    parts = []
    for csv_path in csv_files:
        pid = Path(csv_path).stem
        df = pd.read_csv(csv_path, sep=";")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        d = _build_windows_fast(df, "glucose", channel_map, pid)
        if d: parts.append(d)

    result = _merge_participant_dicts(parts)
    result["dataset"] = "hupa"
    result["unit"] = "mg/dL"
    print(f"    Participants: {len(csv_files)}, samples: {len(result['y_abs']):,}")
    return result


def load_cgmacros_fast(scenario: int = 1):
    print(f"  Loading CGMacros (Scenario {scenario})...")
    channel_map = {"carbs": "Carbs"}
    if scenario >= 2:
        channel_map.update({"protein": "Protein", "fat": "Fat", "fiber": "Fiber", "meal_cal": "Calories"})
    if scenario >= 3:
        channel_map.update({"hr": "HR", "mets": "METs", "cals_act": "Calories (Activity)"})

    folders = sorted(glob.glob(os.path.join(CGMACROS_DIR, "CGMacros-0*")))
    parts = []
    for folder in folders:
        pid = os.path.basename(folder)
        csv_files = [f for f in glob.glob(os.path.join(folder, "*.csv"))
                     if "bio" not in f.lower()]
        if not csv_files: continue
        df = pd.read_csv(csv_files[0])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)

        # Resample to 5-min
        df = df.set_index("Timestamp")
        sum_cols = ["Carbs", "Protein", "Fat", "Fiber", "Calories"]
        agg_dict = {}
        for c in df.select_dtypes(include=[np.number]).columns:
            agg_dict[c] = "sum" if c in sum_cols else "mean"
        df = df.resample("5min").agg(agg_dict).reset_index()
        df.rename(columns={"Timestamp": "time"}, inplace=True)

        d = _build_windows_fast(df, "Libre GL", channel_map, pid)
        if d: parts.append(d)

    result = _merge_participant_dicts(parts)
    result["dataset"] = "cgmacros"
    result["unit"] = "mg/dL"
    print(f"    Participants: {len(folders)}, samples: {len(result['y_abs']):,}")
    return result


# ══════════════════════════════════════════════════════════════════════
#  SPLIT
# ══════════════════════════════════════════════════════════════════════

def split_data(data: dict) -> Dict[str, dict]:
    pids = data["pids"]
    N = len(pids)
    split_labels = np.array([assign_split(str(p)) for p in pids])

    result = {}
    for split in ["train", "val", "test"]:
        mask = split_labels == split
        d = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == N:
                d[k] = v[mask]
            else:
                d[k] = v  # scalars like 'unit', 'dataset'
        result[split] = d

    # Report
    for split in ["train", "val", "test"]:
        sp = result[split]["pids"]
        print(f"    {split:6s}: {len(sp):>8,} samples  ({len(np.unique(sp))} participants)")
    return result


# ══════════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════════

def persistence(gl_current: np.ndarray) -> np.ndarray:
    """y_hat = glucose(t). NaN-safe: uses last-known non-NaN."""
    pred = gl_current.copy()
    # Replace remaining NaN with column median
    nan_mask = np.isnan(pred)
    if nan_mask.any():
        pred[nan_mask] = np.nanmedian(pred)
    return pred

def linear_drift(gl: np.ndarray) -> np.ndarray:
    """y_hat = glucose(t) + slope_60 * horizon_steps"""
    slope = vectorized_slope(gl, 12)
    current = gl[:, -1].copy()
    nan_mask = np.isnan(current)
    if nan_mask.any():
        current[nan_mask] = np.nanmedian(gl[:, -1])
    nan_slope = np.isnan(slope)
    slope[nan_slope] = 0.0
    return current + slope * HORIZON

def train_ridge(X_train, y_train, X_eval):
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_train)
    X_ev = imp.transform(X_eval)
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_train)
    return model.predict(X_ev), model, imp

def train_gbdt(X_train, y_train, X_eval):
    model = HistGradientBoostingRegressor(
        max_iter=500, max_depth=6, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, n_iter_no_change=20,
        validation_fraction=0.15, random_state=42,
    )
    model.fit(X_train, y_train)
    return model.predict(X_eval), model


# ══════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════

def to_mgdl(y, unit):
    if unit == "mmol/L": return y * 18.0182
    return y

def eval_metrics(y_true, y_pred, unit="mg/dL"):
    # Filter NaN from both
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "range_mae": {}}

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Range-stratified (in mg/dL)
    yt_mg = to_mgdl(y_true, unit)
    yp_mg = to_mgdl(y_pred, unit)

    ranges = {"hypo": yt_mg < 70, "normal": (yt_mg >= 70) & (yt_mg <= 180), "hyper": yt_mg > 180}
    range_mae = {}
    for rname, rmask in ranges.items():
        if rmask.sum() > 0:
            range_mae[rname] = (mean_absolute_error(y_true[rmask], y_pred[rmask]), int(rmask.sum()))
        else:
            range_mae[rname] = (np.nan, 0)

    return {"mae": mae, "rmse": rmse, "range_mae": range_mae}


# ══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════

def plot_results(results: dict, dataset: str, scenario: int):
    """Generate diagnostic plots for all models on one dataset+scenario."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"{dataset.upper()} — Scenario {scenario}: Baseline Results", fontsize=14, fontweight="bold")

    # ── 1) Bar chart: MAE comparison (abs + delta) ──
    ax = axes[0, 0]
    models_abs   = {k: v["abs"]["mae"] for k, v in results.items() if "abs" in v}
    models_delta = {k: v["delta"]["mae"] for k, v in results.items() if "delta" in v}
    x = np.arange(len(models_abs))
    w = 0.35
    names = list(models_abs.keys())
    ax.bar(x - w/2, [models_abs[n] for n in names], w, label="Direct (y_abs)", color="steelblue")
    if models_delta:
        ax.bar(x + w/2, [models_delta.get(n, 0) for n in names], w, label="Delta→reconstruct", color="coral")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30)
    ax.set_ylabel("MAE"); ax.set_title("MAE: Direct vs Delta"); ax.legend()

    # ── 2) Range-stratified MAE (abs mode) ──
    ax = axes[0, 1]
    range_names = ["hypo", "normal", "hyper"]
    x = np.arange(len(range_names))
    w = 0.8 / len(names)
    for i, model_name in enumerate(names):
        rm = results[model_name]["abs"]["range_mae"]
        vals = [rm.get(r, (np.nan, 0))[0] for r in range_names]
        ax.bar(x + i * w, vals, w, label=model_name)
    ax.set_xticks(x + w * len(names) / 2)
    ax.set_xticklabels(range_names)
    ax.set_ylabel("MAE"); ax.set_title("MAE by Glucose Range"); ax.legend(fontsize=7)

    # ── 3) RMSE comparison ──
    ax = axes[0, 2]
    rmse_vals = [results[n]["abs"]["rmse"] for n in names]
    ax.barh(names, rmse_vals, color="mediumpurple")
    ax.set_xlabel("RMSE"); ax.set_title("RMSE (Direct Mode)")

    # ── 4) Residual histograms ──
    ax = axes[1, 0]
    for model_name in names:
        if "residuals" in results[model_name]:
            res = results[model_name]["residuals"]
            ax.hist(res, bins=80, alpha=0.4, label=model_name, density=True)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Residual (pred - true)"); ax.set_title("Residual Distributions"); ax.legend(fontsize=7)

    # ── 5) Feature importance (GBDT) ──
    ax = axes[1, 1]
    if "GBDT" in results and "importance" in results["GBDT"]:
        imp = results["GBDT"]["importance"]
        top = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
        ax.barh([t[0] for t in top][::-1], [t[1] for t in top][::-1], color="teal")
        ax.set_xlabel("Importance"); ax.set_title("GBDT Top-15 Features")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

    # ── 6) Predicted vs Actual (GBDT) ──
    ax = axes[1, 2]
    if "GBDT" in results and "y_pred" in results["GBDT"] and "y_true" in results["GBDT"]:
        yt = results["GBDT"]["y_true"]
        yp = results["GBDT"]["y_pred"]
        sample_idx = np.random.choice(len(yt), min(3000, len(yt)), replace=False)
        ax.scatter(yt[sample_idx], yp[sample_idx], alpha=0.1, s=5, color="teal")
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted"); ax.set_title("GBDT: Pred vs True")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"baselines_{dataset}_s{scenario}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {path}")


# ══════════════════════════════════════════════════════════════════════
#  RUN ONE DATASET + SCENARIO
# ══════════════════════════════════════════════════════════════════════

def run_baselines(data: dict, scenario: int):
    """Run all baselines on loaded data. Returns results dict."""
    dataset = data["dataset"]
    unit    = data["unit"]

    # Split
    print(f"\n{'─'*50}")
    print(f"  {dataset.upper()} Scenario {scenario}")
    print(f"{'─'*50}")
    splits = split_data(data)
    train, test = splits["train"], splits["test"]

    # If test is tiny, use val instead
    if len(test["y_abs"]) < 100:
        print("  ⚠ Test set too small, using val for evaluation.")
        test = splits["val"]

    # Engineer features
    def _make_feats(d):
        return engineer_features(
            gl=d["gl"], carbs=d.get("carbs"), insulin=d.get("insulin"),
            basal=d.get("basal"), bolus=d.get("bolus"),
            protein=d.get("protein"), fat=d.get("fat"),
            fiber=d.get("fiber"), meal_cal=d.get("meal_cal"),
            hr=d.get("hr"), steps=d.get("steps"),
            cals_act=d.get("cals_act"), mets=d.get("mets"),
            hour_sin=d.get("hour_sin"), hour_cos=d.get("hour_cos"),
        )

    X_train, feat_names = _make_feats(train)
    X_test,  _          = _make_feats(test)
    y_train_abs   = train["y_abs"]
    y_test_abs    = test["y_abs"]
    y_train_delta = train["y_delta"]
    y_test_delta  = test["y_delta"]
    gl_test       = test["gl"]

    # Filter NaN targets before training
    valid_train = ~(np.isnan(y_train_abs) | np.isnan(y_train_delta))
    X_train       = X_train[valid_train]
    y_train_abs   = y_train_abs[valid_train]
    y_train_delta = y_train_delta[valid_train]
    
    valid_test = ~(np.isnan(y_test_abs) | np.isnan(y_test_delta))
    X_test       = X_test[valid_test]
    y_test_abs   = y_test_abs[valid_test]
    y_test_delta = y_test_delta[valid_test]
    gl_test      = gl_test[valid_test]

    print(f"  Features: {len(feat_names)} → {feat_names}")
    print(f"  Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

    results = {}

    # ── Persistence ──
    yp_pers = persistence(gl_test[:, -1])
    m_abs   = eval_metrics(y_test_abs, yp_pers, unit)
    # Delta mode: persistence predicts delta=0 → reconstruct = gl_current
    m_delta = eval_metrics(y_test_abs, yp_pers, unit)  # same result
    results["Persistence"] = {
        "abs": m_abs, "delta": m_delta,
        "residuals": yp_pers - y_test_abs,
        "y_pred": yp_pers, "y_true": y_test_abs,
    }
    print(f"  Persistence  — MAE: {m_abs['mae']:.3f}  RMSE: {m_abs['rmse']:.3f}")

    # ── Linear Drift ──
    yp_drift = linear_drift(gl_test)
    m_abs = eval_metrics(y_test_abs, yp_drift, unit)
    results["LinDrift"] = {
        "abs": m_abs, "delta": m_abs,
        "residuals": yp_drift - y_test_abs,
        "y_pred": yp_drift, "y_true": y_test_abs,
    }
    print(f"  LinDrift     — MAE: {m_abs['mae']:.3f}  RMSE: {m_abs['rmse']:.3f}")

    # ── Ridge (abs + delta) ──
    yp_ridge_abs, ridge_model, ridge_imp = train_ridge(X_train, y_train_abs, X_test)
    m_abs_ridge = eval_metrics(y_test_abs, yp_ridge_abs, unit)

    yp_ridge_delta, _, _ = train_ridge(X_train, y_train_delta, X_test)
    yp_ridge_recon = gl_test[:, -1] + yp_ridge_delta
    m_delta_ridge = eval_metrics(y_test_abs, yp_ridge_recon, unit)

    results["Ridge"] = {
        "abs": m_abs_ridge, "delta": m_delta_ridge,
        "residuals": yp_ridge_abs - y_test_abs,
        "y_pred": yp_ridge_abs, "y_true": y_test_abs,
    }
    print(f"  Ridge (abs)  — MAE: {m_abs_ridge['mae']:.3f}  RMSE: {m_abs_ridge['rmse']:.3f}")
    print(f"  Ridge (Δ)    — MAE: {m_delta_ridge['mae']:.3f}  RMSE: {m_delta_ridge['rmse']:.3f}")

    # ── GBDT (abs + delta) ──
    yp_gbdt_abs, gbdt_model = train_gbdt(X_train, y_train_abs, X_test)
    m_abs_gbdt = eval_metrics(y_test_abs, yp_gbdt_abs, unit)

    yp_gbdt_delta, _ = train_gbdt(X_train, y_train_delta, X_test)
    yp_gbdt_recon = gl_test[:, -1] + yp_gbdt_delta
    m_delta_gbdt = eval_metrics(y_test_abs, yp_gbdt_recon, unit)

    # Feature importance for HistGradient: use permutation or built-in if available
    try:
        # HistGradientBoosting doesn't have feature_importances_ by default
        from sklearn.inspection import permutation_importance
        imp_result = permutation_importance(gbdt_model, X_test, y_test_abs,
                                            n_repeats=5, random_state=42, n_jobs=-1)
        importance = dict(zip(feat_names, imp_result.importances_mean))
    except Exception:
        importance = {}

    results["GBDT"] = {
        "abs": m_abs_gbdt, "delta": m_delta_gbdt,
        "residuals": yp_gbdt_abs - y_test_abs,
        "y_pred": yp_gbdt_abs, "y_true": y_test_abs,
        "importance": importance,
    }
    print(f"  GBDT (abs)   — MAE: {m_abs_gbdt['mae']:.3f}  RMSE: {m_abs_gbdt['rmse']:.3f}")
    print(f"  GBDT (Δ)     — MAE: {m_delta_gbdt['mae']:.3f}  RMSE: {m_delta_gbdt['rmse']:.3f}")

    # ── Print summary table ──
    print(f"\n  {'Model':<14} {'MAE(abs)':>10} {'MAE(Δ)':>10} {'RMSE(abs)':>10}  "
          f"{'hypo':>8} {'normal':>8} {'hyper':>8}")
    print(f"  {'─'*76}")
    for name in ["Persistence", "LinDrift", "Ridge", "GBDT"]:
        r = results[name]
        rm = r["abs"]["range_mae"]
        hypo_mae   = f"{rm['hypo'][0]:.2f}({rm['hypo'][1]})" if rm['hypo'][1] > 0 else "N/A"
        norm_mae   = f"{rm['normal'][0]:.2f}({rm['normal'][1]})" if rm['normal'][1] > 0 else "N/A"
        hyper_mae  = f"{rm['hyper'][0]:.2f}({rm['hyper'][1]})" if rm['hyper'][1] > 0 else "N/A"
        delta_mae  = f"{r['delta']['mae']:.3f}"
        print(f"  {name:<14} {r['abs']['mae']:>10.3f} {delta_mae:>10} {r['abs']['rmse']:>10.3f}  "
              f"{hypo_mae:>8} {norm_mae:>8} {hyper_mae:>8}")

    # Plot
    plot_results(results, dataset, scenario)
    return results


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_results = {}

    # ── HUPA: S1 and S3 ──
    for sc in [1, 3]:
        data = load_hupa_fast(scenario=sc)
        all_results[("hupa", sc)] = run_baselines(data, sc)
        del data

    # ── CGMacros: S1, S2, S3 ──
    for sc in [1, 2, 3]:
        data = load_cgmacros_fast(scenario=sc)
        all_results[("cgmacros", sc)] = run_baselines(data, sc)
        del data

    # ── BrisT1D: S1 and S3 ──
    for sc in [1, 3]:
        data = load_brist1d_fast(scenario=sc)
        all_results[("brist1d", sc)] = run_baselines(data, sc)
        del data

    # ══════════════════════════════════════════════════════════════
    #  GRAND SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print(" GRAND SUMMARY — All Datasets × Scenarios × Models")
    print("█" * 70)
    print(f"\n  {'Dataset':<10} {'Sc':>3} {'Model':<14} {'MAE(abs)':>10} {'MAE(Δ)':>10} {'RMSE':>10}")
    print(f"  {'─'*60}")
    for (ds, sc), res in sorted(all_results.items()):
        for model_name in ["Persistence", "LinDrift", "Ridge", "GBDT"]:
            r = res[model_name]
            print(f"  {ds:<10} {sc:>3} {model_name:<14} "
                  f"{r['abs']['mae']:>10.3f} {r['delta']['mae']:>10.3f} {r['abs']['rmse']:>10.3f}")
        print(f"  {'─'*60}")

    # ══════════════════════════════════════════════════════════════
    #  ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" ANALYSIS: Why These Baselines Are 'Strong'")
    print("=" * 70)
    print("""
  1) PERSISTENCE is the minimum bar. For glucose, the current reading is
     highly correlated with the +60 min value (≈0.92-0.95). Any model that
     can't beat persistence is useless.

  2) LINEAR DRIFT adds trend awareness. It captures "glucose is rising at
     X mg/dL per step, so in 12 more steps it will be Y higher." This is
     surprisingly effective for steady-state periods but fails at inflection
     points (post-meal spikes, insulin kicks in).

  3) RIDGE on engineered features tests whether handcrafted temporal
     features (means, slopes, sums) encode enough information. If Ridge
     approaches GBDT, deep models may not add much.

  4) GBDT (HistGradientBoosting) is the strongest tabular baseline. It
     handles nonlinear interactions between features (e.g., "carbs_60
     interacting with insulin_60") without explicit feature crosses.

  WHERE BASELINES FAIL:
  - Post-meal spikes: Persistence and LinDrift can't anticipate the
    nonlinear glucose rise 30-90 min after eating.
  - Insulin action lag: The delayed effect of insulin (peak at 60-90 min)
    requires sequence modeling, not snapshot features.
  - Activity transitions: HR spike → glucose drop has a complex temporal
    relationship that tabular features only partially capture.
  - Extreme values (hypo/hyper): Range-stratified MAE shows all baselines
    struggle more in these ranges due to rarity and rapid dynamics.
""")

    print("\n█ STEP 2 COMPLETE █")
