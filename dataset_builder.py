"""
STEP 1 — Unified Dataset Builder for Glucose Forecasting Thesis
================================================================
Produces standardised (X, y, mask, meta) samples from BrisT1D, HUPA-UCM,
and CGMacros, supporting 3 ablation scenarios and 2 target modes.

Author : Thesis ML pipeline
Version: 1.0
"""

from __future__ import annotations

import hashlib
import os
import re
import glob
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

WINDOW_STEPS = 72          # 6 hours at 5-min resolution
HORIZON_STEPS = 12         # +60 min at 5-min resolution
STEP_MINUTES = 5
MAX_FFILL_GAP = 6          # forward-fill glucose for gaps ≤ 30 min (6 × 5 min)

# Paths — adjust as needed
BRIST1D_DIR = "/Users/arinbaswana/Desktop/DATA GRAPHS/Kaggle BG"
HUPA_DIR = "/Users/arinbaswana/Desktop/DATA GRAPHS/Mendeley"
CGMACROS_DIR = "/Users/arinbaswana/Desktop/DATA GRAPHS/cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0/CGMacros_dateshifted365/CGMacros"
OUT_DIR = "/Users/arinbaswana/Desktop/DATA GRAPHS/figures_step1"
os.makedirs(OUT_DIR, exist_ok=True)

# Scenario → channel names (unified naming, mapped per-dataset)
SCENARIO_CHANNELS: Dict[int, List[str]] = {
    1: ["glucose", "carbs"],
    2: ["glucose", "carbs", "protein", "fat", "fiber", "meal_cal", "consumed_pct"],
    3: ["glucose", "carbs", "protein", "fat", "fiber", "meal_cal", "consumed_pct",
        "hr", "steps", "cals_activity", "mets"],
}

# Channels that get robust scaling (median/IQR) — heavy-tailed / sparse
ROBUST_SCALE_CHANNELS = {"carbs", "protein", "fat", "fiber", "meal_cal",
                         "consumed_pct", "bolus", "mets"}
# Channels that get standard scaling (mean/std)
STANDARD_SCALE_CHANNELS = {"glucose", "hr", "steps", "cals_activity",
                           "basal", "insulin"}

# For T1D datasets, insulin channels augment Scenario 1
T1D_INSULIN_CHANNELS = {
    "brist1d": ["insulin"],
    "hupa": ["basal", "bolus"],
}


# ══════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Sample:
    """One supervised sample."""
    X: np.ndarray          # (72, C) float32 — may contain NaN
    y_abs: float           # absolute glucose at t+60
    y_delta: float         # glucose(t+60) - glucose(t)
    mask: np.ndarray       # (72, C) bool — True where data exists
    participant_id: str
    dataset_name: str
    timestamp: str         # ISO string of time t
    scenario: int
    channel_names: List[str]
    glucose_unit: str      # "mmol/L" or "mg/dL"


@dataclass
class DatasetSamples:
    """Collection of samples from one dataset + scenario."""
    samples: List[Sample] = field(default_factory=list)
    dataset_name: str = ""
    scenario: int = 1
    channel_names: List[str] = field(default_factory=list)

    @property
    def X(self) -> np.ndarray:
        return np.stack([s.X for s in self.samples])

    @property
    def y_abs(self) -> np.ndarray:
        return np.array([s.y_abs for s in self.samples], dtype=np.float32)

    @property
    def y_delta(self) -> np.ndarray:
        return np.array([s.y_delta for s in self.samples], dtype=np.float32)

    @property
    def mask(self) -> np.ndarray:
        return np.stack([s.mask for s in self.samples])

    @property
    def pids(self) -> List[str]:
        return [s.participant_id for s in self.samples]


# ══════════════════════════════════════════════════════════════════════
#  SPLIT ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════

def assign_split(pid: str, train_frac: float = 0.68, val_frac: float = 0.16) -> str:
    """Deterministic, reproducible participant-level split via hash."""
    h = int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100
    if h < int(train_frac * 100):
        return "train"
    elif h < int((train_frac + val_frac) * 100):
        return "val"
    else:
        return "test"


# ══════════════════════════════════════════════════════════════════════
#  TIME FEATURES
# ══════════════════════════════════════════════════════════════════════

def make_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Create cyclic time features for a sequence of timestamps.
    Returns (N, 4): hour_sin, hour_cos, minute_sin, minute_cos
    """
    hours = timestamps.hour + timestamps.minute / 60.0
    minutes = timestamps.minute + timestamps.second / 60.0
    feats = np.column_stack([
        np.sin(2 * np.pi * hours / 24.0),
        np.cos(2 * np.pi * hours / 24.0),
        np.sin(2 * np.pi * minutes / 60.0),
        np.cos(2 * np.pi * minutes / 60.0),
    ]).astype(np.float32)
    return feats


def make_time_features_from_single(hour: float, minute: float, n_steps: int = 72) -> np.ndarray:
    """
    For BrisT1D where we only have a single time-of-day.
    Replicate across 72 steps, adjusting for each 5-min step.
    """
    # Step 0 = t-355 min … step 71 = t-0 min
    base_minutes_of_day = hour * 60 + minute
    offsets = np.arange(n_steps) * STEP_MINUTES  # 0..355
    mins_of_day = (base_minutes_of_day - (WINDOW_STEPS - 1) * STEP_MINUTES + offsets) % (24 * 60)
    hours_frac = mins_of_day / 60.0
    mins_frac = mins_of_day % 60
    feats = np.column_stack([
        np.sin(2 * np.pi * hours_frac / 24.0),
        np.cos(2 * np.pi * hours_frac / 24.0),
        np.sin(2 * np.pi * mins_frac / 60.0),
        np.cos(2 * np.pi * mins_frac / 60.0),
    ]).astype(np.float32)
    return feats


# ══════════════════════════════════════════════════════════════════════
#  SCALING
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ChannelScaler:
    """Per-channel scaler fitted on training data only."""
    method: str = "standard"       # "standard" or "robust"
    center: float = 0.0
    scale: float = 1.0

    def fit(self, values: np.ndarray):
        vals = values[~np.isnan(values)]
        if len(vals) == 0:
            self.center, self.scale = 0.0, 1.0
            return self
        if self.method == "robust":
            self.center = float(np.median(vals))
            iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            self.scale = iqr if iqr > 1e-8 else 1.0
        else:
            self.center = float(np.mean(vals))
            self.scale = float(np.std(vals))
            if self.scale < 1e-8:
                self.scale = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.center) / self.scale


class MultiChannelScaler:
    """Fits and applies per-channel scaling."""

    def __init__(self, channel_names: List[str]):
        self.channel_names = channel_names
        self.scalers: Dict[str, ChannelScaler] = {}

    def fit(self, X: np.ndarray):
        """X: (N, 72, C). Fit on non-NaN values per channel."""
        for i, ch in enumerate(self.channel_names):
            method = "robust" if ch in ROBUST_SCALE_CHANNELS else "standard"
            scaler = ChannelScaler(method=method)
            scaler.fit(X[:, :, i].ravel())
            self.scalers[ch] = scaler
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X.copy()
        for i, ch in enumerate(self.channel_names):
            if ch in self.scalers:
                X_out[:, :, i] = self.scalers[ch].transform(X_out[:, :, i])
        return X_out

    def summary(self) -> str:
        lines = []
        for ch, sc in self.scalers.items():
            lines.append(f"  {ch:20s}: method={sc.method:8s}  center={sc.center:10.4f}  scale={sc.scale:10.4f}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  IMPUTATION (CAUSAL ONLY)
# ══════════════════════════════════════════════════════════════════════

def causal_impute(X: np.ndarray, mask: np.ndarray, channel_names: List[str],
                  max_gap: int = MAX_FFILL_GAP) -> np.ndarray:
    """
    Causal imputation: forward-fill glucose within short gaps, zero-fill others.
    Modifies X in-place. Mask stays unchanged (still marks original missingness).
    """
    X_imp = X.copy()
    for ci, ch in enumerate(channel_names):
        for sample_idx in range(X_imp.shape[0]):
            col = X_imp[sample_idx, :, ci]
            m = mask[sample_idx, :, ci]
            if ch == "glucose":
                # Forward-fill short gaps only
                last_valid = np.nan
                gap_count = 0
                for t in range(len(col)):
                    if m[t]:
                        last_valid = col[t]
                        gap_count = 0
                    else:
                        gap_count += 1
                        if gap_count <= max_gap and not np.isnan(last_valid):
                            col[t] = last_valid
                        else:
                            col[t] = 0.0
            else:
                # Zero-fill non-glucose channels
                col[np.isnan(col)] = 0.0
            X_imp[sample_idx, :, ci] = col
    return X_imp


# ══════════════════════════════════════════════════════════════════════
#  BRIST1D LOADER
# ══════════════════════════════════════════════════════════════════════

# Map BrisT1D prefixes → unified channel names
BRIST1D_PREFIX_MAP = {
    "bg": "glucose",
    "insulin": "insulin",
    "carbs": "carbs",
    "hr": "hr",
    "steps": "steps",
    "cals": "cals_activity",
    "activity": "activity_label",
}


def _parse_lag_minutes(col_name: str) -> Optional[Tuple[str, int]]:
    """Parse 'bg-5:55' → ('bg', 355)."""
    m = re.match(r'^([a-z_]+)-(\d+):(\d+)$', col_name)
    if m:
        prefix = m.group(1)
        minutes = int(m.group(2)) * 60 + int(m.group(3))
        return prefix, minutes
    return None


def load_brist1d(scenario: int = 1) -> DatasetSamples:
    """
    Load BrisT1D. Data is already in windowed format.
    Each row → one sample of shape (72, C).
    """
    print("=" * 60)
    print(f"Loading BrisT1D (Scenario {scenario})")
    print("=" * 60)

    train_path = os.path.join(BRIST1D_DIR, "train.csv")
    df = pd.read_csv(train_path)
    print(f"  Raw rows: {len(df):,}")

    # --- Determine channels for this scenario ---
    base_channels = list(SCENARIO_CHANNELS[scenario])
    # T1D: add insulin to all scenarios
    insulin_ch = T1D_INSULIN_CHANNELS["brist1d"]
    for ch in insulin_ch:
        if ch not in base_channels:
            base_channels.append(ch)
    # Add time features
    time_channels = ["hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    all_channels = base_channels + time_channels

    # --- Map unified channel names → BrisT1D prefixes ---
    unified_to_prefix = {v: k for k, v in BRIST1D_PREFIX_MAP.items()}

    # --- Parse all lag columns, organise by prefix and sorted offset ---
    lag_info = {}  # prefix → sorted list of (minutes, col_name)
    for c in df.columns:
        parsed = _parse_lag_minutes(c)
        if parsed:
            prefix, minutes = parsed
            if prefix not in lag_info:
                lag_info[prefix] = []
            lag_info[prefix].append((minutes, c))
    for prefix in lag_info:
        lag_info[prefix].sort(key=lambda x: x[0], reverse=True)
        # Now index 0 = oldest (5:55 = 355 min ago), index 71 = newest (0:00)

    # Verify 72 steps for bg
    assert len(lag_info["bg"]) == 72, f"Expected 72 bg lags, got {len(lag_info['bg'])}"

    # --- Build samples ---
    samples = []
    dropped_no_target = 0

    for idx, row in df.iterrows():
        # Target
        y_abs = row["bg+1:00"]
        if pd.isna(y_abs):
            dropped_no_target += 1
            continue

        bg_current = row[lag_info["bg"][-1][1]]  # bg-0:00
        y_delta = y_abs - bg_current if not pd.isna(bg_current) else np.nan

        # Build X and mask
        X = np.full((WINDOW_STEPS, len(all_channels)), np.nan, dtype=np.float32)
        mask_arr = np.zeros((WINDOW_STEPS, len(all_channels)), dtype=bool)

        for ci, ch in enumerate(base_channels):
            prefix = unified_to_prefix.get(ch)
            if prefix and prefix in lag_info:
                col_list = lag_info[prefix]
                for step_idx, (minutes, col_name) in enumerate(col_list):
                    val = row[col_name]
                    if not pd.isna(val):
                        X[step_idx, ci] = val
                        mask_arr[step_idx, ci] = True

        # Time features
        time_str = str(row["time"])
        parts = time_str.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        time_feats = make_time_features_from_single(hour, minute, WINDOW_STEPS)
        for ti, tch in enumerate(time_channels):
            ci = base_channels.__len__() + ti
            X[:, ci] = time_feats[:, ti]
            mask_arr[:, ci] = True

        samples.append(Sample(
            X=X, y_abs=float(y_abs), y_delta=float(y_delta) if not np.isnan(y_delta) else float(y_abs),
            mask=mask_arr,
            participant_id=str(row["p_num"]),
            dataset_name="brist1d",
            timestamp=time_str,
            scenario=scenario,
            channel_names=all_channels,
            glucose_unit="mmol/L",
        ))

    ds = DatasetSamples(samples=samples, dataset_name="brist1d",
                        scenario=scenario, channel_names=all_channels)
    print(f"  Samples built: {len(ds.samples):,}")
    print(f"  Dropped (no target): {dropped_no_target:,}")
    print(f"  Channels ({len(all_channels)}): {all_channels}")
    return ds


# ══════════════════════════════════════════════════════════════════════
#  HUPA LOADER
# ══════════════════════════════════════════════════════════════════════

HUPA_COL_MAP = {
    "glucose": "glucose",
    "carb_input": "carbs",
    "heart_rate": "hr",
    "steps": "steps",
    "calories": "cals_activity",
    "basal_rate": "basal",
    "bolus_volume_delivered": "bolus",
}


def _build_sliding_windows(df: pd.DataFrame, channel_cols: List[str],
                           glucose_col: str, pid: str, dataset_name: str,
                           scenario: int, glucose_unit: str,
                           all_channel_names: List[str]) -> Tuple[List[Sample], int]:
    """
    Build sliding-window samples from a participant's 5-min DataFrame.
    Returns (samples, n_dropped).
    """
    n = len(df)
    samples = []
    dropped = 0
    time_channels = ["hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    n_base = len(channel_cols)

    for i in range(n - WINDOW_STEPS - HORIZON_STEPS + 1):
        # Window: i .. i+71
        # Target: i + 71 + 12  (i.e. 60 min after end of window)
        target_idx = i + WINDOW_STEPS - 1 + HORIZON_STEPS
        if target_idx >= n:
            break

        y_abs = df.iloc[target_idx][glucose_col]
        if pd.isna(y_abs):
            dropped += 1
            continue

        current_glucose = df.iloc[i + WINDOW_STEPS - 1][glucose_col]
        y_delta = y_abs - current_glucose if not pd.isna(current_glucose) else y_abs

        window = df.iloc[i:i + WINDOW_STEPS]

        # Build X and mask for base channels
        X = np.full((WINDOW_STEPS, len(all_channel_names)), np.nan, dtype=np.float32)
        mask_arr = np.zeros((WINDOW_STEPS, len(all_channel_names)), dtype=bool)

        for ci, col in enumerate(channel_cols):
            vals = window[col].values.astype(np.float32)
            valid = ~np.isnan(vals)
            X[:, ci] = vals
            mask_arr[:, ci] = valid

        # Time features
        timestamps = pd.DatetimeIndex(window["time"])
        time_feats = make_time_features(timestamps)
        for ti in range(4):
            X[:, n_base + ti] = time_feats[:, ti]
            mask_arr[:, n_base + ti] = True

        ts_str = str(df.iloc[i + WINDOW_STEPS - 1]["time"])

        samples.append(Sample(
            X=X, y_abs=float(y_abs), y_delta=float(y_delta),
            mask=mask_arr,
            participant_id=pid,
            dataset_name=dataset_name,
            timestamp=ts_str,
            scenario=scenario,
            channel_names=all_channel_names,
            glucose_unit=glucose_unit,
        ))
    return samples, dropped


def load_hupa(scenario: int = 1) -> DatasetSamples:
    """Load HUPA-UCM dataset. Semicolon-delimited, 5-min resolution."""
    print("=" * 60)
    print(f"Loading HUPA-UCM (Scenario {scenario})")
    print("=" * 60)

    # Determine channels
    base_channels = list(SCENARIO_CHANNELS[scenario])
    # T1D: add insulin
    for ch in T1D_INSULIN_CHANNELS["hupa"]:
        if ch not in base_channels:
            base_channels.append(ch)
    time_channels = ["hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    all_channels = base_channels + time_channels

    # Map unified → HUPA column names
    unified_to_hupa = {v: k for k, v in HUPA_COL_MAP.items()}

    csv_files = sorted(glob.glob(os.path.join(HUPA_DIR, "*.csv")))
    all_samples = []
    total_dropped = 0

    for csv_path in csv_files:
        pid = Path(csv_path).stem  # e.g., "HUPA0001P"
        df = pd.read_csv(csv_path, sep=";")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        # Map columns
        hupa_cols = []
        for ch in base_channels:
            hupa_col = unified_to_hupa.get(ch, ch)
            if hupa_col in df.columns:
                hupa_cols.append(hupa_col)
            else:
                # Channel not available in HUPA → create NaN column
                df[ch] = np.nan
                hupa_cols.append(ch)

        samples, dropped = _build_sliding_windows(
            df, hupa_cols, "glucose", pid, "hupa",
            scenario, "mg/dL", all_channels
        )
        all_samples.extend(samples)
        total_dropped += dropped

    ds = DatasetSamples(samples=all_samples, dataset_name="hupa",
                        scenario=scenario, channel_names=all_channels)
    print(f"  Participants: {len(csv_files)}")
    print(f"  Samples built: {len(ds.samples):,}")
    print(f"  Dropped (no target): {total_dropped:,}")
    print(f"  Channels ({len(all_channels)}): {all_channels}")
    return ds


# ══════════════════════════════════════════════════════════════════════
#  CGMACROS LOADER
# ══════════════════════════════════════════════════════════════════════

CGMACROS_COL_MAP = {
    "Libre GL": "glucose",
    "Carbs": "carbs",
    "Protein": "protein",
    "Fat": "fat",
    "Fiber": "fiber",
    "Calories": "meal_cal",
    "Amount Consumed ": "consumed_pct",  # note trailing space
    "HR": "hr",
    "METs": "mets",
    "Calories (Activity)": "cals_activity",
}


def load_cgmacros(scenario: int = 1) -> DatasetSamples:
    """Load CGMacros. Needs resampling from ~1-min to 5-min."""
    print("=" * 60)
    print(f"Loading CGMacros (Scenario {scenario})")
    print("=" * 60)

    base_channels = list(SCENARIO_CHANNELS[scenario])
    # CGMacros has no insulin
    # Remove channels not in CGMacros
    available = set(CGMACROS_COL_MAP.values())
    base_channels = [ch for ch in base_channels if ch in available]
    # Ensure glucose is first
    if "glucose" not in base_channels:
        base_channels.insert(0, "glucose")

    time_channels = ["hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    all_channels = base_channels + time_channels

    # Reverse map: unified → CGMacros raw column
    unified_to_cgm = {v: k for k, v in CGMACROS_COL_MAP.items()}

    folders = sorted(glob.glob(os.path.join(CGMACROS_DIR, "CGMacros-0*")))
    all_samples = []
    total_dropped = 0

    for folder in folders:
        pid = os.path.basename(folder)
        csv_files = [f for f in glob.glob(os.path.join(folder, "*.csv"))
                     if "bio" not in f.lower()]
        if not csv_files:
            continue

        df = pd.read_csv(csv_files[0])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)

        # Resample to 5-minute intervals
        df = df.set_index("Timestamp")

        # For numeric columns: take mean over 5-min bins
        # For meal columns: take sum (a meal event shouldn't be averaged away)
        sum_cols = ["Carbs", "Protein", "Fat", "Fiber", "Calories"]
        mean_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in sum_cols]

        agg_dict = {}
        for c in mean_cols:
            if c in df.columns:
                agg_dict[c] = "mean"
        for c in sum_cols:
            if c in df.columns:
                agg_dict[c] = "sum"
        # Amount Consumed: take max (percentage consumed)
        ac_col = "Amount Consumed "
        if ac_col in df.columns:
            agg_dict[ac_col] = "max"

        df_5min = df.resample("5min").agg(agg_dict)
        df_5min = df_5min.reset_index()
        df_5min.rename(columns={"Timestamp": "time"}, inplace=True)

        # Map columns
        cgm_cols = []
        for ch in base_channels:
            raw_col = unified_to_cgm.get(ch, ch)
            if raw_col in df_5min.columns:
                cgm_cols.append(raw_col)
            else:
                df_5min[ch] = np.nan
                cgm_cols.append(ch)

        glucose_raw_col = unified_to_cgm.get("glucose", "Libre GL")

        samples, dropped = _build_sliding_windows(
            df_5min, cgm_cols, glucose_raw_col, pid, "cgmacros",
            scenario, "mg/dL", all_channels
        )
        all_samples.extend(samples)
        total_dropped += dropped

    ds = DatasetSamples(samples=all_samples, dataset_name="cgmacros",
                        scenario=scenario, channel_names=all_channels)
    print(f"  Participants: {len(folders)}")
    print(f"  Samples built: {len(ds.samples):,}")
    print(f"  Dropped (no target): {total_dropped:,}")
    print(f"  Channels ({len(all_channels)}): {all_channels}")
    return ds


# ══════════════════════════════════════════════════════════════════════
#  SPLIT + SCALE PIPELINE
# ══════════════════════════════════════════════════════════════════════

def split_samples(ds: DatasetSamples, dataset_name: str
                  ) -> Dict[str, DatasetSamples]:
    """Split samples by participant into train/val/test."""
    splits: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}

    if dataset_name == "brist1d":
        # BrisT1D: all from train.csv → use GroupKFold later.
        # For now, hash-based split for quick iteration.
        for s in ds.samples:
            split = assign_split(s.participant_id)
            splits[split].append(s)
    else:
        for s in ds.samples:
            split = assign_split(s.participant_id)
            splits[split].append(s)

    result = {}
    for split_name, samples in splits.items():
        result[split_name] = DatasetSamples(
            samples=samples, dataset_name=dataset_name,
            scenario=ds.scenario, channel_names=ds.channel_names
        )
    return result


def fit_and_transform(split_data: Dict[str, DatasetSamples]
                      ) -> Tuple[Dict[str, DatasetSamples], MultiChannelScaler]:
    """Fit scaler on train, transform all splits."""
    train_ds = split_data["train"]
    channel_names = train_ds.channel_names

    # Fit scaler on train
    X_train = train_ds.X
    scaler = MultiChannelScaler(channel_names)
    scaler.fit(X_train)
    print(f"\nScaler summary:\n{scaler.summary()}")

    # Transform all splits
    result = {}
    for split_name, ds in split_data.items():
        X = ds.X
        X_scaled = scaler.transform(X)
        # Rebuild samples with scaled X
        new_samples = []
        for i, s in enumerate(ds.samples):
            new_samples.append(Sample(
                X=X_scaled[i], y_abs=s.y_abs, y_delta=s.y_delta,
                mask=s.mask, participant_id=s.participant_id,
                dataset_name=s.dataset_name, timestamp=s.timestamp,
                scenario=s.scenario, channel_names=s.channel_names,
                glucose_unit=s.glucose_unit,
            ))
        result[split_name] = DatasetSamples(
            samples=new_samples, dataset_name=ds.dataset_name,
            scenario=ds.scenario, channel_names=ds.channel_names
        )
    return result, scaler


# ══════════════════════════════════════════════════════════════════════
#  REPORTING & SANITY PLOTS
# ══════════════════════════════════════════════════════════════════════

def print_report(ds: DatasetSamples, split_data: Dict[str, DatasetSamples]):
    """Print comprehensive statistics."""
    print(f"\n{'='*60}")
    print(f"REPORT: {ds.dataset_name.upper()} — Scenario {ds.scenario}")
    print(f"{'='*60}")
    print(f"Total samples: {len(ds.samples):,}")
    print(f"Channels: {ds.channel_names}")
    print(f"Shape per sample: X=({WINDOW_STEPS}, {len(ds.channel_names)})")

    # Per-participant counts
    pid_counts = {}
    for s in ds.samples:
        pid_counts[s.participant_id] = pid_counts.get(s.participant_id, 0) + 1
    print(f"\nPer-participant sample counts:")
    for pid in sorted(pid_counts):
        print(f"  {pid}: {pid_counts[pid]:,}")

    # Missingness
    mask_all = ds.mask
    print(f"\nMissingness rates (% NaN):")
    for ci, ch in enumerate(ds.channel_names):
        miss = 1.0 - mask_all[:, :, ci].mean()
        print(f"  {ch:20s}: {miss*100:6.2f}%")

    # Split sizes
    print(f"\nSplit sizes:")
    for split_name, sds in split_data.items():
        pids = set(s.participant_id for s in sds.samples)
        print(f"  {split_name:6s}: {len(sds.samples):>8,} samples  ({len(pids)} participants: {sorted(pids)})")

    # Target stats
    y_abs = ds.y_abs
    y_delta = ds.y_delta
    print(f"\nTarget statistics (y_abs):")
    print(f"  mean={np.nanmean(y_abs):.2f}  std={np.nanstd(y_abs):.2f}  "
          f"min={np.nanmin(y_abs):.2f}  max={np.nanmax(y_abs):.2f}")
    print(f"Target statistics (y_delta):")
    print(f"  mean={np.nanmean(y_delta):.2f}  std={np.nanstd(y_delta):.2f}  "
          f"min={np.nanmin(y_delta):.2f}  max={np.nanmax(y_delta):.2f}")


def plot_sanity(ds: DatasetSamples, dataset_name: str):
    """Create sanity plots: one sample window + target distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Sample window ---
    ax = axes[0]
    sample = ds.samples[len(ds.samples) // 3]  # pick a sample from ~1/3 in
    x_minutes = np.arange(WINDOW_STEPS) * STEP_MINUTES - (WINDOW_STEPS - 1) * STEP_MINUTES
    gl_idx = ds.channel_names.index("glucose") if "glucose" in ds.channel_names else 0
    gl_trace = sample.X[:, gl_idx]
    gl_mask = sample.mask[:, gl_idx]
    ax.plot(x_minutes, gl_trace, color="steelblue", label="Glucose")
    ax.scatter(x_minutes[~gl_mask], gl_trace[~gl_mask], color="red", s=10, zorder=5, label="Missing")
    # Mark target
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.scatter([60], [sample.y_abs], color="red", marker="*", s=200, zorder=10, label=f"Target={sample.y_abs:.1f}")
    ax.set_title(f"{dataset_name} — Sample Window\n(pid={sample.participant_id})")
    ax.set_xlabel("Minutes relative to t=0")
    ax.set_ylabel(f"Glucose ({sample.glucose_unit})")
    ax.legend(fontsize=7)

    # --- Plot 2: y_abs distribution ---
    ax = axes[1]
    y_abs = ds.y_abs
    ax.hist(y_abs[~np.isnan(y_abs)], bins=60, color="coral", edgecolor="white", alpha=0.8)
    ax.set_title(f"y_abs Distribution\n(μ={np.nanmean(y_abs):.1f}, σ={np.nanstd(y_abs):.1f})")
    ax.set_xlabel(f"Glucose ({ds.samples[0].glucose_unit})")
    ax.set_ylabel("Count")

    # --- Plot 3: y_delta distribution ---
    ax = axes[2]
    y_delta = ds.y_delta
    ax.hist(y_delta[~np.isnan(y_delta)], bins=60, color="mediumpurple", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(f"y_delta Distribution\n(μ={np.nanmean(y_delta):.1f}, σ={np.nanstd(y_delta):.1f})")
    ax.set_xlabel("Δ Glucose (t+60 − t)")
    ax.set_ylabel("Count")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"sanity_{dataset_name}_s{ds.scenario}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sanity plot saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

def run_dataset(loader_fn, dataset_name: str, scenario: int):
    """Load, split, report, and plot for one dataset+scenario."""
    ds = loader_fn(scenario=scenario)
    split_data = split_samples(ds, dataset_name)
    print_report(ds, split_data)
    plot_sanity(ds, dataset_name)
    return ds, split_data


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print(" UNIFIED DATASET BUILDER — Scenario 1 (Carbs)")
    print("█" * 60)

    # Run Scenario 1 for all datasets
    brist1d_ds, brist1d_splits = run_dataset(load_brist1d, "brist1d", scenario=1)
    hupa_ds, hupa_splits = run_dataset(load_hupa, "hupa", scenario=1)
    cgm_ds, cgm_splits = run_dataset(load_cgmacros, "cgmacros", scenario=1)

    # Demonstrate scaling on one dataset
    print("\n" + "=" * 60)
    print("SCALING DEMONSTRATION (HUPA Scenario 1)")
    print("=" * 60)
    hupa_scaled, hupa_scaler = fit_and_transform(hupa_splits)
    print(f"\nTrain X shape after scaling: {hupa_scaled['train'].X.shape}")

    # Demonstrate imputation on one dataset
    print("\n" + "=" * 60)
    print("IMPUTATION DEMONSTRATION (BrisT1D Scenario 1)")
    print("=" * 60)
    train_X = brist1d_splits["train"].X
    train_mask = brist1d_splits["train"].mask
    nans_before = np.isnan(train_X).sum()
    X_imputed = causal_impute(train_X, train_mask, brist1d_ds.channel_names)
    nans_after = np.isnan(X_imputed).sum()
    print(f"  NaNs before imputation: {nans_before:,}")
    print(f"  NaNs after imputation:  {nans_after:,}")

    # Quick shape check for Scenario 2 and 3 (CGMacros only — has all channels)
    print("\n" + "=" * 60)
    print("SCENARIO FEATURE SHAPES")
    print("=" * 60)
    for sc in [1, 2, 3]:
        ds_temp = load_cgmacros(scenario=sc)
        print(f"  CGMacros Scenario {sc}: X shape = (N, 72, {len(ds_temp.channel_names)}) "
              f"channels = {ds_temp.channel_names}")

    print("\n" + "█" * 60)
    print(" STEP 1 COMPLETE")
    print("█" * 60)
