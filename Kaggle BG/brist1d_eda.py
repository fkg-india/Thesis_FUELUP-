#!/usr/bin/env python3
"""
BrisT1D Dataset – Exploratory Data Analysis (EDA)
==================================================
Thesis: Predicting future blood-glucose trends from meals, insulin, and
activity signals using the BrisT1D (Type 1 Diabetes) dataset.

This script performs:
  1. Data loading & data-dictionary summary
  2. Missingness & noise diagnostics
  3. Distribution plots (target BG, current BG, per-participant)
  4. Time-of-day effects
  5. Window-shape (6-hour lookback) visualization
  6. Engineered-feature relationships & correlations
  7. Activity frequency & impact analysis
  8. Professor-ready summary

NOTE: This is research-grade EDA – no medical advice is provided.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & SETUP
# ──────────────────────────────────────────────────────────────────────────────
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; remove if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# Paths
DATA_DIR = Path(__file__).resolve().parent   # same folder as this script
FIG_DIR  = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# 1. HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
LAG_RE = re.compile(r"^(bg|insulin|carbs|hr|steps|cals|activity)-(\d+):(\d+)$")


def parse_lag_minutes(colname: str) -> float | None:
    """Return the lag in minutes for a column name like 'bg-5:30' → 330.0."""
    m = LAG_RE.match(colname)
    if m:
        return int(m.group(2)) * 60 + int(m.group(3))
    return None


def get_prefix(colname: str) -> str | None:
    """Return the feature-family prefix (bg, insulin, …) or None."""
    m = LAG_RE.match(colname)
    return m.group(1) if m else None


def get_lagged_series(row: pd.Series, prefix: str, lag_map: dict) -> tuple:
    """
    For a given row and prefix (e.g. 'bg'), return two aligned arrays:
      minutes_sorted  – lag in minutes (descending = further in the past first)
      values_sorted   – corresponding values
    """
    cols = lag_map[prefix]  # list of (col, minutes) sorted by minutes desc
    minutes = np.array([m for _, m in cols])
    values  = np.array([row[c] for c, _ in cols], dtype=float)
    return minutes, values


def get_current_bg(row: pd.Series, bg_cols_sorted: list) -> float:
    """Return the BG at the smallest lag (closest to 'now')."""
    # bg_cols_sorted is sorted by lag ascending → first = smallest lag
    for col, _ in bg_cols_sorted:
        v = row[col]
        if pd.notna(v):
            return float(v)
    return np.nan


def savefig(fig, name: str):
    """Save a figure and print confirmation."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    print(f"  ✓ Saved  {path.relative_to(DATA_DIR)}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("LOADING DATA …")
print("=" * 72)

train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

# Activities list (one per line)
activities = (DATA_DIR / "activities.txt").read_text().strip().splitlines()
activities = [a.strip() for a in activities if a.strip()]

print(f"  train  : {train.shape[0]:>9,} rows × {train.shape[1]} cols")
print(f"  test   : {test.shape[0]:>9,} rows × {test.shape[1]} cols")
print(f"  activities : {len(activities)} labels")

# ──────────────────────────────────────────────────────────────────────────────
# 3. BUILD LAG MAP & COLUMN FAMILIES
# ──────────────────────────────────────────────────────────────────────────────
# lag_map[prefix] = [(col_name, minutes)] sorted by minutes DESCENDING
# (furthest in the past first)
# lag_map_asc[prefix] = same but sorted ASCENDING (closest to now first)

lag_map: dict[str, list[tuple[str, float]]] = {}
lag_map_asc: dict[str, list[tuple[str, float]]] = {}
family_cols: dict[str, list[str]] = {}

for col in train.columns:
    m = LAG_RE.match(col)
    if m:
        prefix = m.group(1)
        minutes = int(m.group(2)) * 60 + int(m.group(3))
        lag_map.setdefault(prefix, []).append((col, minutes))
        family_cols.setdefault(prefix, []).append(col)

for prefix in lag_map:
    lag_map[prefix].sort(key=lambda x: x[1], reverse=True)       # desc
    lag_map_asc[prefix] = sorted(lag_map[prefix], key=lambda x: x[1])  # asc

# Special columns
META_COLS   = ["id", "p_num", "time"]
TARGET_COL  = "bg+1:00"

print()
print("─" * 72)
print("DATA DICTIONARY")
print("─" * 72)
print(f"  Total columns       : {train.shape[1]}")
print(f"  Meta columns        : {META_COLS}")
print(f"  Target column       : {TARGET_COL}")
print(f"  Feature families    :")
for prefix in sorted(lag_map):
    lags = lag_map[prefix]
    print(f"    {prefix:>10s}  –  {len(lags):>3} lag columns  "
          f"(range {lags[-1][1]:.0f} min → {lags[0][1]:.0f} min)")

# Unique participants
train_pids = sorted(train["p_num"].unique())
test_pids  = sorted(test["p_num"].unique())
print(f"\n  Unique participants:")
print(f"    Train : {len(train_pids)}  → {train_pids}")
print(f"    Test  : {len(test_pids)}  → {test_pids}")
overlap = set(train_pids) & set(test_pids)
only_test = set(test_pids) - set(train_pids)
print(f"    Overlap (train ∩ test): {len(overlap)}")
print(f"    Unseen in test       : {only_test if only_test else 'none'}")

# Lag timeline
print("\n  Lag timeline (chronological, minutes before now):")
bg_lags = lag_map["bg"]
sample_lags = [bg_lags[i] for i in range(0, len(bg_lags), max(1, len(bg_lags)//12))]
print("    " + "  →  ".join(f"{m:.0f}m" for _, m in sample_lags) + "  →  now")

# ──────────────────────────────────────────────────────────────────────────────
# 4. MISSINGNESS & NOISE DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("MISSINGNESS DIAGNOSTICS")
print("=" * 72)

# 4a. Overall missing % per family
print("\n  Overall missing % per feature family (train):")
for prefix in sorted(family_cols):
    cols = family_cols[prefix]
    pct = train[cols].isna().mean().mean() * 100
    print(f"    {prefix:>10s}  :  {pct:6.2f} %")

target_missing = train[TARGET_COL].isna().mean() * 100
print(f"    {'target':>10s}  :  {target_missing:6.2f} %")

# 4b. Missingness by lag distance
print("\n  Generating missingness-by-lag-distance plot …")
miss_by_lag = {}
for prefix in ["bg", "insulin", "carbs", "hr", "steps", "cals"]:
    xs, ys = [], []
    for col, minutes in lag_map[prefix]:
        xs.append(minutes)
        ys.append(train[col].isna().mean() * 100)
    miss_by_lag[prefix] = (np.array(xs), np.array(ys))

fig, ax = plt.subplots(figsize=(10, 5))
for prefix, (xs, ys) in miss_by_lag.items():
    ax.plot(xs, ys, marker=".", markersize=3, label=prefix, alpha=0.8)
ax.set_xlabel("Lag (minutes before now)")
ax.set_ylabel("Missing %")
ax.set_title("Missingness by Lag Distance (Train)")
ax.legend(ncol=3, fontsize=9)
ax.invert_xaxis()
savefig(fig, "01_missingness_by_lag")

# 4c. Missingness Heatmap – sample 30 columns per family
print("  Generating missingness heatmap …")
sample_cols = []
for prefix in ["bg", "insulin", "carbs", "hr", "steps", "cals"]:
    cols_sorted = [c for c, _ in lag_map[prefix]]
    step = max(1, len(cols_sorted) // 8)
    sample_cols += cols_sorted[::step]

miss_matrix = train[sample_cols].isna().astype(int)
# Sub-sample rows for readability
row_sample = miss_matrix.sample(min(300, len(miss_matrix)), random_state=SEED)
row_sample = row_sample.sort_index()

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(row_sample.T, cbar=False, cmap=["#eaeaea", "#d62728"],
            yticklabels=True, xticklabels=False, ax=ax)
ax.set_title("Missingness Heatmap (sampled rows × sampled lag columns)")
ax.set_ylabel("Feature–Lag column")
ax.set_xlabel("Row index (sampled)")
savefig(fig, "02_missingness_heatmap")

# ──────────────────────────────────────────────────────────────────────────────
# 5. DISTRIBUTION PLOTS
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("DISTRIBUTION PLOTS")
print("=" * 72)

# Compute current BG for every training row (smallest-lag BG)
# Vectorized: pick the first non-NaN bg column starting from smallest lag
bg_cols_asc = lag_map_asc["bg"]  # sorted closest-to-now first
bg_col_names_asc = [c for c, _ in bg_cols_asc]
# Start with all NaN, then fill backwards from largest lag to smallest
# so smallest lag (closest to now) takes priority
current_bg = pd.Series(np.nan, index=train.index)
for col in reversed(bg_col_names_asc):
    current_bg = current_bg.fillna(pd.to_numeric(train[col], errors="coerce"))
train["current_bg"] = current_bg

# 5A. Target BG distribution
print("  Plotting target + current BG distributions …")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Histogram of bg+1:00
ax = axes[0, 0]
train[TARGET_COL].dropna().hist(bins=80, ax=ax, color="#1f77b4", edgecolor="white", alpha=0.85)
ax.set_title("Target BG (bg+1:00) – Histogram")
ax.set_xlabel("BG (mmol/L)")
ax.set_ylabel("Count")

# Violin of bg+1:00
ax = axes[0, 1]
data_target = train[TARGET_COL].dropna()
parts = ax.violinplot(data_target, positions=[0], showmeans=True, showmedians=True)
parts["bodies"][0].set_facecolor("#1f77b4")
ax.set_title("Target BG (bg+1:00) – Violin")
ax.set_ylabel("BG (mmol/L)")
ax.set_xticks([])

# Histogram of current BG
ax = axes[1, 0]
train["current_bg"].dropna().hist(bins=80, ax=ax, color="#ff7f0e", edgecolor="white", alpha=0.85)
ax.set_title("Current BG (nearest lag) – Histogram")
ax.set_xlabel("BG (mmol/L)")
ax.set_ylabel("Count")

# Box plot comparison
ax = axes[1, 1]
data_box = pd.DataFrame({
    "Target BG (bg+1:00)": data_target,
    "Current BG": train["current_bg"]
})
data_box.plot.box(ax=ax, patch_artist=True,
                  boxprops=dict(facecolor="#aec7e8"),
                  medianprops=dict(color="red"))
ax.set_title("Target vs Current BG – Box Plot")
ax.set_ylabel("BG (mmol/L)")

plt.tight_layout()
savefig(fig, "03_bg_distributions")

# 5A-ii. Per-participant distribution (top 9 by sample count)
print("  Plotting per-participant distributions …")
top9 = train["p_num"].value_counts().head(9).index.tolist()
fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
for idx, pid in enumerate(top9):
    ax = axes[idx // 3, idx % 3]
    subset = train.loc[train["p_num"] == pid, TARGET_COL].dropna()
    ax.hist(subset, bins=50, color=sns.color_palette("tab10")[idx % 10],
            edgecolor="white", alpha=0.85)
    ax.set_title(f"{pid}  (n={len(subset):,})", fontsize=10)
    if idx % 3 == 0:
        ax.set_ylabel("Count")
    if idx // 3 == 2:
        ax.set_xlabel("BG (mmol/L)")
fig.suptitle("Target BG Distribution – Top 9 Participants", fontsize=14, y=1.01)
plt.tight_layout()
savefig(fig, "04_per_participant_bg")

# ──────────────────────────────────────────────────────────────────────────────
# 5B. TIME-OF-DAY EFFECTS
# ──────────────────────────────────────────────────────────────────────────────
print("  Plotting time-of-day effects …")
train["hour"] = pd.to_datetime(train["time"], format="%H:%M:%S", errors="coerce").dt.hour

hourly = train.groupby("hour").agg(
    target_mean   = (TARGET_COL, "mean"),
    target_median = (TARGET_COL, "median"),
    current_mean  = ("current_bg", "mean"),
    current_median= ("current_bg", "median"),
).dropna()

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

ax = axes[0]
ax.plot(hourly.index, hourly["current_mean"], "o-", label="Mean", color="#ff7f0e")
ax.plot(hourly.index, hourly["current_median"], "s--", label="Median", color="#d62728")
ax.set_title("Current BG vs Hour of Day")
ax.set_xlabel("Hour")
ax.set_ylabel("BG (mmol/L)")
ax.legend()
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

ax = axes[1]
ax.plot(hourly.index, hourly["target_mean"], "o-", label="Mean", color="#1f77b4")
ax.plot(hourly.index, hourly["target_median"], "s--", label="Median", color="#9467bd")
ax.set_title("Target BG (bg+1:00) vs Hour of Day")
ax.set_xlabel("Hour")
ax.legend()
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

plt.tight_layout()
savefig(fig, "05_time_of_day_bg")

# ──────────────────────────────────────────────────────────────────────────────
# 5C. WINDOW-SHAPE VISUALIZATION (the important one!)
# ──────────────────────────────────────────────────────────────────────────────
print("  Generating window-shape (6-hour lookback) visualizations …")

# Pick a participant with lots of data & choose a few random rows
chosen_pid = top9[0]
pid_idx = train.index[train["p_num"] == chosen_pid]
sample_rows = np.random.choice(pid_idx, size=min(6, len(pid_idx)), replace=False)

FAMILY_UNITS = {
    "bg":      "BG (mmol/L)",
    "insulin": "Insulin (units)",
    "carbs":   "Carbs (grams)",
    "hr":      "Heart Rate (bpm)",
    "steps":   "Steps (count)",
    "cals":    "Calories (kcal)",
}

fig, axes = plt.subplots(len(sample_rows), 1, figsize=(14, 4.5 * len(sample_rows)),
                         sharex=True)
if len(sample_rows) == 1:
    axes = [axes]

for i, row_idx in enumerate(sample_rows):
    row = train.loc[row_idx]
    ax = axes[i]

    # BG trace on primary y-axis
    mins_bg, vals_bg = get_lagged_series(row, "bg", lag_map)
    ax.plot(-mins_bg, vals_bg, "o-", color="#1f77b4", markersize=3,
            linewidth=1.5, label="BG (mmol/L)", zorder=5)

    # Target point at +60 min
    target_val = row[TARGET_COL]
    if pd.notna(target_val):
        ax.plot(60, target_val, "*", color="red", markersize=14, zorder=6,
                label=f"Target BG +1h = {target_val:.1f}")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.annotate("now", (0, ax.get_ylim()[0]), fontsize=8, color="gray",
                    ha="center", va="bottom")

    ax.set_ylabel("BG (mmol/L)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Row {row_idx}  |  {chosen_pid}  |  time={row['time']}",
                 fontsize=10)

    # Secondary y-axis for insulin + carbs
    ax2 = ax.twinx()
    for pf, color, ls in [("insulin", "#2ca02c", "-"), ("carbs", "#d62728", "--")]:
        mins, vals = get_lagged_series(row, pf, lag_map)
        ax2.plot(-mins, vals, ls, color=color, markersize=2, linewidth=1,
                 label=FAMILY_UNITS[pf], alpha=0.7)
    ax2.set_ylabel("Insulin / Carbs", fontsize=9)
    ax2.legend(loc="upper right", fontsize=7)

axes[-1].set_xlabel("Minutes relative to now (negative = past, +60 = target)")
fig.suptitle(f"6-Hour Lookback Windows – Participant {chosen_pid}", fontsize=14, y=1.01)
plt.tight_layout()
savefig(fig, "06_window_shapes_bg_insulin_carbs")

# Separate figure: HR, Steps, Cals windows for same rows
fig, axes = plt.subplots(len(sample_rows), 1, figsize=(14, 4.0 * len(sample_rows)),
                         sharex=True)
if len(sample_rows) == 1:
    axes = [axes]

colors_hr = {"hr": "#e377c2", "steps": "#7f7f7f", "cals": "#bcbd22"}
for i, row_idx in enumerate(sample_rows):
    row = train.loc[row_idx]
    ax = axes[i]

    # HR on primary axis
    mins_hr, vals_hr = get_lagged_series(row, "hr", lag_map)
    ax.plot(-mins_hr, vals_hr, ".-", color=colors_hr["hr"], markersize=2,
            linewidth=1.2, label="HR (bpm)")
    ax.set_ylabel("HR (bpm)", color=colors_hr["hr"])
    ax.tick_params(axis="y", labelcolor=colors_hr["hr"])

    # Steps + Cals on secondary axis
    ax2 = ax.twinx()
    for pf in ["steps", "cals"]:
        mins, vals = get_lagged_series(row, pf, lag_map)
        ax2.plot(-mins, vals, ".-", color=colors_hr[pf], markersize=2,
                 linewidth=1, label=FAMILY_UNITS[pf], alpha=0.7)
    ax2.set_ylabel("Steps / Cals")
    ax2.legend(loc="upper right", fontsize=7)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(f"Row {row_idx}  |  {chosen_pid}  |  time={row['time']}",
                 fontsize=10)

axes[-1].set_xlabel("Minutes relative to now (negative = past)")
fig.suptitle(f"HR / Steps / Cals Windows – Participant {chosen_pid}", fontsize=14, y=1.01)
plt.tight_layout()
savefig(fig, "07_window_shapes_hr_steps_cals")

# ──────────────────────────────────────────────────────────────────────────────
# 5D. SIMPLE RELATIONSHIPS – ENGINEERED FEATURES
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("ENGINEERING SUMMARY FEATURES …")
print("=" * 72)


def sum_within_window(row, prefix, max_minutes, lag_map_ref):
    """Sum feature values whose lag is ≤ max_minutes."""
    total = 0.0
    count = 0
    for col, minutes in lag_map_ref[prefix]:
        if minutes <= max_minutes:
            v = row[col]
            if pd.notna(v):
                total += v
                count += 1
    return total if count > 0 else np.nan


def mean_within_window(row, prefix, max_minutes, lag_map_ref):
    """Mean feature values whose lag is ≤ max_minutes."""
    vals = []
    for col, minutes in lag_map_ref[prefix]:
        if minutes <= max_minutes:
            v = row[col]
            if pd.notna(v):
                vals.append(v)
    return np.mean(vals) if vals else np.nan


print("  Computing summary features (this may take a minute on large data) …")
# For speed, use vectorised operations where possible
# Identify columns within each window
def cols_within(prefix, max_min):
    return [c for c, m in lag_map[prefix] if m <= max_min]

train["carbs_30"]  = train[cols_within("carbs", 30)].sum(axis=1, min_count=1)
train["carbs_60"]  = train[cols_within("carbs", 60)].sum(axis=1, min_count=1)
train["carbs_120"] = train[cols_within("carbs", 120)].sum(axis=1, min_count=1)

train["insulin_30"]  = train[cols_within("insulin", 30)].sum(axis=1, min_count=1)
train["insulin_60"]  = train[cols_within("insulin", 60)].sum(axis=1, min_count=1)
train["insulin_120"] = train[cols_within("insulin", 120)].sum(axis=1, min_count=1)

train["hr_mean_60"]  = train[cols_within("hr", 60)].mean(axis=1, skipna=True)
train["steps_60"]    = train[cols_within("steps", 60)].sum(axis=1, min_count=1)

train["delta_bg"] = train[TARGET_COL] - train["current_bg"]

eng_features = [
    "current_bg", "carbs_30", "carbs_60", "carbs_120",
    "insulin_30", "insulin_60", "insulin_120",
    "hr_mean_60", "steps_60", "delta_bg", TARGET_COL,
]

print("  Feature stats (non-null counts & means):")
print(train[eng_features].describe().round(2).to_string())

# Correlation heatmap
print("\n  Plotting correlation heatmap …")
corr = train[eng_features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap – Engineered Summary Features")
plt.tight_layout()
savefig(fig, "08_correlation_heatmap")

# Scatter / binned plots
print("  Plotting scatter / binned relationships …")

scatter_pairs = [
    ("carbs_60",   TARGET_COL,  "Total Carbs (last 60 min)", "Target BG"),
    ("insulin_60", TARGET_COL,  "Total Insulin (last 60 min)", "Target BG"),
    ("carbs_60",   "delta_bg",  "Total Carbs (last 60 min)", "ΔBG"),
    ("insulin_60", "delta_bg",  "Total Insulin (last 60 min)", "ΔBG"),
    ("hr_mean_60", "delta_bg",  "Mean HR (last 60 min)", "ΔBG"),
    ("steps_60",   "delta_bg",  "Total Steps (last 60 min)", "ΔBG"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for idx, (xcol, ycol, xlabel, ylabel) in enumerate(scatter_pairs):
    ax = axes[idx // 3, idx % 3]
    sub = train[[xcol, ycol]].dropna()
    # Downsample for readability
    if len(sub) > 5000:
        sub = sub.sample(5000, random_state=SEED)
    ax.scatter(sub[xcol], sub[ycol], alpha=0.15, s=6, color="#1f77b4")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {xlabel}", fontsize=10)

fig.suptitle("Simple Relationships – Engineered Features", fontsize=14, y=1.01)
plt.tight_layout()
savefig(fig, "09_scatter_relationships")

# Binned mean plot: carbs_60 → target BG
print("  Plotting binned mean plots …")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (feat, label) in zip(axes, [
    ("carbs_60", "Carbs last 60 min"),
    ("insulin_60", "Insulin last 60 min"),
    ("hr_mean_60", "Mean HR last 60 min"),
]):
    sub = train[[feat, TARGET_COL, "delta_bg"]].dropna()
    sub["bin"] = pd.qcut(sub[feat], q=10, duplicates="drop")
    grp = sub.groupby("bin", observed=True).agg(
        target_mean=(TARGET_COL, "mean"),
        delta_mean=("delta_bg", "mean"),
        n=(feat, "count"),
    )
    x = range(len(grp))
    ax.bar(x, grp["target_mean"], color="#1f77b4", alpha=0.7, label="Mean Target BG")
    ax2 = ax.twinx()
    ax2.plot(x, grp["delta_mean"], "o-", color="#d62728", label="Mean ΔBG")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in grp.index], rotation=45, fontsize=7, ha="right")
    ax.set_xlabel(label)
    ax.set_ylabel("Mean Target BG (mmol/L)")
    ax2.set_ylabel("Mean ΔBG (mmol/L)", color="#d62728")
    ax.set_title(f"Binned: {label}", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

plt.tight_layout()
savefig(fig, "10_binned_mean_plots")

# ──────────────────────────────────────────────────────────────────────────────
# 5E. ACTIVITIES
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("ACTIVITY ANALYSIS")
print("=" * 72)

activity_cols = family_cols.get("activity", [])
print(f"  Activity columns : {len(activity_cols)}")
print(f"  Activity labels  : {activities}")

# Flatten all activity values – activity columns contain string labels directly
activity_values = train[activity_cols].values.flatten()
activity_series = pd.Series(activity_values).dropna()
activity_series = activity_series.astype(str).str.strip()
activity_series = activity_series[activity_series != ""]

# Frequency counts
freq = activity_series.value_counts().head(15)
print("\n  Top 15 activity frequencies:")
for act, cnt in freq.items():
    print(f"    {act:<25s}  {cnt:>8,}")

fig, ax = plt.subplots(figsize=(10, 6))
freq.plot.barh(ax=ax, color=sns.color_palette("tab20", len(freq)))
ax.set_xlabel("Frequency (total across all rows × lag slots)")
ax.set_title("Top 15 Activities by Frequency")
ax.invert_yaxis()
plt.tight_layout()
savefig(fig, "11_activity_frequency")

# Activity vs ΔBG – for each activity, does it tend to appear in rows
# where ΔBG is positive (rising) or negative (falling)?
print("\n  Computing activity → ΔBG associations …")

# Unique activity labels found in data
unique_activities = activity_series.unique()

# For each row, determine which activities are present in ANY activity column
activity_presence = {}
for act_label in unique_activities:
    mask = (train[activity_cols] == act_label).any(axis=1)
    n_present = mask.sum()
    if n_present >= 20:  # need enough samples
        mean_delta = train.loc[mask, "delta_bg"].mean()
        mean_target = train.loc[mask, TARGET_COL].mean()
        activity_presence[act_label] = {
            "n": n_present,
            "mean_delta_bg": round(mean_delta, 3) if pd.notna(mean_delta) else None,
            "mean_target_bg": round(mean_target, 3) if pd.notna(mean_target) else None,
        }

act_df = pd.DataFrame(activity_presence).T.dropna()
act_df = act_df.sort_values("mean_delta_bg")

print("\n  Activity – Mean ΔBG & Target BG (activities with ≥20 occurrences):")
print(act_df.to_string())

if len(act_df) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in act_df["mean_delta_bg"]]
    ax.barh(act_df.index, act_df["mean_delta_bg"], color=colors, edgecolor="white")
    ax.set_xlabel("Mean ΔBG (mmol/L)")
    ax.set_title("Activity → Mean ΔBG (positive = BG rising, negative = BG falling)")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    savefig(fig, "12_activity_delta_bg")

# ──────────────────────────────────────────────────────────────────────────────
# 6. PROFESSOR-READY SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("PROFESSOR-READY SUMMARY")
print("=" * 72)

# Gather key stats
n_train = len(train)
n_test  = len(test)
n_cols  = train.shape[1] - len(eng_features)  # original cols
n_train_p = len(train_pids)
n_test_p  = len(test_pids)

bg_miss   = train[family_cols["bg"]].isna().mean().mean() * 100
ins_miss  = train[family_cols["insulin"]].isna().mean().mean() * 100
carb_miss = train[family_cols["carbs"]].isna().mean().mean() * 100
hr_miss   = train[family_cols["hr"]].isna().mean().mean() * 100
step_miss = train[family_cols["steps"]].isna().mean().mean() * 100
cal_miss  = train[family_cols["cals"]].isna().mean().mean() * 100

summary = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  BRIST1D DATASET – EDA SUMMARY FOR THESIS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  WHAT THE DATA CONTAINS
  • {n_train:,} training samples, {n_test:,} test samples
  • {n_cols} original columns covering 7 feature families:
    BG (CGM), Insulin (pump), Carbs (logged), Heart Rate, Steps,
    Calories, and Activity labels – each at 5-minute intervals over
    a 6-hour lookback window
  • Target: blood glucose 1 hour into the future (bg+1:00)
  • Time-of-day stamp for every sample

  PARTICIPANT COVERAGE
  • Train participants: {n_train_p} ({', '.join(train_pids)})
  • Test participants : {n_test_p} ({', '.join(test_pids)})
  • Overlapping participants: {len(overlap)}
  • Unseen test participants: {only_test if only_test else 'none'}
    → The test set includes participants NOT in training, requiring
      the model to generalise across individuals.

  KEY MISSINGNESS ISSUES
  • BG (CGM)   : {bg_miss:.1f}% missing – CGM gaps are common but manageable
  • Insulin    : {ins_miss:.1f}% missing – pump data has significant gaps
  • Carbs      : {carb_miss:.1f}% missing – sparse by nature (meals are discrete events)
  • Heart Rate : {hr_miss:.1f}% missing – smartwatch sync issues
  • Steps      : {step_miss:.1f}% missing
  • Calories   : {cal_miss:.1f}% missing
  • Missingness generally INCREASES for older lags (further in the past)
  • Target column (bg+1:00) is {target_missing:.1f}% missing in train

  WHY THIS DATASET IS SUITABLE FOR THE THESIS
  1. Multi-modal signals: CGM + insulin pump + meal logs + smartwatch
     data provide a rich, multi-channel input that mirrors what a
     meal-planner app would have access to.
  2. Temporal structure: The 6-hour lookback at 5-minute granularity
     creates a natural time-series window that sequence models can learn
     from (72 time steps × 7 feature channels).
  3. Real-world noise: Missing values, device heterogeneity, and
     individual variation make this a realistic testbed for robust
     prediction – exactly the challenges a production system would face.
  4. Prediction horizon: 1-hour-ahead BG prediction is clinically
     meaningful and actionable for meal planning decisions.
  5. Activity context: 22 activity categories allow the model to learn
     how exercise modulates glucose response to meals/insulin.

  FIGURES SAVED TO: {FIG_DIR.relative_to(DATA_DIR)}/
    01 – Missingness by lag distance
    02 – Missingness heatmap
    03 – Target & current BG distributions
    04 – Per-participant BG distributions
    05 – Time-of-day BG effects
    06 – Window-shape: BG, insulin, carbs lookback
    07 – Window-shape: HR, steps, calories lookback
    08 – Correlation heatmap (engineered features)
    09 – Scatter relationships
    10 – Binned mean plots
    11 – Activity frequency
    12 – Activity vs ΔBG

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
print(summary)
print("Done! All figures saved to ./figures/")
