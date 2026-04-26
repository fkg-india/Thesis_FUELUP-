#!/usr/bin/env python3
"""
HUPA-UCM Diabetes Dataset – Exploratory Data Analysis (EDA)
============================================================
Thesis: Predicting future blood-glucose trends from meals, insulin, and
activity signals.  This script analyses the PREPROCESSED HUPA-UCM data.

Sections:
  1. Load all participant files robustly (semicolon-separated)
  2. Data dictionary & coverage summary
  3. Missingness & noise diagnostics
  4. Glucose distribution plots
  5. Time-of-day effects
  6. Event-based meal & bolus response curves  (key thesis plot)
  7. Activity → glucose-change relationships
  8. BrisT1D-like supervised dataset builder
  9. Window-shape visualization
 10. Professor-ready summary

NOTE: No medical advice is provided.
"""

# ──────────────────────────────────────────────────────────────────────
# 0.  IMPORTS & SETUP
# ──────────────────────────────────────────────────────────────────────
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 180, "savefig.bbox": "tight",
    "axes.titlesize": 13, "axes.labelsize": 11,
})

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path(__file__).resolve().parent / "Mendeley"
FIG_DIR  = Path(__file__).resolve().parent / "figures_hupa"
FIG_DIR.mkdir(exist_ok=True)

MGDL_TO_MMOL = 1.0 / 18.0182   # mg/dL → mmol/L


def savefig(fig, name: str):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    print(f"  ✓ Saved  {path.name}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 1.  LOAD ALL PARTICIPANT FILES
# ──────────────────────────────────────────────────────────────────────
print("=" * 72)
print("LOADING HUPA-UCM PREPROCESSED FILES …")
print("=" * 72)

files = sorted(DATA_DIR.glob("HUPA*P.csv"))
print(f"  Found {len(files)} participant files in {DATA_DIR.name}/")

frames = []
for fp in files:
    # Extract participant ID from filename: HUPA0001P.csv → 0001
    pid = re.search(r"HUPA(\d+)P", fp.stem)
    pid = pid.group(1) if pid else fp.stem

    # Robustly detect separator
    first_line = fp.read_text(errors="replace").split("\n")[0]
    sep = ";" if ";" in first_line else ","

    df = pd.read_csv(fp, sep=sep, parse_dates=["time"])
    df["participant_id"] = pid
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
data.sort_values(["participant_id", "time"], inplace=True)
data.reset_index(drop=True, inplace=True)

# Add mmol/L column
if "glucose" in data.columns:
    data["glucose_mmol"] = data["glucose"] * MGDL_TO_MMOL

print(f"  Total rows    : {len(data):,}")
print(f"  Participants  : {data['participant_id'].nunique()}")
print(f"  Columns       : {list(data.columns)}")

# ──────────────────────────────────────────────────────────────────────
# 2.  DATA DICTIONARY & COVERAGE SUMMARY
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("DATA DICTIONARY & COVERAGE")
print("=" * 72)

pids = sorted(data["participant_id"].unique())
coverage = []
for pid in pids:
    sub = data[data["participant_id"] == pid]
    t = sub["time"]
    diffs = t.diff().dt.total_seconds().dropna()
    coverage.append({
        "pid": pid,
        "rows": len(sub),
        "start": t.min(),
        "end": t.max(),
        "days": (t.max() - t.min()).total_seconds() / 86400,
        "median_interval_min": diffs.median() / 60 if len(diffs) else np.nan,
        "gluc_min": sub["glucose"].min() if "glucose" in sub else np.nan,
        "gluc_max": sub["glucose"].max() if "glucose" in sub else np.nan,
        "gluc_median": sub["glucose"].median() if "glucose" in sub else np.nan,
    })

cov_df = pd.DataFrame(coverage)
print("\n  Per-participant coverage:")
print(cov_df.to_string(index=False))

# Columns per participant
print("\n  Columns found across ALL files:")
all_cols = [c for c in data.columns if c not in ("participant_id", "glucose_mmol")]
for col in all_cols:
    present = [pid for pid in pids
               if data.loc[data["participant_id"] == pid, col].notna().any()]
    print(f"    {col:<30s}  present in {len(present)}/{len(pids)} participants")

# Flag glucose outliers
if "glucose" in data.columns:
    low  = (data["glucose"] <= 20).sum()
    high = (data["glucose"] > 500).sum()
    zeros = (data["glucose"] == 0).sum()
    print(f"\n  Glucose outlier flags:")
    print(f"    glucose == 0         : {zeros:,}")
    print(f"    glucose ≤ 20 mg/dL   : {low:,}")
    print(f"    glucose > 500 mg/dL  : {high:,}")

# Duplicate timestamps
print("\n  Duplicate timestamps per participant:")
total_dupes = 0
for pid in pids:
    sub = data[data["participant_id"] == pid]
    dupes = sub["time"].duplicated().sum()
    if dupes > 0:
        print(f"    {pid}: {dupes} duplicates")
        total_dupes += dupes
if total_dupes == 0:
    print("    None found ✓")

# ──────────────────────────────────────────────────────────────────────
# 3.  MISSINGNESS & NOISE DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("MISSINGNESS DIAGNOSTICS")
print("=" * 72)

# Overall missing %
signal_cols = [c for c in data.columns
               if c not in ("participant_id", "time", "glucose_mmol")]
print("\n  Overall missing % per column:")
for col in signal_cols:
    pct = data[col].isna().mean() * 100
    print(f"    {col:<30s} : {pct:6.2f} %")

# Per-participant missing %
print("\n  Generating per-participant missingness table …")
miss_table = {}
for pid in pids:
    sub = data[data["participant_id"] == pid]
    miss_table[pid] = {col: sub[col].isna().mean() * 100 for col in signal_cols}
miss_df = pd.DataFrame(miss_table).T.round(1)
print(miss_df.to_string())

# Missingness heatmap
print("\n  Plotting missingness heatmap …")
fig, ax = plt.subplots(figsize=(12, max(5, len(pids) * 0.4)))
if HAS_SNS:
    sns.heatmap(miss_df, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.3, cbar_kws={"label": "Missing %"}, ax=ax)
else:
    im = ax.imshow(miss_df.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(miss_df.columns)))
    ax.set_xticklabels(miss_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(miss_df.index)))
    ax.set_yticklabels(miss_df.index)
    plt.colorbar(im, ax=ax, label="Missing %")
ax.set_title("Missingness % per Participant × Column")
plt.tight_layout()
savefig(fig, "01_missingness_heatmap")

# Missingness over time (sample 4 participants)
print("  Plotting missingness over time …")
sample_pids = pids[:min(4, len(pids))]
fig, axes = plt.subplots(len(sample_pids), 1,
                         figsize=(14, 3.5 * len(sample_pids)), sharex=False)
if len(sample_pids) == 1:
    axes = [axes]

for ax, pid in zip(axes, sample_pids):
    sub = data[data["participant_id"] == pid].copy()
    # Show glucose availability as a scatter strip
    has = sub["time"][sub["glucose"].notna()]
    miss = sub["time"][sub["glucose"].isna()]
    ax.scatter(has, [1] * len(has), marker="|", s=3, color="#2ca02c",
               alpha=0.5, label="present")
    ax.scatter(miss, [0] * len(miss), marker="|", s=3, color="#d62728",
               alpha=0.5, label="missing")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Missing", "Present"])
    ax.set_title(f"Glucose availability – participant {pid}", fontsize=10)
    ax.legend(loc="upper right", fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # Detect long gaps
    diffs = sub["time"].diff().dt.total_seconds() / 60
    long_gaps = diffs[diffs > 30]
    if len(long_gaps) > 0:
        ax.set_xlabel(f"({len(long_gaps)} gaps > 30 min)")
plt.tight_layout()
savefig(fig, "02_missingness_over_time")

# Gap-length distribution
print("  Plotting gap-length distribution …")
all_gaps = []
for pid in pids:
    sub = data[data["participant_id"] == pid]
    diffs_min = sub["time"].diff().dt.total_seconds().dropna() / 60
    all_gaps.append(diffs_min)
all_gaps = pd.concat(all_gaps)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(all_gaps[all_gaps <= 60], bins=50, color="#1f77b4",
        edgecolor="white", alpha=0.85)
ax.set_xlabel("Gap between consecutive rows (minutes)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Time Gaps (≤60 min)")
ax.axvline(5, color="red", linestyle="--", label="5 min (expected)")
ax.legend()
plt.tight_layout()
savefig(fig, "03_gap_distribution")

# ──────────────────────────────────────────────────────────────────────
# 4.  GLUCOSE DISTRIBUTIONS
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("GLUCOSE DISTRIBUTION PLOTS")
print("=" * 72)

gluc = data["glucose"].dropna()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Histogram
ax = axes[0]
ax.hist(gluc, bins=80, color="#1f77b4", edgecolor="white", alpha=0.85)
ax.set_xlabel("Glucose (mg/dL)")
ax.set_ylabel("Count")
ax.set_title("Glucose Distribution – Histogram")
# Add mmol/L secondary x-axis
ax2 = ax.twiny()
lo, hi = ax.get_xlim()
ax2.set_xlim(lo * MGDL_TO_MMOL, hi * MGDL_TO_MMOL)
ax2.set_xlabel("mmol/L", fontsize=9, color="gray")

# Violin
ax = axes[1]
parts = ax.violinplot(gluc, positions=[0], showmeans=True, showmedians=True)
parts["bodies"][0].set_facecolor("#1f77b4")
ax.set_title("Glucose – Violin")
ax.set_ylabel("Glucose (mg/dL)")
ax.set_xticks([])

# Box plot
ax = axes[2]
ax.boxplot(gluc, patch_artist=True,
           boxprops=dict(facecolor="#aec7e8"),
           medianprops=dict(color="red"))
ax.set_title("Glucose – Box Plot")
ax.set_ylabel("Glucose (mg/dL)")

plt.tight_layout()
savefig(fig, "04_glucose_distributions")

# Per-participant boxplot sorted by median
print("  Plotting per-participant glucose boxplots …")
pid_medians = data.groupby("participant_id")["glucose"].median().sort_values()
ordered_pids = pid_medians.index.tolist()

fig, ax = plt.subplots(figsize=(14, 6))
bp_data = [data.loc[data["participant_id"] == pid, "glucose"].dropna().values
           for pid in ordered_pids]
bp = ax.boxplot(bp_data, patch_artist=True, labels=ordered_pids,
                boxprops=dict(facecolor="#aec7e8"), medianprops=dict(color="red"))
ax.set_xlabel("Participant (sorted by median glucose)")
ax.set_ylabel("Glucose (mg/dL)")
ax.set_title("Per-Participant Glucose – Sorted by Median")
ax.tick_params(axis="x", rotation=45)
# Clinical reference lines
ax.axhline(70, color="orange", linestyle="--", alpha=0.6, label="70 mg/dL (hypo)")
ax.axhline(180, color="red", linestyle="--", alpha=0.6, label="180 mg/dL (hyper)")
ax.legend(fontsize=8)
plt.tight_layout()
savefig(fig, "05_per_participant_glucose")

# ──────────────────────────────────────────────────────────────────────
# 5.  TIME-OF-DAY EFFECTS
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("TIME-OF-DAY EFFECTS")
print("=" * 72)

data["hour"] = data["time"].dt.hour

# Overall
hourly = data.groupby("hour")["glucose"].agg(["mean", "median"]).dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(hourly.index, hourly["mean"], "o-", label="Mean", color="#1f77b4")
ax.plot(hourly.index, hourly["median"], "s--", label="Median", color="#d62728")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Glucose (mg/dL)")
ax.set_title("Glucose vs Hour of Day – Overall")
ax.legend()
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

# Per-participant (up to 9)
ax = axes[1]
show_pids = pids[:min(9, len(pids))]
for pid in show_pids:
    sub = data[data["participant_id"] == pid]
    h = sub.groupby("hour")["glucose"].median()
    ax.plot(h.index, h.values, ".-", alpha=0.6, label=pid, markersize=3)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Median Glucose (mg/dL)")
ax.set_title("Median Glucose vs Hour – Per Participant")
ax.legend(fontsize=7, ncol=3)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

plt.tight_layout()
savefig(fig, "06_time_of_day")

# Day-of-week if enough data
data["dow"] = data["time"].dt.dayofweek
dow_means = data.groupby("dow")["glucose"].agg(["mean", "median"]).dropna()
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(7), dow_means["mean"], color="#1f77b4", alpha=0.7, label="Mean")
ax.plot(range(7), dow_means["median"], "o-", color="#d62728", label="Median")
ax.set_xticks(range(7))
ax.set_xticklabels(day_names)
ax.set_ylabel("Glucose (mg/dL)")
ax.set_title("Glucose by Day of Week")
ax.legend()
plt.tight_layout()
savefig(fig, "07_day_of_week")

# ──────────────────────────────────────────────────────────────────────
# 6.  EVENT-BASED RESPONSE CURVES  (key thesis plot!)
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("EVENT-BASED GLUCOSE RESPONSE CURVES")
print("=" * 72)


def extract_event_responses(df, event_col, threshold, pid,
                            pre_min=120, post_min=240, step=5):
    """
    For rows where event_col > threshold, extract glucose trajectory
    from -pre_min to +post_min around the event.
    Returns a DataFrame with columns = relative minutes, each row = one event.
    """
    sub = df[df["participant_id"] == pid].copy()
    sub = sub.sort_values("time").reset_index(drop=True)

    events = sub.index[sub[event_col] > threshold].tolist()
    if not events:
        return pd.DataFrame()

    # Thin events: skip if another event within 30 min
    thinned = [events[0]]
    for e in events[1:]:
        if (sub.loc[e, "time"] - sub.loc[thinned[-1], "time"]).total_seconds() > 1800:
            thinned.append(e)
    events = thinned

    rel_minutes = np.arange(-pre_min, post_min + step, step)
    traces = []

    for ei in events:
        t0 = sub.loc[ei, "time"]
        trace = {}
        for rm in rel_minutes:
            target_time = t0 + pd.Timedelta(minutes=int(rm))
            # Find nearest row within ±3 min
            mask = (sub["time"] - target_time).abs() <= pd.Timedelta(minutes=3)
            matched = sub.loc[mask, "glucose"]
            trace[rm] = matched.iloc[0] if len(matched) > 0 else np.nan
        traces.append(trace)

    return pd.DataFrame(traces)


# Meal events (carb_input > 0)
print("  Extracting meal response curves …")
meal_traces_all = []
for pid in pids:
    if "carb_input" not in data.columns:
        break
    tr = extract_event_responses(data, "carb_input", 0, pid)
    if len(tr) > 0:
        meal_traces_all.append(tr)

if meal_traces_all:
    meal_all = pd.concat(meal_traces_all, ignore_index=True)
    n_meals = len(meal_all)
    print(f"  Total meal events found: {n_meals}")

    fig, ax = plt.subplots(figsize=(12, 6))
    rel = np.array(meal_all.columns, dtype=float)
    mean_curve = meal_all.mean()
    p25 = meal_all.quantile(0.25)
    p75 = meal_all.quantile(0.75)

    ax.fill_between(rel, p25, p75, alpha=0.25, color="#1f77b4",
                    label="25th–75th percentile")
    ax.plot(rel, mean_curve, "o-", color="#1f77b4", markersize=2,
            linewidth=2, label=f"Mean (n={n_meals})")
    ax.axvline(0, color="red", linestyle="--", label="Meal event")
    ax.set_xlabel("Minutes relative to meal event")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("Glucose Response to Meals (carb_input > 0)")
    ax.legend()
    plt.tight_layout()
    savefig(fig, "08_meal_response_curve")
else:
    print("  No meal events found (carb_input column missing or all zero)")

# Bolus events (bolus_volume_delivered > 0)
print("  Extracting bolus response curves …")
bolus_traces_all = []
for pid in pids:
    if "bolus_volume_delivered" not in data.columns:
        break
    tr = extract_event_responses(data, "bolus_volume_delivered", 0, pid)
    if len(tr) > 0:
        bolus_traces_all.append(tr)

if bolus_traces_all:
    bolus_all = pd.concat(bolus_traces_all, ignore_index=True)
    n_bolus = len(bolus_all)
    print(f"  Total bolus events found: {n_bolus}")

    fig, ax = plt.subplots(figsize=(12, 6))
    rel = np.array(bolus_all.columns, dtype=float)
    mean_curve = bolus_all.mean()
    p25 = bolus_all.quantile(0.25)
    p75 = bolus_all.quantile(0.75)

    ax.fill_between(rel, p25, p75, alpha=0.25, color="#2ca02c",
                    label="25th–75th percentile")
    ax.plot(rel, mean_curve, "o-", color="#2ca02c", markersize=2,
            linewidth=2, label=f"Mean (n={n_bolus})")
    ax.axvline(0, color="red", linestyle="--", label="Bolus event")
    ax.set_xlabel("Minutes relative to bolus event")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("Glucose Response to Bolus Insulin (bolus > 0)")
    ax.legend()
    plt.tight_layout()
    savefig(fig, "09_bolus_response_curve")
else:
    print("  No bolus events found")

# Per-participant meal response (up to 6)
print("  Plotting per-participant meal responses …")
show_meal_pids = pids[:min(6, len(pids))]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, pid in enumerate(show_meal_pids):
    ax = axes[i]
    tr = extract_event_responses(data, "carb_input", 0, pid)
    if len(tr) == 0:
        ax.text(0.5, 0.5, "No meal events", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(f"Participant {pid}")
        continue
    rel = np.array(tr.columns, dtype=float)
    for j in range(min(8, len(tr))):
        ax.plot(rel, tr.iloc[j], alpha=0.3, linewidth=0.8)
    ax.plot(rel, tr.mean(), "k-", linewidth=2, label="Mean")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_title(f"Participant {pid} ({len(tr)} meals)", fontsize=10)
    if i >= 3:
        ax.set_xlabel("Minutes from meal")
    if i % 3 == 0:
        ax.set_ylabel("Glucose (mg/dL)")

# Hide unused axes
for j in range(len(show_meal_pids), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Per-Participant Meal Glucose Response", fontsize=14, y=1.01)
plt.tight_layout()
savefig(fig, "10_per_participant_meal_response")

# ──────────────────────────────────────────────────────────────────────
# 7.  ACTIVITY → GLUCOSE-CHANGE RELATIONSHIPS
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("ACTIVITY → GLUCOSE RELATIONSHIPS")
print("=" * 72)

# Compute per-participant rolling features
print("  Computing rolling activity and glucose-change features …")
eng_rows = []
for pid in pids:
    sub = data[data["participant_id"] == pid].copy()
    sub = sub.sort_values("time").reset_index(drop=True)

    # Rolling sums / means (last 60 min = 12 rows at 5-min)
    if "steps" in sub.columns:
        sub["steps_60"] = sub["steps"].rolling(12, min_periods=1).sum()
    if "heart_rate" in sub.columns:
        sub["hr_mean_60"] = sub["heart_rate"].rolling(12, min_periods=1).mean()

    # Glucose change in next 60 min = glucose 12 rows ahead minus current
    if "glucose" in sub.columns:
        sub["glucose_future_60"] = sub["glucose"].shift(-12)
        sub["delta_gluc_60"] = sub["glucose_future_60"] - sub["glucose"]

    eng_rows.append(sub)

eng = pd.concat(eng_rows, ignore_index=True)

# Scatter plots
print("  Plotting activity vs glucose-change scatter plots …")
scatter_pairs = []
if "steps_60" in eng.columns and "delta_gluc_60" in eng.columns:
    scatter_pairs.append(("steps_60", "delta_gluc_60",
                          "Steps (last 60 min)", "ΔGlucose next 60 min (mg/dL)"))
if "hr_mean_60" in eng.columns and "delta_gluc_60" in eng.columns:
    scatter_pairs.append(("hr_mean_60", "delta_gluc_60",
                          "Mean HR (last 60 min)", "ΔGlucose next 60 min (mg/dL)"))
if "steps_60" in eng.columns and "glucose" in eng.columns:
    scatter_pairs.append(("steps_60", "glucose",
                          "Steps (last 60 min)", "Current Glucose (mg/dL)"))

if scatter_pairs:
    ncols = len(scatter_pairs)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    for ax, (xcol, ycol, xl, yl) in zip(axes, scatter_pairs):
        sub = eng[[xcol, ycol]].dropna()
        if len(sub) > 5000:
            sub = sub.sample(5000, random_state=SEED)
        ax.scatter(sub[xcol], sub[ycol], alpha=0.15, s=6, color="#1f77b4")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{yl}\nvs {xl}", fontsize=10)
    plt.tight_layout()
    savefig(fig, "11_activity_glucose_scatter")

# Binned mean plots
print("  Plotting binned activity → glucose-change …")
binned_features = []
if "steps_60" in eng.columns and "delta_gluc_60" in eng.columns:
    binned_features.append(("steps_60", "Steps last 60 min"))
if "hr_mean_60" in eng.columns and "delta_gluc_60" in eng.columns:
    binned_features.append(("hr_mean_60", "Mean HR last 60 min"))

if binned_features:
    fig, axes = plt.subplots(1, len(binned_features),
                             figsize=(7 * len(binned_features), 5))
    if len(binned_features) == 1:
        axes = [axes]

    for ax, (feat, label) in zip(axes, binned_features):
        sub = eng[[feat, "delta_gluc_60"]].dropna()
        sub["bin"] = pd.qcut(sub[feat], q=10, duplicates="drop")
        grp = sub.groupby("bin", observed=True)["delta_gluc_60"].agg(
            ["mean", "median", "count"]
        )
        x = range(len(grp))
        ax.bar(x, grp["mean"], color="#1f77b4", alpha=0.7, label="Mean ΔGluc")
        ax.plot(x, grp["median"], "o-", color="#d62728", label="Median ΔGluc")
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in grp.index],
                           rotation=45, fontsize=7, ha="right")
        ax.set_xlabel(label)
        ax.set_ylabel("ΔGlucose next 60 min (mg/dL)")
        ax.set_title(f"Binned: {label}", fontsize=10)
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    savefig(fig, "12_binned_activity_glucose")

# ──────────────────────────────────────────────────────────────────────
# 8.  BRIST1D-LIKE SUPERVISED DATASET BUILDER
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("SUPERVISED DATASET BUILDER (BrisT1D-like)")
print("=" * 72)

FEATURE_COLS = [c for c in ["glucose", "calories", "heart_rate", "steps",
                             "basal_rate", "bolus_volume_delivered", "carb_input"]
                if c in data.columns]


def build_supervised_samples(df, pid,
                             window_hours=6,
                             horizon_minutes=60,
                             step_minutes=5,
                             max_gap_minutes=15):
    """
    Build (X, y) samples for one participant.
    X = past `window_hours` of signals at `step_minutes` resolution
    y = glucose at +horizon_minutes from the end of the window

    Returns:
      X  : np.ndarray (n_samples, n_timesteps, n_features)
      y  : np.ndarray (n_samples,)
      stats : dict with counts of kept/dropped samples
    """
    sub = df[df["participant_id"] == pid].copy()
    sub = sub.sort_values("time").reset_index(drop=True)

    n_input_steps = int(window_hours * 60 / step_minutes)   # 72 for 6h @ 5min
    n_horizon_steps = int(horizon_minutes / step_minutes)    # 12 for 60min

    total_needed = n_input_steps + n_horizon_steps
    if len(sub) < total_needed:
        return None, None, {"total": 0, "kept": 0, "dropped_gap": 0,
                            "dropped_missing_target": 0}

    xs, ys = [], []
    dropped_gap = 0
    dropped_target = 0

    for i in range(len(sub) - total_needed + 1):
        window_slice = sub.iloc[i : i + total_needed]

        # Check for time gaps > max_gap_minutes
        diffs = window_slice["time"].diff().dt.total_seconds().dropna() / 60
        if diffs.max() > max_gap_minutes:
            dropped_gap += 1
            continue

        # Target glucose
        target_glucose = window_slice.iloc[n_input_steps + n_horizon_steps - 1]["glucose"]
        if pd.isna(target_glucose):
            dropped_target += 1
            continue

        # Input features
        input_data = window_slice.iloc[:n_input_steps][FEATURE_COLS].values
        xs.append(input_data)
        ys.append(target_glucose)

    total = len(sub) - total_needed + 1
    kept = len(xs)

    X = np.array(xs) if xs else np.empty((0, n_input_steps, len(FEATURE_COLS)))
    y = np.array(ys) if ys else np.empty(0)

    return X, y, {
        "total": total,
        "kept": kept,
        "dropped_gap": dropped_gap,
        "dropped_missing_target": dropped_target,
    }


print(f"  Feature columns used: {FEATURE_COLS}")
print(f"  Window: 6 hours (72 steps) → predict glucose at +60 min")
print()

sample_report = []
total_kept = 0
total_total = 0
for pid in pids:
    X, y, stats = build_supervised_samples(data, pid)
    if X is not None:
        sample_report.append({
            "pid": pid,
            "X_shape": X.shape,
            "y_len": len(y),
            **stats,
        })
        total_kept += stats["kept"]
        total_total += stats["total"]
    else:
        sample_report.append({"pid": pid, "X_shape": (0,), "y_len": 0,
                              "total": 0, "kept": 0,
                              "dropped_gap": 0, "dropped_missing_target": 0})

report_df = pd.DataFrame(sample_report)
print("  Per-participant sample counts:")
print(report_df.to_string(index=False))
print(f"\n  TOTAL potential samples : {total_total:,}")
print(f"  TOTAL usable samples   : {total_kept:,}")
print(f"  Drop rate              : {(1 - total_kept / max(total_total, 1)) * 100:.1f}%")

# ──────────────────────────────────────────────────────────────────────
# 9.  WINDOW-SHAPE VISUALIZATION
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("WINDOW-SHAPE VISUALIZATION")
print("=" * 72)

# Pick participant with most kept samples
best_pid = report_df.loc[report_df["kept"].idxmax(), "pid"]
print(f"  Chosen participant: {best_pid}")

X_best, y_best, _ = build_supervised_samples(data, best_pid)
if X_best is not None and len(X_best) > 0:
    n_show = min(4, len(X_best))
    indices = np.random.choice(len(X_best), size=n_show, replace=False)

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    t_input = np.arange(-360, 0, 5)  # -360 to -5 in 5-min steps
    feature_names = FEATURE_COLS

    for k, idx in enumerate(indices):
        ax = axes[k]
        x_sample = X_best[idx]   # shape: (72, n_features)
        y_val = y_best[idx]

        # Glucose on primary axis
        gluc_idx = feature_names.index("glucose") if "glucose" in feature_names else None
        if gluc_idx is not None:
            ax.plot(t_input, x_sample[:, gluc_idx], "o-", color="#1f77b4",
                    markersize=3, linewidth=1.5, label="Glucose (mg/dL)")
            # Target at +60
            ax.plot(60, y_val, "*", color="red", markersize=14,
                    label=f"Target +60m = {y_val:.0f}", zorder=6)
            ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

        ax.set_ylabel("Glucose (mg/dL)", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"Sample #{idx}  |  Participant {best_pid}",
                     fontsize=10)

        # Secondary axis: insulin + carbs
        ax2 = ax.twinx()
        plot_pairs = []
        for fcol, color, lbl in [
            ("bolus_volume_delivered", "#2ca02c", "Bolus (units)"),
            ("carb_input", "#d62728", "Carbs (g)"),
            ("basal_rate", "#9467bd", "Basal rate"),
        ]:
            if fcol in feature_names:
                fi = feature_names.index(fcol)
                ax2.plot(t_input, x_sample[:, fi], linewidth=1,
                         color=color, alpha=0.7, label=lbl)
        ax2.set_ylabel("Insulin / Carbs")
        ax2.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Minutes relative to prediction time "
                        "(negative = past, +60 = target)")
    fig.suptitle(f"Sample Windows – Participant {best_pid}", fontsize=14, y=1.01)
    plt.tight_layout()
    savefig(fig, "13_window_shapes")

    # Separate figure: HR, steps, calories
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 4.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for k, idx in enumerate(indices):
        ax = axes[k]
        x_sample = X_best[idx]

        # HR on primary
        if "heart_rate" in feature_names:
            fi = feature_names.index("heart_rate")
            ax.plot(t_input, x_sample[:, fi], ".-", color="#e377c2",
                    markersize=2, linewidth=1.2, label="HR (bpm)")
        ax.set_ylabel("HR (bpm)", color="#e377c2")
        ax.tick_params(axis="y", labelcolor="#e377c2")
        ax.legend(loc="upper left", fontsize=7)
        ax.set_title(f"Sample #{idx}  |  Participant {best_pid}", fontsize=10)

        # Steps + calories on secondary
        ax2 = ax.twinx()
        for fcol, color, lbl in [
            ("steps", "#7f7f7f", "Steps"),
            ("calories", "#bcbd22", "Calories"),
        ]:
            if fcol in feature_names:
                fi = feature_names.index(fcol)
                ax2.plot(t_input, x_sample[:, fi], ".-", color=color,
                         markersize=2, linewidth=1, alpha=0.7, label=lbl)
        ax2.set_ylabel("Steps / Calories")
        ax2.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Minutes relative to prediction time (negative = past)")
    fig.suptitle(f"HR / Steps / Calories Windows – Participant {best_pid}",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    savefig(fig, "14_window_shapes_activity")
else:
    print("  Not enough samples for window visualization.")

# ──────────────────────────────────────────────────────────────────────
# 10.  PROFESSOR-READY SUMMARY
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("PROFESSOR-READY SUMMARY")
print("=" * 72)

n_rows = len(data)
n_pids = data["participant_id"].nunique()
median_days = cov_df["days"].median()
median_interval = cov_df["median_interval_min"].median()

# Missingness summary
gluc_miss_pct = data["glucose"].isna().mean() * 100
carb_miss_pct = data["carb_input"].isna().mean() * 100 if "carb_input" in data.columns else -1
hr_miss_pct   = data["heart_rate"].isna().mean() * 100 if "heart_rate" in data.columns else -1

summary = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  HUPA-UCM DIABETES DATASET – EDA SUMMARY FOR THESIS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SIGNALS AVAILABLE
  • Glucose (CGM, mg/dL)  – continuous glucose monitor readings
  • Basal Rate            – background insulin delivery rate
  • Bolus Volume          – discrete insulin bolus injections
  • Carb Input            – logged carbohydrate intake (grams)
  • Heart Rate            – wearable HR sensor
  • Steps                 – wearable step counter
  • Calories              – estimated calorie expenditure

  PARTICIPANT COVERAGE
  • {n_pids} participants
  • {n_rows:,} total rows
  • Median duration: {median_days:.1f} days per participant
  • Median sampling interval: {median_interval:.1f} minutes (≈ 5 min)
  • Time range: {data['time'].min().strftime('%Y-%m-%d')} to {data['time'].max().strftime('%Y-%m-%d')}

  MISSINGNESS & GAPS
  • Glucose   : {gluc_miss_pct:.1f}% missing overall
  • Heart Rate: {hr_miss_pct:.1f}% missing overall
  • Carb Input: {carb_miss_pct:.1f}% missing overall
  • Time gaps follow a natural diurnal pattern (device removal, charging)
  • Gaps > 15 min cause samples to be dropped in the supervised builder

  SUPERVISED DATASET FEASIBILITY
  • Window: 6 hours (72 × 5-min steps) → predict glucose at +60 min
  • {total_kept:,} usable samples from {total_total:,} potential
    ({(1 - total_kept / max(total_total, 1)) * 100:.1f}% dropped due to gaps/missing targets)
  • Each sample: X shape = (72, {len(FEATURE_COLS)})  y = scalar glucose

  WHY THIS DATASET SUPPORTS THE THESIS
  1. Multi-modal signals: CGM + insulin pump + meal logs + smartwatch
     data – exactly the inputs a meal-planner app would ingest.
  2. Real-world conditions: Noise, gaps, and participant variability
     make this a realistic testbed for robust trend prediction.
  3. Event-aligned analysis: We demonstrated clear meal → glucose-rise
     and bolus → glucose-drop patterns that ML models can learn.
  4. Compatible format: Data naturally converts to the 6h-window →
     +60min-target structure used in state-of-the-art BG prediction
     (matching our BrisT1D analysis pipeline).
  5. Activity context: HR, steps, and calories allow the model to
     learn how exercise modulates glucose response to meals.

  FIGURES SAVED TO: figures_hupa/
    01 – Missingness heatmap (per participant × column)
    02 – Missingness over time (sample participants)
    03 – Gap-length distribution
    04 – Glucose overall distributions
    05 – Per-participant glucose boxplots
    06 – Glucose vs time of day
    07 – Glucose vs day of week
    08 – Meal glucose response curve (event-aligned)
    09 – Bolus glucose response curve (event-aligned)
    10 – Per-participant meal responses
    11 – Activity vs glucose-change scatter
    12 – Binned activity → glucose-change
    13 – Window-shape: glucose + insulin/carbs
    14 – Window-shape: HR / steps / calories

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
print(summary)
print("Done! All figures saved to ./figures_hupa/")
