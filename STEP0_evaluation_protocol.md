# STEP 0 — Evaluation Protocol
## Glucose Forecasting for Fuel Up: Thesis ML Pipeline

---

## 1. Target Definition

### 1.1 BrisT1D (Kaggle)
| Item | Value |
|---|---|
| **Target column** | `bg+1:00` |
| **Unit** | mmol/L |
| **Conversion** | mg/dL = mmol/L × 18.0182 (applied only for range-stratified analysis) |
| **Delta target** | `Δy = bg+1:00 − bg-0:00` (change from current glucose) |
| **Notes** | Pre-windowed dataset. Each row is already a 72-step × 7-channel sample. No additional windowing needed. |

### 1.2 HUPA-UCM (Mendeley)
| Item | Value |
|---|---|
| **Target column** | `glucose` at t+60 minutes |
| **Unit** | mg/dL (native) |
| **Delta target** | `Δy = glucose(t+60) − glucose(t)` |
| **Notes** | Raw 5-minute time series. Sliding-window construction required. Semicolon-delimited CSVs. |

### 1.3 CGMacros (PhysioNet)
| Item | Value |
|---|---|
| **Target column** | `Libre GL` |
| **Unit** | mg/dL |
| **Delta target** | `Δy = Libre_GL(t+60) − Libre_GL(t)` |
| **Justification** | Libre GL has **0.03% missingness** vs Dexcom GL at **8.40%** (verified across **all 45 participants**, n=687,580 rows). Libre provides ~280× better coverage. Worst-case participant (CGMacros-004) has only 1.5% Libre missingness, while Dexcom worst-case (CGMacros-028) reaches 24%. Dexcom GL is retained only for optional QA cross-checks (sensor agreement analysis). |
| **Notes** | Raw 5-minute series from 45 participants. Sliding-window construction required. |

### 1.4 Delta-Target Strategy
All models will be trained in **two modes** and the better one reported:
- **Direct mode**: predict `y = glucose(t+60)` directly.
- **Delta mode**: predict `Δy = glucose(t+60) − glucose(t)`, then reconstruct `ŷ = glucose(t) + Δŷ`.

> **Rationale**: Delta targets remove the dominant "current glucose" signal, forcing the model to learn the *dynamics* (meal response, insulin action, activity effects). Literature shows this often improves MAE by 5–15% for horizon ≥30 min.

---

## 2. Split Policy

### 2.1 BrisT1D
| Split | Policy |
|---|---|
| **Train + Val** | Official Kaggle train set, participants `{p01, p02, p03, p04, p05, p06, p10, p11, p12}` (9 participants, 177,024 rows) |
| **Validation strategy** | **GroupKFold (k=3)** over participants for hyperparameter tuning and model selection. Each fold holds out 3 participants, trains on 6. This avoids brittleness from a fixed 2-participant holdout and gives stable validation estimates across all 9 participants. |
| **Test** | Official Kaggle test set (15 participants, 7 unseen). Used **once** for final reported numbers. |
| **Leakage protection** | Participant-level GroupKFold. No row from a held-out participant appears in that fold's training set. |

> **Rationale**: With only 9 participants, a fixed holdout risks high-variance conclusions depending on which participants are held out. GroupKFold produces more reliable estimates and is defensible to a thesis committee.

### 2.2 HUPA-UCM
| Split | Policy |
|---|---|
| **Participants** | 25 total |
| **Train** | 17 participants (~68%) |
| **Validation** | 4 participants (~16%) |
| **Test** | 4 participants (~16%) |
| **Assignment** | Deterministic hash-based on participant ID (reproducible, no cherry-picking). |
| **Secondary check** | Windows may overlap temporally within a participant, which is acceptable. Each sample uses only past context `[t−360, t]` to predict `t+60`. No future values enter X. Overlap between adjacent windows shares some glucose history but does not constitute leakage because targets differ and no future information is used. |

### 2.3 CGMacros
| Split | Policy |
|---|---|
| **Participants** | 45 total |
| **Train** | 31 participants (~69%) |
| **Validation** | 7 participants (~16%) |
| **Test** | 7 participants (~15%) |
| **Assignment** | Deterministic hash-based on participant folder name. |
| **Optional day-block** | Within-participant day-block validation is available as secondary analysis (last 20% of days → local test). Not primary. |

### 2.4 Split Assignment Code (Pseudocode)
```python
import hashlib

def assign_split(pid: str, train_frac=0.68, val_frac=0.16):
    """Deterministic, reproducible participant-level split."""
    h = int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100
    if h < train_frac * 100:
        return "train"
    elif h < (train_frac + val_frac) * 100:
        return "val"
    else:
        return "test"
```

---

## 3. Metrics Suite

### 3.1 Primary Regression Metrics
| Metric | Formula | Notes |
|---|---|---|
| **MAE** | `mean(\|ŷ − y\|)` | Primary ranking metric. Robust to outliers. |
| **RMSE** | `sqrt(mean((ŷ − y)²))` | Penalizes large errors more. Important for safety-critical ranges. |
| **Delta MAE** | `mean(\|Δŷ − Δy\|)` | Only when using delta-target mode. Measures ability to predict *change*. |
| **Macro-Avg MAE** | `mean(MAE_hypo, MAE_normal, MAE_hyper)` | Equal weight to each glycemic range regardless of sample count. Highlights performance in rare but critical ranges. |

### 3.2 Range-Stratified Metrics (Clinical Relevance)
All glucose values converted to **mg/dL** for this analysis.

| Range | Label | Threshold | Clinical Significance |
|---|---|---|---|
| **Hypoglycemia** | `hypo` | < 70 mg/dL | Dangerous low; hardest to predict, rarest class |
| **Normal** | `normal` | 70–180 mg/dL | Time in Range (TIR); bulk of samples |
| **Hyperglycemia** | `hyper` | > 180 mg/dL | High glucose; post-meal spikes, poor control |

Report: `MAE_hypo`, `MAE_normal`, `MAE_hyper`, `RMSE_hypo`, `RMSE_normal`, `RMSE_hyper`, **sample count per range**, and **Macro-Avg MAE** (unweighted mean across the three ranges, giving equal importance to rare but clinically critical hypo events).

> For BrisT1D (mmol/L native): hypo < 3.9, normal 3.9–10.0, hyper > 10.0.

### 3.3 Uncertainty Metrics (for quantile / MC-Dropout models)
| Metric | Definition |
|---|---|
| **PICP** (Prediction Interval Coverage Probability) | Fraction of true values falling within the predicted interval (target: ≥ nominal, e.g., 90%) |
| **MPIW** (Mean Prediction Interval Width) | Average width of interval. Lower = sharper. |
| **Calibration curve** | Plot: nominal coverage (x) vs. observed coverage (y) for quantiles 10%, 20%, …, 90%. Perfect = diagonal. |
| **Interval Score** | Combined metric penalizing both miscoverage and interval width: `IS = MPIW + (2/α) × coverage_penalty`. Optional but useful as a single summary number. |
| **NLL** (Negative Log-Likelihood) | If model outputs a distribution (Gaussian), score with `−log p(y \| μ, σ)`. |

### 3.4 Diagnostic Plots (per model, per dataset)
1. **Predicted vs. Actual scatter** (with identity line)
2. **Residual histogram** (should be ~symmetric, zero-centered)
3. **Error vs. time-of-day** (detect dawn-phenomenon modeling failure)
4. **Error vs. glucose level** (detect heteroscedasticity)
5. **Worst-20 failure cases** (table + glucose trace plots showing context)
6. **Post-meal spike analysis**: MAE for samples within 30–120 min after a meal event

### 3.5 Time-of-Day Encoding Policy
| Dataset | Available temporal info | Encoding |
|---|---|---|
| BrisT1D | `time` column (HH:MM:SS, no date) | `sin(2π·hour/24)`, `cos(2π·hour/24)`, `sin(2π·minute/60)`, `cos(2π·minute/60)` |
| HUPA-UCM | Full ISO timestamps with dates | Same hour/minute encoding + optional `sin(2π·dow/7)`, `cos(2π·dow/7)` |
| CGMacros | Full timestamps with dates | Same as HUPA |

> Day-of-week encoding is included where actual dates are available. It captures weekly patterns (e.g., weekend eating habits) but is not expected to be a strong predictor.

---

## 4. Leakage Checklist

| # | Check | Status |
|---|---|---|
| 1 | **No participant overlap** between train/val/test | ☐ Verify after split |
| 2 | **Scaling statistics** (mean, std, median, IQR) computed on **train set only**, then applied to val/test | ☐ Implement in preprocessor |
| 3 | **No future data in features**: sliding window at time `t` uses only `[t−360, t]` for X and `t+60` for y | ☐ Assert in window builder |
| 4 | **BrisT1D lag columns**: verify `bg-5:55` is truly 5h55m *before* target, not *after* | ☐ Spot-check 5 rows |
| 5 | **Delta target uses only `glucose(t)` from the input window**, not from a separately-leaked column | ☐ Verify in code |
| 6 | **Time features** (sin/cos hour) derived from the window's timestamp `t`, not from any test-set metadata | ☐ Verify in code |
| 7 | **No data augmentation** that synthesizes test-participant-like samples | ☐ N/A (no augmentation planned) |
| 8 | **Cross-validation**: if used, folds are participant-level GroupKFold | ☐ Implement if needed |
| 9 | **Imputation**: any fill (forward-fill, interpolation) is **causal** — only uses past values within the window, never future | ☐ Verify in preprocessor |
| 10 | **Random seed** does not inadvertently fix test participants to "easy" ones | ☐ Hash-based split is seed-independent |

---

## 5. Scenario Feature Matrix

| Feature | Scenario 1 (Carbs) | Scenario 2 (Full Macros) | Scenario 3 (Macros + Activity) |
|---|---|---|---|
| Glucose history (72 steps) | ✅ | ✅ | ✅ |
| Carbs | ✅ | ✅ | ✅ |
| Insulin (BrisT1D/HUPA only) | ✅ (sub-baseline) | ✅ | ✅ |
| Protein | — | ✅ | ✅ |
| Fat | — | ✅ | ✅ |
| Fiber | — | ✅ | ✅ |
| Calories (meal) | — | ✅ | ✅ |
| Portion / Amount Consumed | — | ✅ | ✅ |
| Heart Rate | — | — | ✅ |
| Steps / METs | — | — | ✅ |
| Activity Calories | — | — | ✅ |
| sin/cos(hour), sin/cos(minute) | ✅ | ✅ | ✅ |
| Missingness mask channels | ✅ | ✅ | ✅ |

> **Note on Scenario 1 for T1D datasets**: Insulin (basal + bolus) is included as a sub-baseline because carbs-only without insulin is clinically unrealistic for T1D patients. CGMacros (non-T1D) does not have insulin, so Scenario 1 is purely glucose + carbs.

---

## 6. Model Lineup

| Model | Type | Purpose |
|---|---|---|
| **Persistence** | Baseline | `ŷ = glucose(t)`. Lower bound; any useful model must beat this. |
| **Linear Drift** | Baseline | `ŷ = glucose(t) + slope₆₀ × 60`, where `slope₆₀` is the linear trend over the last 60 min. Stronger than persistence; if deep models can't beat this, something is wrong. |
| **Linear Ridge** | Baseline | Flattened 72×C input → Ridge regression. Tests if temporal structure matters. |
| **GBDT (LightGBM)** | Baseline | Flattened features + engineered aggregates. Strong tabular baseline. |
| **GRU** | Deep | Sequence model. Standard RNN baseline for time series. |
| **Transformer Encoder** | Deep | Self-attention over 72 steps. Can handle irregular patterns + masking. |
| **Quantile GRU/Transformer** | Uncertainty | Predicts quantiles (10th, 50th, 90th). Thesis showpiece for uncertainty. |

---

## 7. Supervisor Confirmation Needed

> [!IMPORTANT]
> The following decisions require explicit approval before proceeding to Step 1.

1. **CGMacros target = Libre GL**: ✅ Confirmed across all 45 participants (0.03% vs 8.40% missing).\n2. **BrisT1D validation**: ✅ **GroupKFold (k=3)** over participants. Official test set used once at the end.\n3. **HUPA 17/4/4 split**: ✅ Hash-based deterministic assignment.\n4. **Delta-target dual training**: ✅ Train both direct and delta mode, report the better.\n5. **Insulin in Scenario 1**: ✅ Included for T1D datasets (BrisT1D/HUPA). Excluded for CGMacros.\n6. **Metrics priority**: ✅ MAE primary, RMSE secondary, Macro-Avg MAE for fairness across ranges.\n7. **Uncertainty models**: ✅ Quantile regression (10/50/90) primary. MC-Dropout secondary.\n8. **Minimum seeds**: ✅ 3 seeds per model per scenario.\n9. **Linear drift baseline**: ✅ Added as a stronger-than-persistence baseline.\n10. **Time-of-day encoding**: ✅ sin/cos(hour/minute) everywhere; day-of-week where dates available.\n\n**Status: ALL ITEMS APPROVED. Ready for Step 1.**

---

*Protocol version: 1.0 — Generated 2026-04-02*
