"""
Fuel Up+ Interactive Demo — Backend Server
===========================================
Flask server that loads trained PyTorch models and serves glucose predictions.
Provides sample participant data and real-time inference with uncertainty bands.

Usage:
    python demo_server.py
    → Open http://localhost:5050 in your browser
"""

import os, sys, json, copy
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_from_directory

# ── Import model architectures from existing code ──
sys.path.insert(0, os.path.dirname(__file__))
from deep_models import GRUModel, TransformerModel, WINDOW
from baselines import (
    load_hupa_fast, load_brist1d_fast, load_cgmacros_fast,
    split_data, assign_split
)

app = Flask(__name__, static_folder="demo_static")

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
CKPT_DIR = os.path.join(os.path.dirname(__file__), "figures_step3", "checkpoints")
DEVICE = torch.device("cpu")   # CPU for demo (instant, no MPS issues)
STEP_MIN = 5

# Pre-loaded models cache
_models = {}
_sample_data = {}

# ══════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════

def load_model(model_type, dataset, scenario, mode):
    """Load a saved checkpoint. Returns (model, input_dim)."""
    key = f"{model_type}_{mode}_{dataset}_s{scenario}"
    if key in _models:
        return _models[key]

    ckpt_path = os.path.join(CKPT_DIR, f"{model_type}_{mode}_{dataset}_s{scenario}.pt")
    if not os.path.exists(ckpt_path):
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    input_dim = ckpt["input_dim"]
    hp = ckpt["hp"]

    if model_type == "GRU":
        model = GRUModel(
            input_dim=input_dim,
            hidden_dim=hp["hidden_dim"],
            n_layers=hp["n_layers"],
            dropout=hp["dropout"]
        )
    else:
        nhead = max(1, min(4, hp["hidden_dim"] // 16))
        model = TransformerModel(
            input_dim=input_dim,
            hidden_dim=hp["hidden_dim"],
            n_layers=hp["n_layers"],
            dropout=hp["dropout"],
            nhead=nhead
        )

    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    _models[key] = (model, input_dim)
    return model, input_dim


def predict_with_uncertainty(model, x_tensor, K=20):
    """
    Run MC-Dropout inference for uncertainty estimation.
    Returns (mean_pred, q10, q90).
    """
    model.train()  # Enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(K):
            p = model(x_tensor).cpu().numpy()
            preds.append(p)
    preds = np.array(preds)  # (K, batch)
    
    mean_pred = np.mean(preds, axis=0)
    q10 = np.percentile(preds, 10, axis=0)
    q90 = np.percentile(preds, 90, axis=0)
    model.eval()
    return mean_pred, q10, q90


# ══════════════════════════════════════════════════════════════════════
#  PREPARE SAMPLE DATA
# ══════════════════════════════════════════════════════════════════════

def prepare_sample_participants():
    """Load a few real participant windows for the demo."""
    global _sample_data
    print("Loading sample participant data for demo...")

    datasets_to_load = [
        ("hupa", 1, load_hupa_fast),
        ("brist1d", 1, load_brist1d_fast),
        ("cgmacros", 1, load_cgmacros_fast),
    ]

    for ds_name, scenario, loader_fn in datasets_to_load:
        try:
            data = loader_fn(scenario=scenario)
            splits = split_data(data)
            test = splits.get("test", splits.get("val"))

            # Pick interesting samples (meals with glucose spikes)
            gl = test["gl"]
            y_abs = test["y_abs"]
            pids = test["pids"]
            
            # Get unique participants
            unique_pids = np.unique(pids)
            
            samples = []
            for pid in unique_pids[:3]:  # Max 3 participants
                pid_mask = pids == pid
                pid_gl = gl[pid_mask]
                pid_y = y_abs[pid_mask]
                
                # Pick 5 diverse samples: low, medium, high glucose, spike, dip
                n = pid_mask.sum()
                if n < 5:
                    indices = np.arange(n)
                else:
                    # Get samples at different glucose levels
                    sorted_idx = np.argsort(pid_y)
                    indices = [
                        sorted_idx[0],                    # lowest
                        sorted_idx[n // 4],               # Q1
                        sorted_idx[n // 2],               # median
                        sorted_idx[3 * n // 4],           # Q3
                        sorted_idx[-1],                   # highest
                    ]
                
                for i, idx in enumerate(indices):
                    trace = pid_gl[idx]
                    # Replace NaN with interpolation
                    trace_clean = trace.copy()
                    nans = np.isnan(trace_clean)
                    if nans.any():
                        not_nan = ~nans
                        if not_nan.any():
                            trace_clean[nans] = np.interp(
                                np.flatnonzero(nans),
                                np.flatnonzero(not_nan),
                                trace_clean[not_nan]
                            )
                    
                    unit = "mmol/L" if ds_name == "brist1d" else "mg/dL"
                    # Convert for display
                    display_trace = trace_clean.tolist()
                    display_target = float(pid_y[idx])
                    
                    samples.append({
                        "id": f"{pid}_sample{i}",
                        "pid": str(pid),
                        "dataset": ds_name,
                        "unit": unit,
                        "glucose_history": display_trace,
                        "actual_future": display_target,
                        "label": f"{pid} — Sample {i+1}"
                    })
            
            _sample_data[ds_name] = {
                "name": ds_name.upper(),
                "unit": "mmol/L" if ds_name == "brist1d" else "mg/dL",
                "samples": samples
            }
            del data
            print(f"  {ds_name}: loaded {len(samples)} sample windows")
        except Exception as e:
            print(f"  Warning: Could not load {ds_name}: {e}")
    
    print(f"Sample data ready: {list(_sample_data.keys())}")


# ══════════════════════════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("demo_static", "index.html")


@app.route("/manual")
def manual():
    return send_from_directory("demo_static", "manual.html")


@app.route("/api/datasets")
def get_datasets():
    """Return available datasets and sample participants."""
    return jsonify(_sample_data)


@app.route("/api/analyze", methods=["POST"])
def analyze_glucose():
    """
    Analyze a glucose trajectory and return human-readable insights.
    Examines: trend, rate of change, zone, volatility, likely causes.
    """
    data = request.json
    gl = np.array(data["glucose_history"], dtype=np.float32)
    unit = data.get("unit", "mg/dL")
    prediction = data.get("prediction")  # may be None
    q10 = data.get("q10")
    q90 = data.get("q90")
    safety = data.get("safety")
    actual = data.get("actual")

    # Convert thresholds
    is_mmol = unit == "mmol/L"
    hypo_thresh = 3.9 if is_mmol else 70
    hyper_thresh = 10.0 if is_mmol else 180
    normal_high = 7.8 if is_mmol else 140
    uname = "mmol/L" if is_mmol else "mg/dL"

    # --- Current state ---
    current = float(gl[-1])
    if current < hypo_thresh:
        zone = "hypoglycaemic"
        zone_emoji = "🔴"
        zone_desc = f"dangerously LOW at {current:.1f} {uname} (below {hypo_thresh} {uname}). Risk of dizziness, confusion, and fainting."
    elif current <= normal_high:
        zone = "normal"
        zone_emoji = "🟢"
        zone_desc = f"in the NORMAL range at {current:.1f} {uname} (between {hypo_thresh}–{normal_high} {uname}). This is the healthy target zone."
    elif current <= hyper_thresh:
        zone = "elevated"
        zone_emoji = "🟡"
        zone_desc = f"ELEVATED at {current:.1f} {uname} (between {normal_high}–{hyper_thresh} {uname}). Commonly seen 1–2 hours after eating, especially carb-heavy meals."
    else:
        zone = "hyperglycaemic"
        zone_emoji = "🔴"
        zone_desc = f"in the HYPERGLYCAEMIC zone at {current:.1f} {uname} (above {hyper_thresh} {uname}). If sustained, this causes long-term organ damage."

    # --- Trend analysis (last 30 min = 6 readings) ---
    recent = gl[-6:]
    slope_30 = float(np.polyfit(np.arange(6), recent, 1)[0])  # per 5-min step
    rate_per_hour = slope_30 * 12  # per hour

    if abs(rate_per_hour) < (0.3 if is_mmol else 5):
        trend = "stable"
        trend_emoji = "➡️"
        trend_desc = f"relatively STABLE over the last 30 minutes (changing ~{abs(rate_per_hour):.1f} {uname}/hr)."
    elif rate_per_hour > 0:
        fast = rate_per_hour > (1.5 if is_mmol else 25)
        trend = "rising_fast" if fast else "rising"
        trend_emoji = "📈" if not fast else "🚀"
        trend_desc = f"RISING {'rapidly ' if fast else ''}at ~{rate_per_hour:.1f} {uname}/hr over the last 30 min."
    else:
        fast = rate_per_hour < (-1.5 if is_mmol else -25)
        trend = "falling_fast" if fast else "falling"
        trend_emoji = "📉" if not fast else "⚠️"
        trend_desc = f"FALLING {'rapidly ' if fast else ''}at ~{abs(rate_per_hour):.1f} {uname}/hr over the last 30 min."

    # --- Volatility (glucose variability) ---
    std_6h = float(np.nanstd(gl))
    cv = (std_6h / max(float(np.nanmean(gl)), 1)) * 100  # coefficient of variation
    if cv < 15:
        volatility = "low"
        vol_desc = f"Glucose has been very STABLE over 6 hours (CV={cv:.0f}%, std={std_6h:.1f} {uname})."
    elif cv < 30:
        volatility = "moderate"
        vol_desc = f"Moderate glucose VARIABILITY over 6 hours (CV={cv:.0f}%, std={std_6h:.1f} {uname}). Some fluctuation is normal after meals."
    else:
        volatility = "high"
        vol_desc = f"HIGH glucose variability (CV={cv:.0f}%, std={std_6h:.1f} {uname}). This indicates glucose is swinging significantly — common with large meals or insulin timing issues."

    # --- Detect spikes/dips in the 6h window ---
    events = []
    # Look for glucose peaks (local maxima > 30 min window)
    for i in range(12, len(gl) - 6, 6):
        window = gl[max(0,i-6):min(len(gl),i+6)]
        if gl[i] == np.nanmax(window):
            rise = float(gl[i] - gl[max(0,i-12)])
            thresh = 1.5 if is_mmol else 25
            if rise > thresh:
                mins_ago = (len(gl) - 1 - i) * 5
                events.append({
                    "type": "spike",
                    "time_ago": mins_ago,
                    "value": float(gl[i]),
                    "rise": rise,
                    "desc": f"Glucose SPIKE detected ~{mins_ago} min ago — rose by {rise:.1f} {uname} to {gl[i]:.1f} {uname}."
                })
        if gl[i] == np.nanmin(window):
            drop = float(gl[max(0,i-12)] - gl[i])
            thresh = 1.5 if is_mmol else 25
            if drop > thresh:
                mins_ago = (len(gl) - 1 - i) * 5
                events.append({
                    "type": "dip",
                    "time_ago": mins_ago,
                    "value": float(gl[i]),
                    "drop": drop,
                    "desc": f"Glucose DIP detected ~{mins_ago} min ago — dropped by {drop:.1f} {uname} to {gl[i]:.1f} {uname}."
                })

    # --- Likely cause analysis ---
    causes = []

    # Spike causes
    if any(e["type"] == "spike" for e in events):
        spike = next(e for e in events if e["type"] == "spike")
        if spike["time_ago"] < 120:
            causes.append({
                "factor": "Recent Meal (Carbohydrates)",
                "emoji": "🍞",
                "confidence": "High",
                "explanation": f"A glucose spike ~{spike['time_ago']} min ago strongly suggests a recent meal. Carbohydrates cause blood sugar to peak 30–90 minutes after eating. Higher carb meals (pasta, rice, bread) produce larger spikes."
            })
        if spike["time_ago"] >= 120:
            causes.append({
                "factor": "Delayed Carb Absorption",
                "emoji": "🍕",
                "confidence": "Medium",
                "explanation": f"The spike ~{spike['time_ago']} min ago may be from slow-digesting foods. High-fat meals (pizza, fried food) delay carb absorption, causing a later, prolonged glucose rise."
            })

    # Dip causes
    if any(e["type"] == "dip" for e in events):
        causes.append({
            "factor": "Insulin Action",
            "emoji": "💉",
            "confidence": "High",
            "explanation": "A glucose dip is typically caused by insulin action. Rapid-acting insulin peaks 60–90 min after injection, pulling glucose down. If the patient took too much insulin relative to their meal, this causes hypoglycaemia."
        })

    # Rising trend causes
    if trend.startswith("rising"):
        if current > normal_high:
            causes.append({
                "factor": "Insufficient Insulin Coverage",
                "emoji": "⚠️",
                "confidence": "Medium",
                "explanation": "Rising glucose in the elevated/hyper zone may indicate that the insulin dose was insufficient to cover the meal's carbs. This is called an insulin-to-carb ratio mismatch."
            })
        else:
            causes.append({
                "factor": "Post-Meal Rise",
                "emoji": "🍽️",
                "confidence": "Medium",
                "explanation": "A rising trend often indicates that a recent meal is being digested. Glucose typically rises 15–30 min after eating and peaks at 60–90 min."
            })

    # Falling trend causes
    if trend.startswith("falling"):
        causes.append({
            "factor": "Active Insulin",
            "emoji": "💉",
            "confidence": "High",
            "explanation": "Glucose is falling, likely due to insulin actively lowering blood sugar. If the rate is rapid (>25 mg/dL/hr), there's a risk of going too low (hypoglycaemia)."
        })
        if not any(c["factor"].startswith("Physical") for c in causes):
            causes.append({
                "factor": "Physical Activity",
                "emoji": "🏃",
                "confidence": "Low",
                "explanation": "Exercise increases glucose uptake by muscles, causing blood sugar to drop. Even moderate exercise (walking, climbing stairs) can accelerate a fall, especially if combined with insulin."
            })

    # Stable causes
    if trend == "stable" and zone == "normal":
        causes.append({
            "factor": "Balanced State",
            "emoji": "✅",
            "confidence": "High",
            "explanation": "Glucose is stable in the normal range — insulin and glucose production are balanced. This is the ideal state. No meal is currently being digested."
        })

    if not causes:
        causes.append({
            "factor": "Baseline Glucose Dynamics",
            "emoji": "🔄",
            "confidence": "Low",
            "explanation": "No strong signal detected. Blood sugar is naturally influenced by hormones (cortisol, glucagon), stress, sleep quality, and the time of day — even without meals or insulin."
        })

    # --- Prediction explanation ---
    pred_explanation = None
    if prediction is not None:
        pred = float(prediction)
        change = pred - current
        direction = "rise" if change > 0 else "fall"
        
        # Confidence band analysis
        band_width = float(q90) - float(q10) if q10 is not None and q90 is not None else 0
        narrow = band_width < (1.0 if is_mmol else 20)
        
        conf_text = f"The 80% confidence band is {('narrow' if narrow else 'wide')} ({float(q10):.1f}–{float(q90):.1f} {uname}), meaning the model is {'very confident' if narrow else 'somewhat uncertain'} about this prediction."
        
        # What safety score means
        safety_text = ""
        if safety == "safe":
            safety_text = "The safety score of 95/100 means the model's worst-case prediction (Q90) stays below 140 mg/dL — very safe."
        elif safety == "moderate":
            safety_text = "A score of 70/100 means the worst-case stays under 180 mg/dL but exceeds 140 — moderate risk."
        elif safety == "caution":
            safety_text = "A score of 45/100 means the average prediction is OK but the worst-case (Q90) crosses 180 mg/dL — proceed with caution."
        elif safety == "high_risk":
            safety_text = "A score of 20/100 means even the average prediction exceeds 180 mg/dL — high risk of hyperglycaemia."

        # Accuracy check
        acc_text = ""
        if actual is not None:
            error = abs(pred - float(actual))
            acc_text = f"The actual glucose turned out to be {float(actual):.1f} {uname} — the model was off by {error:.1f} {uname}."
            if error < (1.0 if is_mmol else 15):
                acc_text += " That's an excellent prediction!"
            elif error < (2.0 if is_mmol else 30):
                acc_text += " That's within the typical CGM sensor error range."
            else:
                acc_text += " This is a larger error, possibly due to an unexpected event (unlogged meal, insulin, or activity)."

        pred_explanation = {
            "direction": direction,
            "change": abs(change),
            "summary": f"The model predicts glucose will {direction} by {abs(change):.1f} {uname} over the next 60 minutes, from {current:.1f} to {pred:.1f} {uname}.",
            "confidence": conf_text,
            "safety": safety_text,
            "accuracy": acc_text,
        }

    return jsonify({
        "current": {"value": current, "zone": zone, "zone_emoji": zone_emoji, "description": zone_desc},
        "trend": {"direction": trend, "emoji": trend_emoji, "rate_per_hour": round(rate_per_hour, 1), "description": trend_desc},
        "volatility": {"level": volatility, "cv": round(cv, 1), "description": vol_desc},
        "events": events[:3],  # max 3
        "causes": causes[:4],  # max 4
        "prediction_explanation": pred_explanation,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run prediction on submitted glucose data.
    Expects: { glucose_history: [72 values], dataset: "hupa", model: "GRU", mode: "delta" }
    Returns: { prediction, q10, q90, unit }
    """
    data = request.json
    gl_history = np.array(data["glucose_history"], dtype=np.float32)
    dataset = data.get("dataset", "hupa")
    model_type = data.get("model", "GRU")
    mode = data.get("mode", "delta")
    scenario = int(data.get("scenario", 1))

    # Ensure 72 steps
    if len(gl_history) < WINDOW:
        gl_history = np.pad(gl_history, (WINDOW - len(gl_history), 0), 
                          mode='constant', constant_values=np.nan)
    elif len(gl_history) > WINDOW:
        gl_history = gl_history[-WINDOW:]

    # Load model
    result = load_model(model_type, dataset, scenario, mode)
    if result is None:
        return jsonify({"error": f"Model not found: {model_type}_{mode}_{dataset}_s{scenario}"}), 404
    model, input_dim = result

    # Build input tensor: [channels | mask]
    n_channels = input_dim // 2
    X = np.zeros((1, WINDOW, input_dim), dtype=np.float32)
    
    # Channel 0 = glucose
    gl_clean = gl_history.copy()
    mask = ~np.isnan(gl_clean)
    gl_clean[~mask] = 0.0
    X[0, :, 0] = gl_clean
    X[0, :, n_channels] = mask.astype(np.float32)  # glucose mask
    # Other channel masks default to 0 (no data)

    # Normalize (simple z-score using data stats)
    gl_valid = gl_clean[mask]
    if len(gl_valid) > 0:
        mean_gl = gl_valid.mean()
        std_gl = gl_valid.std()
        if std_gl < 1e-6:
            std_gl = 1.0
        X[0, :, 0] = (X[0, :, 0] - mean_gl) / std_gl

    x_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    # Predict with uncertainty
    mean_pred, q10, q90 = predict_with_uncertainty(model, x_tensor, K=20)
    
    pred = float(mean_pred[0])
    low = float(q10[0])
    high = float(q90[0])

    # If delta mode, reconstruct absolute
    if mode == "delta":
        current_gl = float(gl_history[~np.isnan(gl_history)][-1]) if not np.all(np.isnan(gl_history)) else 0
        pred = current_gl + pred
        low = current_gl + low
        high = current_gl + high

    unit = "mmol/L" if dataset == "brist1d" else "mg/dL"
    
    # Compute meal safety score (how likely to stay < 180 mg/dL)
    pred_mgdl = pred * 18.0182 if unit == "mmol/L" else pred
    high_mgdl = high * 18.0182 if unit == "mmol/L" else high
    
    if high_mgdl < 140:
        safety = "safe"
        safety_score = 95
    elif high_mgdl < 180:
        safety = "moderate"
        safety_score = 70
    elif pred_mgdl < 180:
        safety = "caution" 
        safety_score = 45
    else:
        safety = "high_risk"
        safety_score = 20

    return jsonify({
        "prediction": round(pred, 2),
        "q10": round(low, 2),
        "q90": round(high, 2),
        "unit": unit,
        "safety": safety,
        "safety_score": safety_score,
        "model": f"{model_type} ({mode})",
        "dataset": dataset,
    })


@app.route("/api/meal_compare", methods=["POST"])
def meal_compare():
    """
    Compare multiple meal options by simulating their glucose impact.
    Expects: { glucose_history: [...], meals: [{name, carbs, protein, fat, fiber}, ...], dataset, model, mode }
    """
    data = request.json
    gl_history = np.array(data["glucose_history"], dtype=np.float32)
    meals = data.get("meals", [])
    dataset = data.get("dataset", "hupa")
    model_type = data.get("model", "GRU")
    mode = data.get("mode", "delta")
    scenario = int(data.get("scenario", 1))

    if not meals:
        return jsonify({"error": "No meals provided"}), 400

    # Get base prediction (no meal)
    result = load_model(model_type, dataset, scenario, mode)
    if result is None:
        return jsonify({"error": "Model not found"}), 404
    model, input_dim = result

    rankings = []
    for meal in meals:
        carbs = meal.get("carbs", 0)
        protein = meal.get("protein", 0)
        fat = meal.get("fat", 0)
        
        # Simulate meal effect: add carbs to recent history
        # This is a simplified simulation — in production, carbs would
        # be injected into the carbs channel at the current timestep
        simulated_gl = gl_history.copy()
        
        n_channels = input_dim // 2
        X = np.zeros((1, WINDOW, input_dim), dtype=np.float32)
        
        gl_clean = simulated_gl.copy()
        mask = ~np.isnan(gl_clean)
        gl_clean[~mask] = 0.0
        X[0, :, 0] = gl_clean
        X[0, :, n_channels] = mask.astype(np.float32)
        
        # Add carbs signal to carbs channel if available
        if n_channels > 1:
            X[0, -1, 1] = carbs  # inject carbs at current time
            X[0, -1, n_channels + 1] = 1.0  # carbs mask
        
        # Normalize glucose
        gl_valid = gl_clean[mask]
        if len(gl_valid) > 0:
            mean_gl = gl_valid.mean()
            std_gl = max(gl_valid.std(), 1e-6)
            X[0, :, 0] = (X[0, :, 0] - mean_gl) / std_gl

        x_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        mean_pred, q10, q90 = predict_with_uncertainty(model, x_tensor, K=15)
        
        pred = float(mean_pred[0])
        low = float(q10[0])
        high = float(q90[0])
        
        if mode == "delta":
            current = float(gl_history[~np.isnan(gl_history)][-1]) if not np.all(np.isnan(gl_history)) else 0
            pred += current
            low += current
            high += current
        
        unit = "mmol/L" if dataset == "brist1d" else "mg/dL"
        pred_mgdl = pred * 18.0182 if unit == "mmol/L" else pred
        high_mgdl = high * 18.0182 if unit == "mmol/L" else high
        
        if high_mgdl < 140:
            score = 95
        elif high_mgdl < 180:
            score = 70
        elif pred_mgdl < 180:
            score = 45
        else:
            score = 20
        
        rankings.append({
            "name": meal.get("name", "Unknown"),
            "carbs": carbs,
            "protein": protein,
            "fat": fat,
            "prediction": round(pred, 2),
            "q10": round(low, 2),
            "q90": round(high, 2),
            "safety_score": score,
            "unit": unit,
        })
    
    # Sort by safety score descending
    rankings.sort(key=lambda x: x["safety_score"], reverse=True)
    
    return jsonify({"rankings": rankings})


# ══════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Fuel Up+ Interactive Demo Server")
    print("=" * 60)
    
    # Pre-load sample data
    prepare_sample_participants()
    
    # Pre-load best models
    for ds in ["hupa", "brist1d", "cgmacros"]:
        for mode in ["abs", "delta"]:
            for mt in ["GRU", "Transformer"]:
                result = load_model(mt, ds, 1, mode)
                if result:
                    print(f"  Loaded: {mt}_{mode}_{ds}_s1 ✓")
    
    print("\n  Server starting at http://localhost:5050")
    print("  Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
