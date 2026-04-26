# Fuel Up+

Fuel Up+ is a research prototype for uncertainty-aware blood glucose forecasting and meal-risk comparison. The project evaluates sequence models on continuous glucose monitoring (CGM), nutrition, insulin, and activity data from three public datasets:

- **HUPA-UCM**: Type 1 diabetes CGM, insulin, meals, and activity signals.
- **BrisT1D**: Type 1 diabetes CGM, insulin, carbohydrate, heart rate, and step data.
- **CGMacros**: Non-diabetic CGM data with meal macronutrients and activity features.

The core forecasting task is:

> Use the previous 6 hours of data sampled at 5-minute intervals to predict glucose 60 minutes ahead.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `dataset_builder.py` | Builds unified supervised windows across datasets and scenarios. |
| `baselines.py` | Persistence, linear drift, Ridge, and GBDT baselines. |
| `deep_models.py` | GRU and Transformer training, checkpointing, and result plots. |
| `uncertainty_models.py` | Quantile and MC-Dropout uncertainty experiments. |
| `demo_server.py` | Flask backend for the interactive prediction demo. |
| `demo_static/` | Browser UI and presentation manual for the demo. |
| `STEP0_evaluation_protocol.md` | Evaluation design, split rules, and reporting protocol. |
| `figures_*` | Generated EDA, baseline, model, and uncertainty figures. |
| `Kaggle BG/`, `Mendeley/`, `cgmacros-*` | Local dataset copies used by the experiments. |

## Setup

Create a Python environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch should be installed for your hardware. On Apple Silicon, the default PyPI build supports the MPS backend used by the training scripts.

## Running the Demo

```bash
python demo_server.py
```

Then open:

```text
http://localhost:5050
```

The demo loads trained checkpoints from `figures_step3/checkpoints/` and serves sample participant windows from the local datasets.

## Reproducing Results

Run the experiment stages independently:

```bash
python dataset_builder.py
python baselines.py
python deep_models.py --smoke-test
python uncertainty_models.py --subset 0.1
```

Useful options:

- `deep_models.py --dataset hupa --scenario 1 --subset 0.1`
- `deep_models.py --dataset cgmacros --scenario 3 --subset 0.1`
- `uncertainty_models.py --subset 0.1`

The scripts write plots and logs into the `figures_*` directories.

## Evaluation Summary

The experimental design uses participant-level train/validation/test splits to avoid leakage across people. Models are evaluated with MAE and RMSE on 60-minute-ahead forecasts, with additional per-participant and failure-case analysis.

Input scenarios:

- **Scenario 1**: glucose plus carbohydrate and insulin fields where available.
- **Scenario 2**: Scenario 1 plus full macronutrients for CGMacros.
- **Scenario 3**: Scenario 2 plus activity features where available.

Target modes:

- **Absolute**: predict glucose at `t + 60 min`.
- **Delta**: predict the change from current glucose to `t + 60 min`.

## Notes

- This is a research prototype, not a medical device.
- Large public datasets and generated figures are included to keep the demo reproducible for presentation.
- Paths are repository-relative, so scripts can be run from a fresh clone without editing personal machine paths.
