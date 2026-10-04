# Lab 2: Model Training, Calibration and Versioning with GitHub Actions

On every push to `main`:

1. `model_retraining_on_push.yml` trains a RandomForest on synthetic data, evaluates F1, and commits a timestamped model to `models/` and metrics to `metrics/`.
2. `model_calibration_on_push.yml` runs after retraining, calibrates the latest model (sigmoid), and commits `model_<timestamp>_calibrated.joblib` plus a calibration report.

## Results (example run)

- F1 score: 0.8998
- Brier score: 0.0957 before calibration, 0.0796 after

## Run locally

    python -m venv lab_02 && source lab_02/bin/activate
    pip install -r requirements.txt
    date +%Y%m%d%H%M%S > timestamp.txt
    python -m src.train_model
    python -m src.evaluate_model
    python -m src.calibrate_model
