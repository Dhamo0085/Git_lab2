import glob
import json
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from src.data import get_data

path = sorted(glob.glob("models/model_*_rf.joblib"))[-1]
ts = path.split("model_")[1].split("_rf")[0]
_, X_cal, X_te, _, y_cal, y_te = get_data()

base = joblib.load(path)
calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
calibrated.fit(X_cal, y_cal)
joblib.dump(calibrated, f"models/model_{ts}_calibrated.joblib")

result = {
    "brier_before": brier_score_loss(y_te, base.predict_proba(X_te)[:, 1]),
    "brier_after": brier_score_loss(y_te, calibrated.predict_proba(X_te)[:, 1]),
}
with open(f"metrics/{ts}_calibration.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"Calibrated model for {ts}: {result}")
