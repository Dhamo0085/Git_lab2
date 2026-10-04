import os
import json
import joblib
from sklearn.metrics import f1_score
from src.data import get_data

ts = open("timestamp.txt").read().strip()
_, _, X_te, _, _, y_te = get_data()

model = joblib.load(f"models/model_{ts}_rf.joblib")
f1 = f1_score(y_te, model.predict(X_te))

os.makedirs("metrics", exist_ok=True)
with open(f"metrics/{ts}_metrics.json", "w") as f:
    json.dump({"f1_score": f1}, f, indent=2)
print(f"F1 score: {f1:.4f}")
