import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.data import get_data

ts = open("timestamp.txt").read().strip()
X_tr, _, _, y_tr, _, _ = get_data()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)

os.makedirs("models", exist_ok=True)
joblib.dump(model, f"models/model_{ts}_rf.joblib")
print(f"Saved models/model_{ts}_rf.joblib")
