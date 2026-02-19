"""
anomaly.py — Isolation Forest anomaly detection on sensor data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES = ["temperature_C", "vibration_mm_s", "pressure_bar", "rpm", "current_A"]


def run_anomaly_detection(df: pd.DataFrame, contamination: float = 0.15) -> pd.DataFrame:
    result = df.copy()
    feature_cols = [c for c in FEATURES if c in df.columns]
    X = df[feature_cols].dropna()

    if len(X) < 10:
        result["anomaly_score"] = 0.0
        result["anomaly_label"] = "insufficient_data"
        result["anomaly_flag"] = False
        return result

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
    preds  = model.fit_predict(X_scaled)
    scores = model.score_samples(X_scaled)

    result.loc[X.index, "anomaly_score"] = np.round(-scores, 4)
    result.loc[X.index, "anomaly_flag"]  = preds == -1
    result.loc[X.index, "anomaly_label"] = ["🔴 ANOMALY" if p == -1 else "🟢 Normal" for p in preds]

    result["anomaly_score"] = result["anomaly_score"].fillna(0.0)
    result["anomaly_flag"]  = result["anomaly_flag"].fillna(False)
    result["anomaly_label"] = result["anomaly_label"].fillna("🟢 Normal")

    return result


def get_anomaly_summary(df: pd.DataFrame) -> dict:
    if "anomaly_flag" not in df.columns:
        return {}
    summary = {}
    for machine, grp in df.groupby("machine_id"):
        summary[machine] = {
            "total": int(len(grp)),
            "anomalies": int(grp["anomaly_flag"].sum()),
            "anomaly_rate": round(float(grp["anomaly_flag"].mean()) * 100, 1),
            "max_score": round(float(grp["anomaly_score"].max()), 3)
        }
    return summary