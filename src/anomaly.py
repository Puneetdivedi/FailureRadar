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


def calculate_health_score(anomaly_rate: float, max_score: float) -> float:
    """
    Calculate a 0-100 health score based on anomaly rate and severity.
    """
    # Base score is 100
    score = 100.0
    
    # Deduct based on anomaly rate (penalty up to 60 points)
    # 20% anomaly rate = -60 points
    score -= min(60.0, anomaly_rate * 3.0)
    
    # Deduct based on max anomaly score (penalty up to 40 points)
    # Score 0.8+ is severe
    score -= min(40.0, max_score * 40.0)
    
    return max(0.0, round(score, 1))


def get_anomaly_summary(df: pd.DataFrame) -> dict:
    if "anomaly_flag" not in df.columns:
        return {}
    summary = {}
    for machine, grp in df.groupby("machine_id"):
        anomaly_rate = float(grp["anomaly_flag"].mean()) * 100
        max_score = float(grp["anomaly_score"].max())
        
        summary[machine] = {
            "total": int(len(grp)),
            "anomalies": int(grp["anomaly_flag"].sum()),
            "anomaly_rate": round(anomaly_rate, 1),
            "max_score": round(max_score, 3),
            "health_score": calculate_health_score(anomaly_rate, max_score)
        }
    return summary
