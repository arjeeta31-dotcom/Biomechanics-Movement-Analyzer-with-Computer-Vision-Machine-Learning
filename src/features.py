from __future__ import annotations

import numpy as np
import pandas as pd


ANGLE_COLUMNS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "trunk_lean_angle",
]


def extract_biomechanics_features(df: pd.DataFrame) -> dict:
    valid = df[df["pose_detected"]].copy()
    features: dict[str, float] = {}

    if valid.empty:
        return {"valid_pose_ratio": 0.0}

    features["valid_pose_ratio"] = len(valid) / max(len(df), 1)
    duration = valid["time_sec"].max() - valid["time_sec"].min()
    features["duration_sec"] = float(duration)

    for col in ANGLE_COLUMNS:
        if col not in valid:
            continue
        series = valid[col].dropna()
        if series.empty:
            continue
        features[f"{col}_mean"] = float(series.mean())
        features[f"{col}_min"] = float(series.min())
        features[f"{col}_max"] = float(series.max())
        features[f"{col}_rom"] = float(series.max() - series.min())
        features[f"{col}_std"] = float(series.std(ddof=0))
        features[f"{col}_mean_abs_velocity"] = _mean_abs_velocity(series, valid.loc[series.index, "time_sec"])

    features["knee_rom_asymmetry"] = _asymmetry(features, "left_knee_angle_rom", "right_knee_angle_rom")
    features["hip_rom_asymmetry"] = _asymmetry(features, "left_hip_angle_rom", "right_hip_angle_rom")
    features["ankle_rom_asymmetry"] = _asymmetry(features, "left_ankle_angle_rom", "right_ankle_angle_rom")
    features["mean_knee_angle_asymmetry"] = _asymmetry(features, "left_knee_angle_mean", "right_knee_angle_mean")
    features["movement_smoothness_score"] = _smoothness_score(valid)
    return features


def _mean_abs_velocity(values: pd.Series, times: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    dv = np.diff(values.to_numpy())
    dt = np.diff(times.to_numpy())
    dt = np.where(dt == 0, np.nan, dt)
    velocity = np.abs(dv / dt)
    return float(np.nanmean(velocity))


def _asymmetry(features: dict, left_key: str, right_key: str) -> float:
    left = features.get(left_key)
    right = features.get(right_key)
    if left is None or right is None:
        return 0.0
    denominator = max(abs(left), abs(right), 1e-6)
    return float(abs(left - right) / denominator)


def _smoothness_score(df: pd.DataFrame) -> float:
    cols = [col for col in ["left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle"] if col in df]
    jerk_values = []
    for col in cols:
        values = df[col].dropna().to_numpy()
        if len(values) >= 4:
            jerk_values.extend(np.abs(np.diff(values, n=3)))
    if not jerk_values:
        return 0.0
    return float(1 / (1 + np.mean(jerk_values)))

