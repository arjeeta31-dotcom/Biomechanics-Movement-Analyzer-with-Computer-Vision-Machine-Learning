from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path("models/biomech_classifier.joblib")


def train_classifier(feature_table: pd.DataFrame, label_column: str = "label") -> dict:
    if label_column not in feature_table.columns:
        raise ValueError(f"Missing label column: {label_column}")

    y = feature_table[label_column]
    x = feature_table.drop(columns=[label_column, "video_path"], errors="ignore")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
        ]
    )

    if len(feature_table) >= 6 and y.nunique() > 1:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    else:
        pipeline.fit(x, y)
        report = {"note": "Dataset too small for train/test split. Model trained on all rows."}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipeline, "feature_columns": list(x.columns)}, MODEL_PATH)
    return {"model_path": str(MODEL_PATH), "report": report}


def load_model_if_available():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_from_features(model_bundle: dict, features: dict) -> dict:
    model = model_bundle["model"]
    columns = model_bundle["feature_columns"]
    row = pd.DataFrame([{col: features.get(col, np.nan) for col in columns}])
    label = model.predict(row)[0]
    result = {"label": str(label)}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(row)[0]
        classes = model.classes_
        result["probabilities"] = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
    return result

