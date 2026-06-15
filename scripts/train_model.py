from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features import extract_biomechanics_features
from src.modeling import train_classifier
from src.pose_estimation import process_video


def build_feature_table(labels_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    rows = []
    for _, row in labels.iterrows():
        video_path = Path(row["video_path"])
        if not video_path.is_absolute():
            video_path = ROOT / video_path
        print(f"Processing {video_path}")
        pose_df = process_video(video_path)
        features = extract_biomechanics_features(pose_df)
        features["video_path"] = str(video_path)
        features["label"] = row["label"]
        rows.append(features)
    return pd.DataFrame(rows)


def main(labels_csv: str):
    feature_table = build_feature_table(Path(labels_csv))
    output_path = ROOT / "data" / "feature_table.csv"
    output_path.parent.mkdir(exist_ok=True)
    feature_table.to_csv(output_path, index=False)
    result = train_classifier(feature_table)
    print(f"Saved feature table: {output_path}")
    print(result)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/train_model.py data/labels.csv")
        raise SystemExit(1)
    main(sys.argv[1])

