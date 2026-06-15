from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features import extract_biomechanics_features
from src.pose_estimation import process_video


def main(video_path: str):
    path = Path(video_path)
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    angles_path = output_dir / f"{path.stem}_joint_angles.csv"
    features_path = output_dir / f"{path.stem}_features.csv"
    annotated_path = output_dir / f"{path.stem}_annotated.mp4"

    df = process_video(path, annotated_output_path=annotated_path)
    features = extract_biomechanics_features(df)

    df.to_csv(angles_path, index=False)
    import pandas as pd

    pd.DataFrame([features]).to_csv(features_path, index=False)
    print(f"Saved angles: {angles_path}")
    print(f"Saved features: {features_path}")
    print(f"Saved annotated video: {annotated_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/process_video.py path/to/video.mp4")
        raise SystemExit(1)
    main(sys.argv[1])

