from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from src.biomechanics import calculate_angle, trunk_lean_angle


POSE = mp.solutions.pose
DRAWING = mp.solutions.drawing_utils


def process_video(
    video_path: Path,
    annotated_output_path: Path | None = None,
    max_frames: int | None = None,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if annotated_output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_output_path), fourcc, fps, (width, height))

    rows = []
    with POSE.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            row = _empty_row(frame_idx, fps)

            if results.pose_landmarks:
                row.update(_extract_angles(results.pose_landmarks.landmark, width, height))
                row["pose_detected"] = True
                if writer:
                    DRAWING.draw_landmarks(frame, results.pose_landmarks, POSE.POSE_CONNECTIONS)
                    _draw_angle_labels(frame, row)

            if writer:
                writer.write(frame)

            rows.append(row)
            frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    return pd.DataFrame(rows)


def _empty_row(frame_idx: int, fps: float) -> dict:
    return {
        "frame": frame_idx,
        "time_sec": frame_idx / fps,
        "pose_detected": False,
        "left_knee_angle": np.nan,
        "right_knee_angle": np.nan,
        "left_hip_angle": np.nan,
        "right_hip_angle": np.nan,
        "left_ankle_angle": np.nan,
        "right_ankle_angle": np.nan,
        "left_elbow_angle": np.nan,
        "right_elbow_angle": np.nan,
        "left_shoulder_angle": np.nan,
        "right_shoulder_angle": np.nan,
        "trunk_lean_angle": np.nan,
    }


def _extract_angles(landmarks, width: int, height: int) -> dict:
    p = lambda idx: (landmarks[idx].x * width, landmarks[idx].y * height)

    left_shoulder = p(POSE.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = p(POSE.PoseLandmark.RIGHT_SHOULDER.value)
    left_elbow = p(POSE.PoseLandmark.LEFT_ELBOW.value)
    right_elbow = p(POSE.PoseLandmark.RIGHT_ELBOW.value)
    left_wrist = p(POSE.PoseLandmark.LEFT_WRIST.value)
    right_wrist = p(POSE.PoseLandmark.RIGHT_WRIST.value)
    left_hip = p(POSE.PoseLandmark.LEFT_HIP.value)
    right_hip = p(POSE.PoseLandmark.RIGHT_HIP.value)
    left_knee = p(POSE.PoseLandmark.LEFT_KNEE.value)
    right_knee = p(POSE.PoseLandmark.RIGHT_KNEE.value)
    left_ankle = p(POSE.PoseLandmark.LEFT_ANKLE.value)
    right_ankle = p(POSE.PoseLandmark.RIGHT_ANKLE.value)
    left_foot = p(POSE.PoseLandmark.LEFT_FOOT_INDEX.value)
    right_foot = p(POSE.PoseLandmark.RIGHT_FOOT_INDEX.value)

    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
    hip_mid = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

    return {
        "left_knee_angle": calculate_angle(left_hip, left_knee, left_ankle),
        "right_knee_angle": calculate_angle(right_hip, right_knee, right_ankle),
        "left_hip_angle": calculate_angle(left_shoulder, left_hip, left_knee),
        "right_hip_angle": calculate_angle(right_shoulder, right_hip, right_knee),
        "left_ankle_angle": calculate_angle(left_knee, left_ankle, left_foot),
        "right_ankle_angle": calculate_angle(right_knee, right_ankle, right_foot),
        "left_elbow_angle": calculate_angle(left_shoulder, left_elbow, left_wrist),
        "right_elbow_angle": calculate_angle(right_shoulder, right_elbow, right_wrist),
        "left_shoulder_angle": calculate_angle(left_elbow, left_shoulder, left_hip),
        "right_shoulder_angle": calculate_angle(right_elbow, right_shoulder, right_hip),
        "trunk_lean_angle": trunk_lean_angle(shoulder_mid, hip_mid),
    }


def _draw_angle_labels(frame, row: dict) -> None:
    label = f"L knee: {row['left_knee_angle']:.1f} | R knee: {row['right_knee_angle']:.1f}"
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 220, 30), 2)
    label = f"Trunk lean: {row['trunk_lean_angle']:.1f}"
    cv2.putText(frame, label, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 220, 30), 2)

