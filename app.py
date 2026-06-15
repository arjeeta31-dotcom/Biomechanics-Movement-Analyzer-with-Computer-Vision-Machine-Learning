from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.features import extract_biomechanics_features
from src.modeling import load_model_if_available, predict_from_features
from src.pose_estimation import process_video
from src.visualization import plot_angle_timeseries, plot_feature_summary


st.set_page_config(page_title="Biomechanics CV + ML Analyzer", layout="wide")

st.title("Applied Biomechanics CV + ML Analyzer")
st.caption("Pose estimation, joint-angle analysis, feature extraction, and ML prediction from movement videos")

with st.sidebar:
    st.header("Analysis Settings")
    max_frames = st.slider("Max frames to process", min_value=100, max_value=3000, value=900, step=100)
    show_annotated = st.checkbox("Generate annotated video", value=True)
    st.info("For ML prediction, train a model first using scripts/train_model.py.")

uploaded_file = st.file_uploader("Upload biomechanics video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        video_path = tmpdir_path / uploaded_file.name
        video_path.write_bytes(uploaded_file.getbuffer())

        annotated_path = tmpdir_path / f"annotated_{uploaded_file.name}"

        with st.spinner("Running pose estimation and biomechanical analysis..."):
            pose_df = process_video(
                video_path=video_path,
                annotated_output_path=annotated_path if show_annotated else None,
                max_frames=max_frames,
            )
            features = extract_biomechanics_features(pose_df)

        st.subheader("Pose Tracking Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Frames processed", len(pose_df))
        c2.metric("Detected frames", int(pose_df["pose_detected"].sum()) if not pose_df.empty else 0)
        c3.metric("Feature count", len(features))

        if show_annotated and annotated_path.exists():
            st.subheader("Annotated Video")
            st.video(str(annotated_path))

        st.subheader("Joint Angle Time Series")
        st.plotly_chart(plot_angle_timeseries(pose_df), use_container_width=True)

        st.subheader("Biomechanical Feature Summary")
        feature_df = pd.DataFrame([features])
        st.dataframe(feature_df, use_container_width=True)
        st.plotly_chart(plot_feature_summary(features), use_container_width=True)

        model = load_model_if_available()
        if model:
            prediction = predict_from_features(model, features)
            st.subheader("ML Prediction")
            st.success(f"Predicted movement class: {prediction['label']}")
            if prediction.get("probabilities"):
                st.json(prediction["probabilities"])
        else:
            st.warning("No trained model found. Run `python scripts/train_model.py data/labels.csv` to enable ML prediction.")

        st.subheader("Download Results")
        st.download_button(
            "Download frame-level joint angles CSV",
            data=pose_df.to_csv(index=False),
            file_name="joint_angles.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download extracted features CSV",
            data=feature_df.to_csv(index=False),
            file_name="biomechanics_features.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a movement video to begin analysis.")

