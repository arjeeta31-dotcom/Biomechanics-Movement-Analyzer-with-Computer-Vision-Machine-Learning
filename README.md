# Applied Biomechanics Computer Vision + Machine Learning Analyzer

This project converts an applied biomechanics class assignment into a portfolio-ready **Machine Learning + Computer Vision** project.

It analyzes human movement videos using pose estimation, extracts biomechanical features such as joint angles and range of motion, and trains a machine learning model for movement classification or abnormality/risk prediction.

## Project Idea

```text
Input Video
    -> Pose Estimation
    -> Joint Landmark Tracking
    -> Biomechanical Angle Calculation
    -> Feature Extraction
    -> ML Classification / Risk Prediction
    -> Streamlit Dashboard
```

## What This Project Does

- Upload a biomechanics movement video.
- Detect human pose landmarks frame by frame using MediaPipe.
- Compute joint angles:
  - knee angle
  - hip angle
  - ankle angle
  - shoulder angle
  - elbow angle
  - trunk lean
- Extract ML features:
  - min/max/mean joint angles
  - range of motion
  - angular velocity
  - left-right asymmetry
  - movement smoothness
- Train a Random Forest classifier from labeled video features.
- Predict movement class or possible biomechanical risk.
- Display plots and annotated video in a Streamlit dashboard.

## Example Resume Bullets

```text
- Developed a computer vision-based biomechanics analyzer using MediaPipe and OpenCV to extract human pose landmarks from movement videos.

- Engineered joint-angle, range-of-motion, angular velocity, and left-right asymmetry features for biomechanical movement assessment.

- Trained a machine learning classifier using scikit-learn to classify movement patterns and flag possible abnormal biomechanics.

- Built an interactive Streamlit dashboard for video upload, pose visualization, feature analysis, and ML-based prediction.
```

## How To Run On Your Computer

Open PowerShell:

```powershell
cd C:\Users\asus\Documents\Codex\2026-06-14\project-title-automated-multimodal-legal-medical\outputs\biomechanics_cv_ml_analyzer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL, usually:

```text
http://localhost:8501
```

Upload your class video:

```text
F:/SHILPI/MTech BMD IIT Indore/Sem 2/BSE 643 Applied biomechanics/2502171003_video 2.mp4
```

## How To Train The ML Model

First create a labels CSV like:

```csv
video_path,label
data/videos/normal_walk_01.mp4,normal
data/videos/knee_valgus_01.mp4,knee_valgus
data/videos/asymmetric_gait_01.mp4,asymmetric
```

Then run:

```powershell
python scripts/train_model.py data/labels.csv
```

The trained model will be saved to:

```text
models/biomech_classifier.joblib
```

## Important Note

If you only have one video, the project can still analyze and visualize biomechanics. For machine learning training, you need multiple labeled videos.

## Disclaimer

This project is for educational and portfolio use. It is not a medical diagnostic system.

