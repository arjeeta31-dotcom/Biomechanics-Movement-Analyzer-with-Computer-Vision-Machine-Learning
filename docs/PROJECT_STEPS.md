# Step-by-Step Project Plan

## 1. Define The Problem

Convert a biomechanics assignment into a CV/ML project that analyzes movement videos.

Example objective:

> Automatically extract joint kinematics from a human movement video and classify movement quality using machine learning.

## 2. Input Data

Use your class video as the first demo input.

For ML training, collect multiple videos and label them:

- `normal`
- `knee_valgus`
- `asymmetric`
- `limited_range_of_motion`
- `unstable_movement`

## 3. Pose Estimation

File: `src/pose_estimation.py`

The system uses MediaPipe Pose to detect body landmarks in each frame.

## 4. Biomechanical Angle Calculation

File: `src/biomechanics.py`

The system calculates angles using three landmark points:

```text
hip-knee-ankle -> knee angle
shoulder-hip-knee -> hip angle
knee-ankle-foot -> ankle angle
```

## 5. Feature Engineering

File: `src/features.py`

Features include:

- mean joint angle
- minimum joint angle
- maximum joint angle
- range of motion
- angular velocity
- left-right asymmetry
- smoothness score

## 6. Machine Learning

File: `src/modeling.py`

The project trains a Random Forest classifier using scikit-learn.

This is suitable for a college project because it is explainable, works with tabular biomechanical features, and does not need a huge dataset.

## 7. Dashboard

File: `app.py`

The Streamlit dashboard shows:

- uploaded video
- pose tracking summary
- annotated video
- joint-angle plots
- feature table
- ML prediction
- CSV downloads

## 8. GitHub Portfolio Story

Explain it like this:

> I converted raw biomechanics movement videos into structured kinematic features using pose estimation, then trained an ML classifier to identify movement patterns and possible biomechanical abnormalities.

