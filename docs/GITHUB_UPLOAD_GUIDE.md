# GitHub Upload Guide

Upload the full `biomechanics_cv_ml_analyzer` folder to GitHub.

## Upload These

```text
README.md
requirements.txt
.gitignore
app.py
src/
scripts/
data/labels_template.csv
docs/
```

## Do Not Upload These

```text
.venv/
outputs/
models/*.joblib
data/videos/
data/raw/
__pycache__/
*.pyc
```

## Why Not Upload Videos?

Videos can be large and may contain private personal movement data. GitHub is not good for large video files.

Better options:

- upload only a very small demo video if it is safe and allowed
- use Google Drive and link it in README
- use GitHub Releases for sample media
- keep private class videos out of the repo

## What Each File Means

`README.md`

Main explanation of the project and how to run it.

`app.py`

Streamlit dashboard.

`requirements.txt`

Python dependencies.

`src/pose_estimation.py`

Extracts human pose landmarks from video using MediaPipe.

`src/biomechanics.py`

Calculates joint angles.

`src/features.py`

Turns frame-level angles into ML-ready features.

`src/modeling.py`

Trains and loads the scikit-learn classifier.

`src/visualization.py`

Creates plots for the dashboard.

`scripts/process_video.py`

Command-line script to process one video and export CSV files.

`scripts/train_model.py`

Command-line script to train the ML model from labeled videos.

`data/labels_template.csv`

Template showing how to label videos for ML training.

`docs/PROJECT_STEPS.md`

Step-by-step explanation for your report and interview.

## Suggested GitHub Repo Name

```text
biomechanics-cv-ml-analyzer
```

## Suggested GitHub Description

```text
Computer vision and machine learning system for biomechanics video analysis using MediaPipe, OpenCV, scikit-learn, and Streamlit.
```

