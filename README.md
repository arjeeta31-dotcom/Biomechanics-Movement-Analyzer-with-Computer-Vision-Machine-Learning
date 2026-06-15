# Autonomous Biomechanical Movement Analytics Pipeline using Computer Vision & ML

An enterprise-grade Computer Vision and Machine Learning pipeline designed to automate human movement tracking, extract high-fidelity biomechanical kinematics, and classify movement quality. Powered by **MediaPipe** for frame-level pose landmark tracking, **OpenCV** for spatial stream isolation, and **Scikit-Learn** for deterministic hazard/pattern prediction, this system bridges the gap between raw video feeds and actionable human kinetics.

## Key Features

* **Real-time Pose Estimation:** Extracts 33 normalized 3D human body landmarks frame-by-frame utilizing optimized MediaPipe perception graphs.
* **Biomechanical Feature Kinematics:** Explicit mathematical calculators computing critical joint tracking matrices including knee flexion, hip orientation, ankle deviation, trunk lean, and movement smoothness.
* **Dynamic Time Series Extraction:** Maps spatial coordinates into structured continuous waveforms to analyze exact range-of-motion (ROM) and left-right lateral asymmetries.
* **Deterministic Classifier Loop:** Downstreams continuous kinematic variables into a production Random Forest engine to flag anomalous movement mechanics or structural risks.
* **Interactive Visualization Hub:** A high-performance Streamlit dashboard offering dual layout control: side-by-side rendering of annotated diagnostic videos and interactive Plotly metric timeseries.

---

##  System Architecture

```text
  ┌──────────────────────────────┐
  │   Raw Movement Video Input   │ (MP4, MOV, AVI stream wrappers)
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │   MediaPipe Pose Inference   │ (Normalized 3D Landmark Localization)
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │   Kinematic Feature Engine   │ (Computes ROM, Angles, Velocities, Asymmetry)
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │     Scikit-Learn Engine      │ (Random Forest Pattern Mapping & Diagnostics)
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │ Interactive UI / Plotly Hub  │ (Real-time Metric Stream Rendering)
  └──────────────────────────────┘
```
## Project Repository Anatomy

```text

├── app.py                     # Main Streamlit dashboard application UI
├── requirements.txt           # Verified Python ecosystem package locks
├── .gitignore                 # Strictly locks out multi-gigabyte video arrays and binary models
├── src/                       # core analytics pipeline packages
│   ├── __init__.py
│   ├── pose_estimation.py     # MediaPipe orchestration and video stream parser
│   ├── features.py            # High-fidelity joint angle and asymmetry feature extraction
│   ├── modeling.py            # Joblib model loaders and serialization interfaces
│   └── visualization.py       # High-performance Plotly graph generator blocks
├── scripts/                   # Production CLI orchestration utilities
│   └── train_model.py         # Automated pipeline training and validation controller
├── models/                    # Serialized ML artifacts
│   └── .gitkeep               # Preserves directory hierarchy (stores local joblib layers)
└── data/                      # Local datasets
    └── labels.csv             # Target manifest file mapping tracking assets to labels

 ```

## Quick Start Local Configuration

# Initialize Sandbox & Dependencies
Provision a secure virtual runtime space to deploy the ecosystem configurations:

- ** Create and trigger the virtual python sandbox**
python -m venv .venv

- ** Activate environment tracking (Windows Command Prompt) **
.venv\Scripts\activate

- ** Install performance dependency layers**
pip install -r requirements.txt

# Launch Streamlit Analytics UI
Launch the native visualization platform locally on your browser:

streamlit run app.py

The server will cleanly route and spinning up your client runtime dashboard at: http://localhost:8501

# Model Training & Batch Orchestration
To compile and update the machine learning classifier layers against custom lab records, format a standard execution manifest data/labels.csv:

video_path,label
data/videos/normal_walk_01.mp4,normal
data/videos/knee_valgus_01.mp4,knee_valgus
data/videos/asymmetric_gait_01.mp4,asymmetric

- **Execute the optimized training engine utility directly:**

 python scripts/train_model.py data/labels.csv

 The system pipeline automatically evaluates the dataset, trains a localized Random Forest array, and outputs a highly optimized serialization layer to models/biomech_classifier.joblib.

 Disclaimer: This repository is developed entirely as a technical portfolio engineering project. The metrics, calculations, and analytical insights rendered do not constitute professional clinical diagnostics, medical validation, or sports medicine advice.


 

  









