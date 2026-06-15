from __future__ import annotations

import pandas as pd
import plotly.express as px

from src.features import ANGLE_COLUMNS


def plot_angle_timeseries(df: pd.DataFrame):
    available = [col for col in ANGLE_COLUMNS if col in df.columns]
    if not available or df.empty:
        return px.line(title="No pose angle data available")
    long_df = df.melt(id_vars=["time_sec"], value_vars=available, var_name="angle", value_name="degrees")
    fig = px.line(long_df, x="time_sec", y="degrees", color="angle", title="Joint Angles Over Time")
    fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Angle (degrees)")
    return fig


def plot_feature_summary(features: dict):
    rom_features = {k: v for k, v in features.items() if k.endswith("_rom")}
    if not rom_features:
        return px.bar(title="No range-of-motion features available")
    data = pd.DataFrame({"feature": list(rom_features.keys()), "value": list(rom_features.values())})
    fig = px.bar(data, x="feature", y="value", title="Range of Motion Features")
    fig.update_layout(xaxis_title="Feature", yaxis_title="Degrees")
    return fig

