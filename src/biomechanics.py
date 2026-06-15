from __future__ import annotations

import math

import numpy as np


def calculate_angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Returns angle ABC in degrees."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    c_arr = np.array(c, dtype=float)

    ba = a_arr - b_arr
    bc = c_arr - b_arr

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")

    cosine = np.dot(ba, bc) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def trunk_lean_angle(shoulder_mid: tuple[float, float], hip_mid: tuple[float, float]) -> float:
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    if dx == 0 and dy == 0:
        return float("nan")
    vertical = np.array([0.0, -1.0])
    trunk = np.array([dx, dy], dtype=float)
    denom = np.linalg.norm(trunk) * np.linalg.norm(vertical)
    cosine = np.dot(trunk, vertical) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)

