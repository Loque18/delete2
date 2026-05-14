from math import atan2, cos, sin, sqrt, pi
import numpy as np

def angle_diff(a: float, b: float):
    return atan2(sin(a - b), cos(a - b))

def clamp(value, min_value=-1.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def distance(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def deg2rad(degrees):
    return degrees * (pi / 180)

def rad2Deg(radians):
    return radians * (180 / pi)

def safe_sensor(v, max_range=20.0):
        if v is None:
            return max_range
        if np.isinf(v) or np.isnan(v):
            return max_range
        return float(v)