"""
shape_detection.py
Detect Circle, Triangle, Rectangle, Square from contours using polygon
approximation + circularity test.
"""

import cv2
import numpy as np


def find_contours(mask, min_area: int = 1200):
    """Find external contours from a binary mask and filter by area."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def classify_shape(contour) -> str:
    """Classify a single contour as Triangle / Square / Rectangle / Circle / Unknown."""
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return "Unknown"

    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    sides = len(approx)
    area = cv2.contourArea(contour)

    # Circularity = 4*pi*A / P^2  -> ~1.0 for perfect circles
    circularity = 4 * np.pi * area / (peri * peri)

    if sides == 3:
        return "Triangle"
    if sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h) if h else 0
        return "Square" if 0.90 <= ratio <= 1.10 else "Rectangle"
    if sides >= 5 and circularity > 0.78:
        return "Circle"
    return "Unknown"
