"""
color_detection.py
HSV-based color detection for Red, Blue, Green, Yellow.
Returns a binary mask per color and a helper to identify the dominant color
inside a given contour region.
"""

import cv2
import numpy as np

# HSV ranges (OpenCV: H in [0,179], S/V in [0,255])
# Red wraps around 0/180, so we use two ranges.
COLOR_RANGES = {
    "Red":    [((0, 120, 70),   (10, 255, 255)),
               ((170, 120, 70), (179, 255, 255))],
    "Blue":   [((94, 110, 60),  (130, 255, 255))],
    "Green":  [((36, 80, 60),   (86, 255, 255))],
    "Yellow": [((18, 110, 110), (34, 255, 255))],
}

# BGR colors for drawing each label
DRAW_COLORS = {
    "Red":    (60, 60, 255),
    "Blue":   (255, 140, 40),
    "Green":  (80, 255, 120),
    "Yellow": (0, 230, 255),
}


def preprocess_hsv(frame_bgr):
    """Blur slightly and convert to HSV. Blur reduces noise for stable masks."""
    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


def build_color_masks(hsv):
    """Return dict {color_name: binary_mask} after morphological cleanup."""
    masks = {}
    kernel = np.ones((5, 5), np.uint8)
    for name, ranges in COLOR_RANGES.items():
        mask_total = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask_total = m if mask_total is None else cv2.bitwise_or(mask_total, m)
        # Clean: open removes specks, close fills holes
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)
        masks[name] = mask_total
    return masks


def dominant_color_in_contour(masks, contour):
    """Return (color_name, pixel_count) with the most overlap inside the contour."""
    # Build a filled mask of the contour
    if not masks:
        return None, 0
    sample = next(iter(masks.values()))
    cmask = np.zeros(sample.shape, np.uint8)
    cv2.drawContours(cmask, [contour], -1, 255, thickness=-1)

    best_name, best_count = None, 0
    for name, m in masks.items():
        count = int(cv2.countNonZero(cv2.bitwise_and(m, cmask)))
        if count > best_count:
            best_name, best_count = name, count
    return best_name, best_count
