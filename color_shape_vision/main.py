"""
main.py
Real-Time Color & Shape Detection System
----------------------------------------
Cyberpunk-style AI vision dashboard built with OpenCV + NumPy.

Controls:
    S -> save screenshot
    Q -> quit

Run:
    python main.py
"""

import cv2
import numpy as np
import sys
from collections import Counter

from utils import (
    FPSCounter, draw_translucent_panel, draw_neon_text, draw_scan_line,
    draw_tracking_box, save_screenshot, play_beep_async, ensure_dir,
    NEON_CYAN, NEON_PINK, NEON_GREEN, NEON_YELLOW, WHITE,
)
from color_detection import (
    preprocess_hsv, build_color_masks, dominant_color_in_contour, DRAW_COLORS,
)
from shape_detection import find_contours, classify_shape


CAPTURE_DIR = "captures"
WINDOW_NAME = "AI Vision  |  Color & Shape Detection"


def open_webcam(index: int = 0):
    """Open the default webcam with friendly error handling."""
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Is another app using it?")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def detect_objects(frame):
    """Run full color+shape pipeline. Returns list of detections."""
    hsv = preprocess_hsv(frame)
    masks = build_color_masks(hsv)

    detections = []
    for color_name, mask in masks.items():
        for contour in find_contours(mask):
            shape = classify_shape(contour)
            if shape == "Unknown":
                continue
            # Confirm dominant color matches (helps reduce false positives)
            dom, _ = dominant_color_in_contour(masks, contour)
            if dom != color_name:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append({
                "color": color_name,
                "shape": shape,
                "bbox": (x, y, w, h),
                "contour": contour,
            })
    return detections


def draw_dashboard(frame, detections, fps, frame_idx, captures_saved):
    """Render the cyberpunk dashboard overlay."""
    h, w = frame.shape[:2]

    # Animated scan line
    draw_scan_line(frame, frame_idx)

    # --- Top header bar ---
    draw_translucent_panel(frame, 10, 10, w - 20, 50, border=NEON_CYAN)
    draw_neon_text(frame, "AI VISION  //  COLOR + SHAPE DETECTION",
                   (24, 42), scale=0.7, color=NEON_CYAN, thickness=2)
    draw_neon_text(frame, f"FPS {fps:5.1f}", (w - 140, 42),
                   scale=0.7, color=NEON_GREEN, thickness=2)

    # --- Left stats panel ---
    panel_w, panel_h = 230, 200
    px, py = 10, 75
    draw_translucent_panel(frame, px, py, panel_w, panel_h, border=NEON_PINK)
    draw_neon_text(frame, "DETECTION STATS", (px + 14, py + 28),
                   scale=0.6, color=NEON_PINK, thickness=2)

    total = len(detections)
    colors = Counter(d["color"] for d in detections)
    shapes = Counter(d["shape"] for d in detections)

    draw_neon_text(frame, f"Total Objects : {total}", (px + 14, py + 60),
                   scale=0.5, color=WHITE)
    draw_neon_text(frame, "Shapes:", (px + 14, py + 88), scale=0.5, color=NEON_YELLOW)
    row = py + 108
    for s in ["Circle", "Triangle", "Rectangle", "Square"]:
        draw_neon_text(frame, f"  {s:<10}: {shapes.get(s, 0)}", (px + 14, row),
                       scale=0.45, color=WHITE)
        row += 18

    # --- Right colors panel ---
    cpx = w - 230 - 10
    cpy = 75
    cph = 140
    draw_translucent_panel(frame, cpx, cpy, 230, cph, border=NEON_GREEN)
    draw_neon_text(frame, "COLOR CHANNELS", (cpx + 14, cpy + 28),
                   scale=0.6, color=NEON_GREEN, thickness=2)
    row = cpy + 56
    for c in ["Red", "Blue", "Green", "Yellow"]:
        cnt = colors.get(c, 0)
        bar_w = min(140, cnt * 28)
        cv2.rectangle(frame, (cpx + 14, row - 10), (cpx + 14 + 140, row + 2),
                      (60, 60, 60), 1)
        cv2.rectangle(frame, (cpx + 14, row - 10),
                      (cpx + 14 + bar_w, row + 2), DRAW_COLORS[c], -1)
        draw_neon_text(frame, f"{c:<7} {cnt}", (cpx + 160, row),
                       scale=0.45, color=WHITE)
        row += 22

    # --- Bottom hint bar ---
    draw_translucent_panel(frame, 10, h - 40, w - 20, 30, border=NEON_CYAN, alpha=0.5)
    draw_neon_text(frame,
                   f"[S] Save Screenshot   [Q] Quit   Auto-Saved: {captures_saved}",
                   (24, h - 19), scale=0.5, color=NEON_CYAN)

    # --- Draw each detection ---
    for d in detections:
        x, y, w_, h_ = d["bbox"]
        col = DRAW_COLORS[d["color"]]
        # Contour boundary
        cv2.drawContours(frame, [d["contour"]], -1, col, 2, cv2.LINE_AA)
        # Futuristic bracket tracking box
        draw_tracking_box(frame, x, y, w_, h_, color=col)
        # Label chip
        label = f"{d['color']} {d['shape']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx, ly = x, max(0, y - th - 10)
        cv2.rectangle(frame, (lx, ly), (lx + tw + 14, ly + th + 10),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (lx, ly), (lx + tw + 14, ly + th + 10),
                      col, 1, cv2.LINE_AA)
        draw_neon_text(frame, label, (lx + 7, ly + th + 3),
                       scale=0.55, color=col, thickness=1)


def signature(detections):
    """Hashable summary of current detections, used to spot 'new' objects."""
    return tuple(sorted((d["color"], d["shape"]) for d in detections))


def main():
    ensure_dir(CAPTURE_DIR)
    cap = open_webcam(0)
    fps_counter = FPSCounter()
    frame_idx = 0
    prev_sig = ()
    captures_saved = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[ERROR] Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)  # mirror, feels natural for webcams
            detections = detect_objects(frame)
            fps = fps_counter.update()

            # Auto snapshot when a new object signature appears
            sig = signature(detections)
            if detections and sig != prev_sig and set(sig) - set(prev_sig):
                save_screenshot(frame.copy(), CAPTURE_DIR, prefix="auto")
                captures_saved += 1
                play_beep_async()
            prev_sig = sig

            draw_dashboard(frame, detections, fps, frame_idx, captures_saved)
            cv2.imshow(WINDOW_NAME, frame)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                break
            if key in (ord('s'), ord('S')):
                path = save_screenshot(frame, CAPTURE_DIR, prefix="manual")
                captures_saved += 1
                print(f"[INFO] Saved screenshot -> {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
