"""
utils.py
Helper utilities: FPS counter, futuristic UI overlays, screenshot saver, sound.
"""

import cv2
import numpy as np
import time
import os
import threading

# ---------- Theme (Cyberpunk / Neon) ----------
NEON_CYAN = (255, 255, 0)        # BGR
NEON_PINK = (200, 50, 255)
NEON_GREEN = (80, 255, 120)
NEON_YELLOW = (0, 255, 255)
NEON_RED = (80, 80, 255)
PANEL_BG = (20, 10, 30)          # dark purple-ish
WHITE = (255, 255, 255)


class FPSCounter:
    """Smooth FPS counter using exponential moving average."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self.last_time = time.time()

    def update(self) -> float:
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            instant = 1.0 / dt
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant
        return self.fps


def draw_translucent_panel(frame, x, y, w, h, color=PANEL_BG, alpha=0.55, border=NEON_CYAN):
    """Draw a semi-transparent panel with a neon border."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Neon border (double line for glow effect)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x - 2, y - 2), (x + w + 2, y + h + 2), border, 1, cv2.LINE_AA)
    # Corner accents
    c = 14
    for (cx, cy) in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
        cv2.line(frame, (cx, cy), (cx + (c if cx == x else -c), cy), NEON_PINK, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + (c if cy == y else -c)), NEON_PINK, 2, cv2.LINE_AA)


def draw_neon_text(frame, text, org, scale=0.6, color=NEON_CYAN, thickness=1):
    """Text with a soft glow underneath."""
    x, y = org
    # Glow
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (color[0] // 2, color[1] // 2, color[2] // 2), thickness + 2, cv2.LINE_AA)
    # Main
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_scan_line(frame, frame_idx: int):
    """Animated horizontal AI-scan line moving up and down."""
    h, w = frame.shape[:2]
    period = 120  # frames per cycle
    t = (frame_idx % period) / period
    # Bounce: 0 -> 1 -> 0
    pos = int(abs(2 * t - 1) * (h - 1))
    overlay = frame.copy()
    cv2.line(overlay, (0, pos), (w, pos), NEON_CYAN, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


def draw_tracking_box(frame, x, y, w, h, color=NEON_GREEN):
    """Futuristic bracket-style bounding box (instead of full rectangle)."""
    l = max(10, min(w, h) // 5)  # bracket length
    # Top-left
    cv2.line(frame, (x, y), (x + l, y), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x, y + l), color, 2, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (x + w, y), (x + w - l, y), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x + w, y), (x + w, y + l), color, 2, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (x, y + h), (x + l, y + h), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y + h), (x, y + h - l), color, 2, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - l, y + h), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - l), color, 2, cv2.LINE_AA)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_screenshot(frame, folder: str = "captures", prefix: str = "capture") -> str:
    """Save the current frame to /captures with a timestamped filename."""
    ensure_dir(folder)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    path = os.path.join(folder, f"{prefix}_{ts}_{ms:03d}.png")
    cv2.imwrite(path, frame)
    return path


def play_beep_async():
    """Play a short beep without blocking the main loop. Falls back silently."""
    def _beep():
        try:
            # Windows
            import winsound
            winsound.Beep(1000, 80)
        except Exception:
            # Unix-ish terminal bell
            try:
                print("\a", end="", flush=True)
            except Exception:
                pass
    threading.Thread(target=_beep, daemon=True).start()
