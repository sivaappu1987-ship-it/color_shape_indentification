# Real-Time Color & Shape Detection System

A modern, cyberpunk-styled AI computer-vision app built with **Python**, **OpenCV**, and **NumPy**. It opens your webcam and detects **colors** (Red, Blue, Green, Yellow) and **shapes** (Circle, Triangle, Rectangle, Square) in real time, drawing neon overlays, a futuristic dashboard, an FPS counter, and detection statistics.

## ✨ Features

- Real-time HSV color filtering (Red / Blue / Green / Yellow)
- Shape classification (Circle / Triangle / Rectangle / Square)
- Contour boundaries + bracket-style tracking boxes
- Live labels: `Red Circle`, `Blue Rectangle`, `Green Triangle`, …
- Cyberpunk UI: neon borders, transparent info panels, animated AI scan line
- Real-time FPS counter (smoothed)
- Detection stats panel (total objects, per-shape, per-color)
- Multiple simultaneous detections
- **Auto snapshot** when a new object appears
- **S** → save screenshot, **Q** → quit
- Optional beep sound on detection
- Webcam error handling

## 📁 Project Structure

```
vision/
├── main.py                # Entry point — webcam loop + dashboard rendering
├── color_detection.py     # HSV ranges + mask building
├── shape_detection.py     # Contour + polygon shape classifier
├── utils.py               # FPS, neon UI helpers, screenshot, sound
├── requirements.txt
├── README.md
└── captures/              # Auto-created — saved screenshots land here
```

## ⚙️ Setup

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt
```

## ▶️ Run

```bash
python main.py
```

The webcam opens automatically inside a window titled **“AI Vision | Color & Shape Detection”**.

### Controls

| Key | Action            |
|-----|-------------------|
| `S` | Save screenshot   |
| `Q` | Quit application  |

All screenshots go to the **`captures/`** folder (auto-created).

## 🖼️ Sample Output Explained

When you hold up a colored object (e.g. a red ball, a blue sticky-note, a green triangle of paper):

- A **colored contour** outlines the object.
- A **bracket tracking box** wraps it with neon corners.
- A **label chip** above shows e.g. `Red Circle` in the matching neon color.
- The **left panel** updates: `Total Objects: 3`, `Circle: 1`, `Square: 2`, …
- The **right panel** shows per-color bar meters.
- The **top bar** shows the project title and a live **FPS** value.
- A faint **scan line** sweeps the frame, giving an AI-scanner feel.
- When a brand-new object appears, a snapshot is auto-saved into `captures/` and a short beep plays.

## 🧠 How it works (short version)

1. Frame → Gaussian blur → convert to **HSV**.
2. For each color, build a binary mask using HSV ranges + morphological open/close.
3. Find external contours per mask, filter tiny ones by area.
4. Approximate each contour with `cv2.approxPolyDP`:
   - 3 sides → **Triangle**
   - 4 sides + aspect ratio ≈ 1 → **Square**, otherwise **Rectangle**
   - many sides + high circularity (`4πA/P²`) → **Circle**
5. Cross-check the dominant color inside the contour to reduce false positives.
6. Render the cyberpunk dashboard with translucent panels and neon text.

## 🛠️ Troubleshooting

- **Webcam doesn’t open** → make sure no other app (Zoom, Teams, browser) is using it. Try changing `open_webcam(0)` to `open_webcam(1)` in `main.py`.
- **Colors not detected well** → lighting matters a lot. Try brighter, more even lighting, or widen the HSV ranges in `color_detection.py`.
- **Low FPS** → reduce the capture resolution in `open_webcam()` (e.g. 640×480).

## 📜 License

MIT — free to use for portfolio, hackathons, and learning.
