# motion-capture-fun
Learning motion capture with a webcam, using [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) and OpenCV.

Tracks your hands, body and face from a webcam, counts raised fingers, and lets you control music playback with hand gestures.

## Features

- **Hand tracking** – up to two hands, 21 landmarks each, drawn as a skeleton with Left/Right labels.
- **Finger counting** – detects which fingers are raised using 3D world landmarks, so it works with the palm facing the camera, tilted, or sideways. Raised fingertips are circled in green, and a smoothed count (no flicker) is shown above each hand along with the raised finger names and thumb reach.
- **Body pose / arm tracking** – draws shoulders and both arms, and works out whether each arm is held out to the side (shown as `left arm OUT (0.8)` under the hand).
- **Gesture music control (Windows)** – controls whatever app owns the Windows media session (Spotify, YouTube in a browser, etc.):

  | Gesture | Action |
  | --- | --- |
  | Fist | Pause |
  | 1 finger (index) | Play |
  | 2 fingers (index + middle) | Next song |
  | 3 fingers (index–ring, or thumb–middle) | Previous song |

  A gesture only counts while that arm is **held out to the side**, so normal hand movement in front of you doesn't trigger anything. It must be held for 0.5 s (a progress bar shows at the bottom), then all gestures are ignored for a 4 s cooldown. The status line shows the current gesture, cooldown and last action.
- **Face tracking** – 478 face landmarks plus expression scores (`jawOpen`, `eyeBlinkLeft`, ...). Detection runs, but the face overlay is currently commented out in `main.py` (`draw_face`).
- **FPS counter** and mirrored video so it feels like a mirror.

## Setup (Windows)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

The model files (hand, face and pose) are downloaded into `models/` on first run.

| Key | Action |
| --- | --- |
| `q` / `Esc` | Quit |
| `g` | Toggle gesture music control |
| `m` | Toggle full face mesh (only visible when the face overlay is enabled) |

## Project layout

| File | What it does |
| --- | --- |
| `main.py` | Webcam loop, runs the detectors and draws everything |
| `fingers.py` | Finger-up detection and count smoothing |
| `arms.py` | Decides whether an arm is held out to the side |
| `media.py` | Gesture classification, hold/cooldown trigger, Windows media control |

## Tuning

Thresholds live at the top of each module:

- `fingers.py` – `FINGER_STRAIGHT_DEG`, `THUMB_STRAIGHT_DEG`, `THUMB_OUT` (raise the angles if bent fingers count as up)
- `arms.py` – `ARM_OUT_REACH` (how far out the arm must be, in shoulder-widths), `MIN_VISIBILITY`
- `media.py` – `HOLD_SECONDS`, `COOLDOWN_SECONDS`, and the `GESTURES` / `GESTURE_ACTIONS` mappings
