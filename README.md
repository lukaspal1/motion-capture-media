# motion-capture-fun
Learning motion capture with a webcam, using [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) and OpenCV.

Tracks your face (478 landmarks + expression scores) and up to two hands (21 landmarks each).

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

The model files are downloaded into `models/` on first run.

| Key | Action |
| --- | --- |
| `q` / `Esc` | Quit |
| `m` | Toggle full face mesh |
