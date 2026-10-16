"""Webcam face + hand tracking with MediaPipe.

Run:  python main.py
Keys: q / Esc = quit,  m = toggle face mesh detail,  g = toggle gesture music control
Gestures: fist = pause, 1 (index) = play, 2 = next song, 3 = previous song
          (only counts while that arm is held out to the side)
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from arms import ELBOWS, SHOULDERS, WRISTS, is_arm_out
from fingers import FINGER_NAMES, CountSmoother, fingers_up, thumb_reach
from media import GestureTrigger, MediaController, classify

MODELS_DIR = Path(__file__).parent / "models"
MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
FACE_CONTOURS = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
FACE_MESH = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION


def ensure_model(name: str) -> Path:
    """Download a model file the first time it's needed."""
    path = MODELS_DIR / name
    if not path.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(MODEL_URLS[name], path)
    return path


def create_detectors():
    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model("hand_landmarker.task"))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model("face_landmarker.task"))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,  # expression scores like "jawOpen", "eyeBlinkLeft"
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model("pose_landmarker_lite.task"))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )
    return hands, face, pose


def to_pixels(landmarks, width, height):
    """Landmarks are normalized 0..1 — convert to pixel coordinates."""
    return [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]


def draw_connections(frame, points, connections, color, thickness=1):
    for c in connections:
        cv2.line(frame, points[c.start], points[c.end], color, thickness, cv2.LINE_AA)


FINGER_TIPS = [4, 8, 12, 16, 20]


def draw_hands(frame, result, smoother):
    """Draw each hand. Returns (finger_states, wrist_landmark) per hand,
    where finger_states is [thumb, index, middle, ring, pinky]."""
    h, w = frame.shape[:2]
    labels = []
    all_up = []
    for landmarks, world, handedness in zip(
        result.hand_landmarks, result.hand_world_landmarks, result.handedness
    ):
        pts = to_pixels(landmarks, w, h)
        label = handedness[0].category_name
        labels.append(label)

        up = fingers_up(world)
        all_up.append((up, landmarks[0]))
        count = smoother.update(label, sum(up))

        draw_connections(frame, pts, HAND_CONNECTIONS, (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        # Raised fingertips get a big green dot, lowered ones gray.
        for tip, is_up in zip(FINGER_TIPS, up):
            cv2.circle(frame, pts[tip], 9, (0, 255, 0) if is_up else (120, 120, 120), 2)

        # Big count above the hand, label + raw finger states under the wrist.
        top_x = min(x for x, _ in pts)
        top_y = min(y for _, y in pts)
        cv2.putText(frame, str(count), (top_x, max(top_y - 15, 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
        raised = ",".join(n for n, u in zip(FINGER_NAMES, up) if u) or "fist"
        cv2.putText(frame, f"{label}: {raised}  thumb {thumb_reach(world):+.2f}",
                    (pts[0][0] - 40, pts[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    smoother.forget_missing(labels)
    return all_up


def draw_arms(frame, pose):
    """Draw shoulders and both arms from the body pose."""
    h, w = frame.shape[:2]
    pts = to_pixels(pose, w, h)
    cv2.line(frame, pts[SHOULDERS["left"]], pts[SHOULDERS["right"]], (255, 0, 255), 2)
    for side in ("left", "right"):
        s, e, wr = SHOULDERS[side], ELBOWS[side], WRISTS[side]
        cv2.line(frame, pts[s], pts[e], (255, 0, 255), 2)
        cv2.line(frame, pts[e], pts[wr], (255, 0, 255), 2)


def gated_gestures(frame, pose, hands_up):
    """Classify each hand's gesture, but only keep it if that arm is out to the side."""
    h, w = frame.shape[:2]
    gestures = []
    for up, wrist in hands_up:
        out, side, reach = is_arm_out(pose, wrist) if pose else (False, None, None)
        text = "no body" if reach is None else f"{side} arm {'OUT' if out else 'in'} ({reach:.1f})"
        cv2.putText(frame, text, (int(wrist.x * w) - 40, int(wrist.y * h) + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if out else (0, 140, 255), 2)
        gestures.append(classify(up) if out else None)
    return gestures


def draw_gesture_status(frame, enabled, trigger, gesture, progress):
    h = frame.shape[0]
    if not enabled:
        text, color = "Gestures OFF (press g)", (120, 120, 120)
    elif trigger.cooldown_left() > 0:
        text, color = f"Cooldown {trigger.cooldown_left():.1f}s", (0, 140, 255)
    elif gesture:
        text, color = f"{gesture}: {'done' if trigger.fired else 'hold...'}", (0, 255, 255)
    else:
        text, color = "Gestures ON (arm out)  0=pause 1=play 2=next 3=prev", (255, 255, 255)
    cv2.putText(frame, text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if trigger.last_action:
        cv2.putText(frame, f"last: {trigger.last_action}", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Hold progress bar.
    if enabled and gesture:
        cv2.rectangle(frame, (10, h - 12), (10 + int(200 * progress), h - 6), color, -1)


# def draw_face(frame, result, show_mesh):
#     h, w = frame.shape[:2]
#     for landmarks in result.face_landmarks:
#         pts = to_pixels(landmarks, w, h)
#         if show_mesh:
#             draw_connections(frame, pts, FACE_MESH, (80, 80, 80), 1)
#         draw_connections(frame, pts, FACE_CONTOURS, (255, 200, 0), 1)

#     # Show a few expression scores (0..1) for the first face.
#     if result.face_blendshapes:
#         scores = {b.category_name: b.score for b in result.face_blendshapes[0]}
#         for i, name in enumerate(["jawOpen", "eyeBlinkLeft", "eyeBlinkRight", "mouthSmileLeft"]):
#             cv2.putText(frame, f"{name}: {scores[name]:.2f}", (10, 60 + i * 22),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (try a different index in cv2.VideoCapture).")

    hands, face, pose_detector = create_detectors()
    smoother = CountSmoother()
    trigger = GestureTrigger(MediaController())
    gestures_on = True
    show_mesh = False
    start = time.monotonic()
    prev = start

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror so it feels like a mirror

            # MediaPipe wants RGB; OpenCV gives BGR.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)

            hand_result = hands.detect_for_video(image, timestamp_ms)
            face_result = face.detect_for_video(image, timestamp_ms)
            pose_result = pose_detector.detect_for_video(image, timestamp_ms)
            pose = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None

            # draw_face(frame, face_result, show_mesh)
            if pose:
                draw_arms(frame, pose)
            hands_up = draw_hands(frame, hand_result, smoother)

            # Use the first hand that's making a known gesture with its arm out.
            gesture = next((g for g in gated_gestures(frame, pose, hands_up) if g), None)
            progress = trigger.update(gesture if gestures_on else None)
            draw_gesture_status(frame, gestures_on, trigger, gesture, progress)

            now = time.monotonic()
            fps = 1 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Motion Capture Fun", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_mesh = not show_mesh
            if key == ord("g"):
                gestures_on = not gestures_on
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        face.close()
        pose_detector.close()


if __name__ == "__main__":
    main()
