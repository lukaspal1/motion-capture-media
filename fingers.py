"""Count how many fingers are held up on a hand.

Uses MediaPipe's hand *world* landmarks (real 3D positions in meters, centered
on the hand), so it works whether your palm faces the camera, is tilted, or
is turned sideways.

Hand landmark indices:
    0 wrist
    thumb  1 CMC, 2 MCP, 3 IP,  4 tip
    index  5 MCP, 6 PIP, 7 DIP, 8 tip
    middle 9 MCP, 10 PIP, 11 DIP, 12 tip
    ring   13 MCP, 14 PIP, 15 DIP, 16 tip
    pinky  17 MCP, 18 PIP, 19 DIP, 20 tip
"""

from collections import Counter, deque

import numpy as np

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# (MCP, PIP, DIP, TIP) for the four non-thumb fingers.
FINGER_JOINTS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]

# Tuning knobs. Raise the angles if bent fingers are counted as up.
FINGER_STRAIGHT_DEG = 135  # PIP joint must be at least this straight
THUMB_STRAIGHT_DEG = 130   # thumb IP joint must be at least this straight
THUMB_OUT = 0.2            # how far the thumb tip must stick out past the index knuckle, in palm sizes


def _joint_angle(a, b, c):
    """Angle at point b (degrees). 180 = perfectly straight."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def thumb_reach(world_landmarks):
    """How far the thumb tip sticks out sideways past the index knuckle, in palm sizes.

    Measured along the knuckle line (pinky knuckle -> index knuckle), so:
    positive = thumb out to the side, around 0 = resting against the index finger,
    negative = folded across the palm toward the pinky.
    """
    p = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
    palm_size = np.linalg.norm(p[9] - p[0])  # wrist to middle knuckle
    across = p[5] - p[17]
    across /= np.linalg.norm(across) + 1e-9
    return float(np.dot(p[4] - p[5], across) / palm_size)


def fingers_up(world_landmarks):
    """Return a list of 5 bools: [thumb, index, middle, ring, pinky]."""
    p = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])

    # Thumb: straight AND sticking out past the side of the hand (not folded across the palm).
    thumb_straight = _joint_angle(p[2], p[3], p[4]) > THUMB_STRAIGHT_DEG
    thumb_out = thumb_reach(world_landmarks) > THUMB_OUT
    states = [bool(thumb_straight and thumb_out)]

    # Other fingers: straight at the PIP joint, and tip farther from the wrist than the PIP.
    for mcp, pip, dip, tip in FINGER_JOINTS:
        straight = _joint_angle(p[mcp], p[pip], p[dip]) > FINGER_STRAIGHT_DEG
        extended = np.linalg.norm(p[tip] - p[0]) > np.linalg.norm(p[pip] - p[0])
        states.append(bool(straight and extended))
    return states


class CountSmoother:
    """Removes flicker by reporting the most common count over the last few frames."""

    def __init__(self, window=7):
        self.window = window
        self.history = {}

    def update(self, hand_label, count):
        hist = self.history.setdefault(hand_label, deque(maxlen=self.window))
        hist.append(count)
        return Counter(hist).most_common(1)[0][0]

    def forget_missing(self, visible_labels):
        for label in list(self.history):
            if label not in visible_labels:
                del self.history[label]
