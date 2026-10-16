"""Decide whether an arm is held out to the side, using MediaPipe body pose.

Pose landmark indices used here (the person's own left/right):
    11 left shoulder   12 right shoulder
    13 left elbow      14 right elbow
    15 left wrist      16 right wrist
"""

SHOULDERS = {"left": 11, "right": 12}
ELBOWS = {"left": 13, "right": 14}
WRISTS = {"left": 15, "right": 16}

# Tuning knobs.
ARM_OUT_REACH = 0.5    # wrist must be this many shoulder-widths outside its shoulder
MIN_VISIBILITY = 0.5    # ignore the pose if the shoulders aren't clearly visible


def nearest_arm(pose, hand_wrist):
    """Which arm ("left"/"right") does this hand belong to? Picks the closer pose wrist."""
    def dist2(side):
        w = pose[WRISTS[side]]
        return (w.x - hand_wrist.x) ** 2 + (w.y - hand_wrist.y) ** 2
    return min(WRISTS, key=dist2)


def arm_reach(pose, side, hand_wrist):
    """How far the hand is outside its shoulder, in shoulder-widths.

    0 = wrist directly in line with the shoulder, negative = in front of the body,
    1 = one shoulder-width out to the side. Returns None if the shoulders aren't visible.
    """
    left, right = pose[SHOULDERS["left"]], pose[SHOULDERS["right"]]
    if min(left.visibility, right.visibility) < MIN_VISIBILITY:
        return None
    shoulder_width = abs(left.x - right.x)
    if shoulder_width < 1e-3:  # standing sideways to the camera
        return None

    shoulder = pose[SHOULDERS[side]]
    mid_x = (left.x + right.x) / 2
    outward = 1 if shoulder.x > mid_x else -1  # which screen direction is "away from the body"
    # Use the hand tracker's wrist, it's more precise than the pose wrist.
    return (hand_wrist.x - shoulder.x) * outward / shoulder_width


def is_arm_out(pose, hand_wrist):
    """Return (is_out, side, reach) for the arm attached to this hand."""
    side = nearest_arm(pose, hand_wrist)
    reach = arm_reach(pose, side, hand_wrist)
    return reach is not None and reach > ARM_OUT_REACH, side, reach
