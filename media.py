"""Control music playback with hand gestures (Windows).

Talks to Windows' media session (the same thing the media overlay/lock screen
uses), so it works with Spotify, YouTube in a browser, etc. It has separate
Play and Pause commands, so a fist always pauses and never resumes.
"""

import asyncio
import threading
import time

from winrt.windows.media.control import (
    GlobalSystemMediaTransportControlsSessionManager as SessionManager,
)

# Exact finger patterns: [thumb, index, middle, ring, pinky]. A gesture can have
# several patterns, e.g. people show "3" either with or without the thumb.
# Only the index finger counts as "1" here, a middle- or pinky-only 1 won't match.
GESTURES = {
    "fist": [(False, False, False, False, False)],
    "one": [(False, True, False, False, False)],
    "two": [(False, True, True, False, False)],
    "three": [(False, True, True, True, False), (True, True, True, False, False)],
}
GESTURE_ACTIONS = {"fist": "pause", "one": "play", "two": "next", "three": "previous"}

HOLD_SECONDS = 0.5      # how long a gesture must be held before it fires
COOLDOWN_SECONDS = 4.0  # after any gesture fires, ignore all gestures for this long


def classify(up):
    """Return the gesture name for a list of finger states, or None."""
    for name, patterns in GESTURES.items():
        if tuple(up) in patterns:
            return name
    return None


class MediaController:
    """Sends play/pause to whatever app currently owns Windows' media session."""

    def play(self):
        self._run_in_background("try_play_async")

    def pause(self):
        self._run_in_background("try_pause_async")

    def next(self):
        self._run_in_background("try_skip_next_async")

    def previous(self):
        self._run_in_background("try_skip_previous_async")

    def _run_in_background(self, method):
        # The Windows API is async; run it off the camera loop so video never stutters.
        threading.Thread(target=asyncio.run, args=(self._send(method),), daemon=True).start()

    @staticmethod
    async def _send(method):
        manager = await SessionManager.request_async()
        session = manager.get_current_session()
        if session is None:
            print("No media session found. Is something playing (or paused) in Spotify?")
            return
        await getattr(session, method)()


class GestureTrigger:
    """Fires an action once a gesture is held steady, then waits for it to change.

    After firing, every gesture is ignored for a cooldown period.
    """

    def __init__(self, controller, hold_seconds=HOLD_SECONDS, cooldown_seconds=COOLDOWN_SECONDS):
        self.controller = controller
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.current = None      # gesture being held right now
        self.since = 0.0         # when it started being held
        self.fired = False       # already triggered for this hold?
        self.cooldown_until = 0.0
        self.last_action = ""

    def cooldown_left(self):
        """Seconds until gestures are accepted again (0 if ready)."""
        return max(self.cooldown_until - time.monotonic(), 0.0)

    def update(self, gesture):
        """Call every frame. Returns hold progress 0..1 for the current gesture."""
        now = time.monotonic()
        if gesture != self.current:
            self.current, self.since, self.fired = gesture, now, False
        if gesture is None or now < self.cooldown_until:
            return 0.0

        # A hold that started during the cooldown only counts from when it ended.
        held = now - max(self.since, self.cooldown_until)
        progress = min(held / self.hold_seconds, 1.0)
        if progress >= 1.0 and not self.fired:
            self.fired = True
            self.cooldown_until = now + self.cooldown_seconds
            action = GESTURE_ACTIONS[gesture]
            getattr(self.controller, action)()
            self.last_action = action
            print(f"{gesture} -> {action}")
        return progress
