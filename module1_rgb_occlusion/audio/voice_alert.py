"""
voice_alert.py
---------------
Text-to-speech alert system: speaks out loud when a survivor signal
(tap or voice) is detected, instead of just printing text.

On Mac, this uses the built-in `say` command (no extra installation
needed). On other platforms, it falls back to the `pyttsx3` library.

This is meant to be used alongside live_demo.py for a more impactful,
audible demo - e.g., for showing the project to a professor.
"""

import sys
import subprocess

ALERT_MESSAGES = {
    "tap": "Human detected. Tapping signal identified.",
    "voice": "Human detected. Voice signal identified.",
}


def speak(message):
    """
    Speaks the given message out loud.
    Uses macOS's built-in `say` command if available, otherwise
    falls back to pyttsx3 (cross-platform).
    """
    if sys.platform == "darwin":
        subprocess.run(["say", message])
    else:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
        except ImportError:
            print(f"[voice alert unavailable] {message}")


def announce_detection(label):
    """
    Speaks an alert message for a given detected label ("tap" or "voice").
    Does nothing for "noise" (no alert needed for background sound).
    """
    message = ALERT_MESSAGES.get(label)
    if message:
        speak(message)


if __name__ == "__main__":
    # Quick manual test
    print("Testing voice alerts...")
    announce_detection("tap")
    announce_detection("voice")