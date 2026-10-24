"""Natural AI voice (Microsoft neural voices via edge-tts), played in the
background so the camera never freezes. Needs internet.
If edge-tts fails (e.g. no internet), it falls back to the basic Windows voice.
"""
import asyncio
import ctypes
import hashlib
import os
import queue
import threading

# ---------- change the voice here ----------
VOICE = "en-IN-NeerjaNeural"
# Other voices to try:
#   en-US-AriaNeural   en-US-JennyNeural   en-US-GuyNeural (male)
#   en-GB-SoniaNeural  hi-IN-SwaraNeural (Hindi)
# -------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_q = queue.Queue()
_fallback_engine = None


def _make_mp3(text):
    """Return the path of an mp3 for this text (cached after the first time)."""
    name = hashlib.md5((VOICE + "|" + text).encode("utf-8")).hexdigest() + ".mp3"
    path = os.path.join(CACHE_DIR, name)
    if not os.path.exists(path):
        import edge_tts

        async def run():
            await edge_tts.Communicate(text, VOICE).save(path)

        asyncio.run(run())
    return path


def _play_mp3(path):
    mci = ctypes.windll.winmm.mciSendStringW      # built into Windows
    mci("close vc", None, 0, 0)
    mci(f'open "{path}" type mpegvideo alias vc', None, 0, 0)
    mci("play vc wait", None, 0, 0)
    mci("close vc", None, 0, 0)


def _basic_voice(text):
    global _fallback_engine
    try:
        if _fallback_engine is None:
            import pyttsx3
            _fallback_engine = pyttsx3.init()
        _fallback_engine.say(text)
        _fallback_engine.runAndWait()
    except Exception as e:
        print("Voice not available:", e)


def _worker():
    while True:
        text = _q.get()
        try:
            _play_mp3(_make_mp3(text))
        except Exception as e:
            print("AI voice failed, using basic voice:", e)
            _basic_voice(text)


threading.Thread(target=_worker, daemon=True).start()


def speak(text):
    """Say the text out loud (always on)."""
    if not text.strip():
        return
    while not _q.empty():          # drop old waiting items so speech keeps up
        try:
            _q.get_nowait()
        except queue.Empty:
            break
    _q.put(text)
