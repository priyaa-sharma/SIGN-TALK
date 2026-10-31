import os
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import HandTracker, extract_features, draw_hand
from voice import speak
from words_common import (CLIP_SECONDS, WORD_DISPLAY, clip_to_sequence,
                          frame_features, sequence_to_vector)
from signmatic import SignMaticTracker, pool_sequences

APP_NAME = "SignTalk"
TITLE_FONT = [r"C:\Windows\Fonts\bahnschrift.ttf", r"C:\Windows\Fonts\bahnschrift.ttf"]
BODY_FONT = [r"C:\Windows\Fonts\segoeui.ttf", "arial.ttf"]
HEADER = (92, 78, 78)
STATUS_BOX = (152, 134, 134)
SENTENCE_BOX = (209, 208, 208)
BOX_OUTLINE = (152, 134, 134)
ACCENT = (92, 78, 78)
TEXT = (0, 0, 0)
HINT_TEXT = (0,0,0)
TITLE_COLOR = (209, 208, 208)

AUTO_CONF = 0.75
NEED_AGREE = 2
CHECK_EVERY = 0.4
COOLDOWN = 2.0


def load_font(paths, size):
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


F_TITLE = load_font(TITLE_FONT, 40)
F_STATUS = load_font(BODY_FONT, 22)
F_SENT = load_font(BODY_FONT, 32)


def fit_text(draw, text, font, max_w):
    """If the text is too long, keep only its latest part."""
    text = text.strip()
    while text and draw.textlength(text, font=font) > max_w:
        text = text[1:]
    return text


def render_ui(frame, status, sentence):
    """Header, status box and sentence box drawn on top of the video."""
    H, W = frame.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    od.rectangle([0, 0, W, 60], fill=HEADER + (235,))

    status = fit_text(od, status, F_STATUS, W - 60)
    pw = od.textlength(status, font=F_STATUS) + 44
    od.rectangle([W / 2 - pw / 2, 82, W / 2 + pw / 2, 118], fill=STATUS_BOX + (240,))

    shown = fit_text(od, sentence, F_SENT, W - 100)
    hint = not shown
    if hint:
        shown = "your sentence appears here"
    tw = od.textlength(shown, font=F_STATUS if hint else F_SENT) + 50
    od.rectangle([W / 2 - tw / 2, H - 92, W / 2 + tw / 2, H - 26],
                 fill=SENTENCE_BOX + (240,), outline=BOX_OUTLINE + (255,), width=3)

    img = Image.alpha_composite(img, overlay)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 36), APP_NAME, font=F_TITLE, fill=TITLE_COLOR, anchor="mm")
    d.text((W / 2, 100), status, font=F_STATUS, fill=TEXT, anchor="mm")
    if hint:
        d.text((W / 2, H - 59), shown, font=F_STATUS, fill=HINT_TEXT, anchor="mm")
    else:
        d.text((W / 2, H - 59), shown, font=F_SENT, fill=TEXT, anchor="mm")
    d.rectangle([0, 0, W - 1, H - 1], outline=ACCENT, width=6)
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


model = joblib.load("model.pkl")
word_model = joblib.load("words_model.pkl") if os.path.exists("words_model.pkl") else None
if word_model is not None:
    word_model.n_jobs = 1
signmatic_model = (
    joblib.load("signmatic_model.pkl")
    if os.path.exists("signmatic_model.pkl") else None
)
if signmatic_model is not None:
    signmatic_model.n_jobs = 1
auto_ok = word_model is not None and "NONE" in word_model.classes_
signmatic_ok = signmatic_model is not None
tracker = HandTracker()
pose_tracker = SignMaticTracker() if signmatic_ok else None
cap = cv2.VideoCapture(0)

WINDOW = deque(maxlen=12)
HOLD_SECONDS = 1.0
MIN_CONF = 0.70
sentence, current, since, added = "", None, time.time(), False
word_note, note_until = "", 0.0
buf = deque()
signmatic_buf = deque(maxlen=30)
last_check, last_auto_label, auto_streak = 0.0, None, 0
cooldown_until, letters_paused_until = 0.0, 0.0


def add_word(label, p):
    """Add a recognised word to the sentence and speak it."""
    global sentence, word_note, note_until, cooldown_until, letters_paused_until
    t = time.time()
    text = WORD_DISPLAY.get(label, label)
    if sentence and not sentence.endswith(" "):
        sentence += " "
    sentence += text + " "
    speak(text)
    word_note, note_until = f"Output: {text} ({p:.0%})", t + 3
    cooldown_until = letters_paused_until = t + COOLDOWN
    buf.clear()


while True:
    ok, frame = cap.read()
    if not ok:
        break
    raw_frame = frame
    frame = cv2.flip(frame, 1)
    landmarks = tracker.find(frame)
    if landmarks:
        draw_hand(frame, landmarks)
    now = time.time()
    pred, conf = None, 0.0

    buf.append((now, frame_features(landmarks) if landmarks else None))
    while buf and now - buf[0][0] > CLIP_SECONDS:
        buf.popleft()
    if pose_tracker is not None:
        signmatic_buf.append(pose_tracker.frame_features(raw_frame))

    if ((auto_ok or signmatic_ok) and now >= cooldown_until
            and now - last_check >= CHECK_EVERY
            and ((auto_ok and len(buf) > 15
                  and buf[-1][0] - buf[0][0] >= 1.5)
                 or (signmatic_ok and len(signmatic_buf) == 30))):
        last_check = now
        legacy_candidate = None
        signmatic_candidate = None
        if auto_ok and len(buf) > 15 and buf[-1][0] - buf[0][0] >= 1.5:
            seq = clip_to_sequence([f for _, f in buf])
            if seq is not None:
                probs = word_model.predict_proba([sequence_to_vector(seq)])[0]
                i = int(np.argmax(probs))
                label, p = word_model.classes_[i], probs[i]
                if label != "NONE":
                    legacy_candidate = (str(label), float(p))
                    if p >= 0.5:
                        letters_paused_until = now + 1.2

        if signmatic_ok and len(signmatic_buf) == 30:
            features = pool_sequences(
                np.asarray(signmatic_buf, dtype=np.float32)[None, ...]
            )
            probs = signmatic_model.predict_proba(features)[0]
            i = int(np.argmax(probs))
            label, p = str(signmatic_model.classes_[i]).upper(), float(probs[i])
            if label != "IDLE":
                signmatic_candidate = (label, p)
                if p >= 0.5:
                    letters_paused_until = now + 1.2

        candidate = None
        if legacy_candidate and legacy_candidate[1] >= AUTO_CONF:
            candidate = ("legacy", *legacy_candidate)
        elif signmatic_candidate and signmatic_candidate[1] >= AUTO_CONF:
            candidate = ("signmatic", *signmatic_candidate)

        if candidate is None:
            last_auto_label, auto_streak = None, 0
        else:
            source, label, p = candidate
            candidate_key = f"{source}:{label}"
            if candidate_key == last_auto_label:
                auto_streak += 1
            else:
                last_auto_label, auto_streak = candidate_key, 1
            if auto_streak >= NEED_AGREE:
                add_word(label, p)
                last_auto_label, auto_streak = None, 0

    if landmarks:
        probs = model.predict_proba([extract_features(landmarks)])[0]
        i = int(np.argmax(probs))
        if probs[i] >= MIN_CONF:
            pred, conf = model.classes_[i], probs[i]
        WINDOW.append(pred)
    else:
        WINDOW.clear()

    stable = Counter(WINDOW).most_common(1)[0][0] if WINDOW else None
    if now < letters_paused_until:
        current, since, added = stable, now, False
    elif stable != current:
        current, since, added = stable, now, False
    elif current and not added and now - since >= HOLD_SECONDS:
        if current == "SPACE":
            sentence += " "
        elif current == "DEL":
            sentence = sentence[:-1]
        else:
            sentence += current if len(current) == 1 else f" {current} "
            speak(WORD_DISPLAY.get(current, current))
        added = True

    if now < note_until:
        status = word_note
    elif auto_ok or signmatic_ok:
        status = f"Output: {pred or '-'}  ({conf:.0%})"
    elif word_model is None and signmatic_model is None:
        status = f"Output: {pred or '-'}  ({conf:.0%})   |   run train_words.py"
    else:
        status = f"Output: {pred or '-'}  ({conf:.0%})   |   record NONE clips or run train_signmatic.py"
    frame = render_ui(frame, status, sentence)
    cv2.imshow(APP_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if pose_tracker is not None:
    pose_tracker.close()