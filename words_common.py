import numpy as np

SEQ_LEN = 30            # every clip is squeezed to 30 frames
CLIP_SECONDS = 2.0      # how long you sign for
COUNTDOWN = 2.0         # get-ready time before each clip

WORD_DISPLAY = {
    "THANKS": "THANK YOU",
    "THANKYOU": "THANK YOU",
    "ILOVEYOU": "I LOVE YOU",
}


def frame_features(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist_xy = pts[0, :2].copy()
    pts = pts - pts[0]
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts = pts / scale
    return np.concatenate([pts.flatten(), wrist_xy])


def clip_to_sequence(frames):
   
    valid = np.array([i for i, f in enumerate(frames) if f is not None])
    if len(frames) < 10 or len(valid) < 0.5 * len(frames):
        return None
    filled = []
    for i in range(len(frames)):
        j = valid[np.argmin(np.abs(valid - i))]   # nearest frame with a hand
        filled.append(frames[j])
    filled = np.array(filled)
    pick = np.linspace(0, len(filled) - 1, SEQ_LEN).round().astype(int)
    return filled[pick]


def sequence_to_vector(seq):
    wrist_move = seq[:, 63:65] - seq[0, 63:65]
    return np.concatenate([seq.flatten(), wrist_move.flatten()])
