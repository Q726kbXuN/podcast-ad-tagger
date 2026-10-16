"""The tagger (ONNX) and the rules that turn its per-segment output into ad spans."""
import json
import os

import numpy as np

CLASSES = ["content", "sponsor_ad", "house_promo", "membership_pitch", "boilerplate"]
MIN_SEGMENTS = 2        # a span needs at least this many segments
MAX_GAP = 2             # bridge up to this many "content" segments inside a span
CAND_MIN_P = 0.25       # "unsure" candidates: sponsor probability at least this ...
CAND_MIN_SEGMENTS = 2   # ... for at least this many segments

MODEL_DIR = os.environ.get("PODCAST_AD_TAGGER_MODEL_DIR") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "model"))


class Tagger:
    def __init__(self, onnx_path=None):
        import onnxruntime as ort
        path = onnx_path or os.path.join(MODEL_DIR, "ad_tagger.onnx")
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def probs(self, emb, feats):
        """emb [N, 1024] unit vectors, feats [N, 28] raw features -> probs [N, 5]."""
        out = self.session.run(None, {"emb": np.asarray(emb, np.float32)[None],
                                      "feats": np.asarray(feats, np.float32)[None]})
        return out[0][0]


def runs(labels):
    """Per-segment class ids -> [(first, last, class_id)], smoothed: short gaps of content
    inside a run are bridged, and runs shorter than MIN_SEGMENTS are dropped."""
    out, i, n = [], 0, len(labels)
    while i < n:
        c = labels[i]
        if c == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and (labels[j + 1] == c or
                             (j + 1 + MAX_GAP < n and labels[j + 1] == 0 and
                              any(labels[j + 1 + k] == c for k in range(1, MAX_GAP + 1)))):
            j += 1
        if j - i + 1 >= MIN_SEGMENTS:
            out.append((i, j, int(c)))
        i = j + 1
    return out


def find_spans(probs, segments):
    """-> list of {start, end, type, p, first_segment, last_segment}: start/end in seconds from
    the segments, p the span's mean probability for its type."""
    spans = []
    for i, j, c in runs(probs.argmax(axis=1)):
        spans.append({"start": round(float(segments[i]["start"]), 2), "end": round(float(segments[j]["end"]), 2),
                      "type": CLASSES[c], "p": round(float(probs[i:j + 1, c].mean()), 3),
                      "first_segment": i, "last_segment": j})
    return spans


def find_candidates(probs, spans, segments):
    """Stretches outside every span that look like they might be a sponsor read: worth showing
    a listener as "maybe an ad" instead of skipping.  -> list of {start, end, p}."""
    p_sp = probs[:, CLASSES.index("sponsor_ad")]
    n = len(p_sp)
    inside = np.zeros(n, bool)
    for sp in spans:
        inside[sp["first_segment"]:sp["last_segment"] + 1] = True
    hot = (~inside) & (p_sp >= CAND_MIN_P)
    out, i = [], 0
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (hot[j + 1] or (j + 2 < n and hot[j + 2] and not inside[j + 1])):
            j += 1
        if j - i + 1 >= CAND_MIN_SEGMENTS:
            out.append({"start": round(float(segments[i]["start"]), 2), "end": round(float(segments[j]["end"]), 2),
                        "p": round(float(p_sp[i:j + 1].mean()), 3)})
        i = j + 1
    return out


def model_info():
    with open(os.path.join(MODEL_DIR, "ad_tagger.json")) as fh:
        return json.load(fh)
