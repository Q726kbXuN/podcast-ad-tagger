#!/usr/bin/env python3
"""Regenerate ../../testdata: the reference values every port is checked against.

    python tools/make_testdata.py --embed-model /path/to/Qwen3-Embedding-0.6B-Q8_0.gguf

Everything is synthetic (generated audio, made-up sentences), so it can be shared freely.
"""
import argparse
import base64
import json
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from podcast_ad_tagger import audio, features, tagger  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "testdata"))
SR = audio.SR


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def synth():
    """30 s of 24 kHz stereo: talk-ish, a gap, a louder stereo music bed, talk, noise and a sweep."""
    rng = np.random.default_rng(1234)
    t = np.arange(30 * SR) / SR
    L = np.zeros_like(t); R = np.zeros_like(t)

    def talk(a, b, gain, f_lo, f_hi):
        m = (t >= a) & (t < b)
        f0 = f_lo + (f_hi - f_lo) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t[m]))
        ph = 2 * np.pi * np.cumsum(f0) / SR
        v = sum(np.sin(k * ph) / k for k in range(1, 12))
        env = np.clip(np.sin(2 * np.pi * 3.7 * t[m]), 0, None) ** 0.6
        sig = gain * env * v + 0.003 * rng.standard_normal(m.sum())
        L[m] += sig; R[m] += sig

    talk(0, 10, 0.12, 110, 180)
    m = (t >= 11) & (t < 21)                      # louder, wide stereo music bed with a voice on top
    chord_l = sum(np.sin(2 * np.pi * f * t[m]) for f in (220, 277.2, 329.6))
    chord_r = sum(np.sin(2 * np.pi * f * t[m]) for f in (246.9, 311.1, 370.0))
    L[m] += 0.08 * chord_l; R[m] += 0.08 * chord_r
    talk(11, 21, 0.25, 140, 220)
    talk(21, 26, 0.08, 100, 150)
    m = (t >= 26) & (t < 28)
    n = rng.standard_normal(m.sum()) * 0.05
    L[m] += n; R[m] += 0.5 * n
    m = t >= 28                                   # log sweep, left only
    L[m] += 0.1 * np.sin(2 * np.pi * 100 * (80 ** ((t[m] - 28) / 2) - 1) / np.log(80) * 2)
    pcm = np.clip(np.stack([L, R], axis=1), -1, 32767 / 32768)
    return np.round(pcm * 32768).astype("<i2")


def frames_of(mid, side=None):
    ex = audio.FrameExtractor()
    for i in range(0, len(mid), 7919):            # odd chunk size: streaming must not matter
        ex.feed(mid[i:i + 7919], None if side is None else side[i:i + 7919])
    return ex.finish()


def frame_rows(fr):
    return [[float(fr.scalars[n][i]) for n in audio.FRAME_NAMES] + [float(x) for x in fr.bands[i]]
            for i in range(len(fr))]


SEGMENTS = [(0.0, 2.4, 6), (2.4, 5.1, 9), (5.1, 9.8, 14), (11.0, 13.5, 8), (13.5, 16.2, 10),
            (16.2, 20.9, 15), (21.0, 23.2, 7), (23.2, 25.9, 9), (26.0, 27.9, 4), (28.0, 29.95, 3)]
SENTENCES = [
    "Welcome back to the show, today we're talking about how bridges are built.",
    "This episode is brought to you by Acme Mattress, use code SLEEP for twenty percent off your first order.",
    "So the engineers had to figure out how to keep the river open during construction.",
    "If you like the show, the best thing you can do is leave us a review and tell a friend.",
    "Support for this podcast comes from Example Bank, where checking accounts have no monthly fees.",
    "And that's when she realized the numbers in the old ledger didn't add up.",
    "Head to example dot com slash podcast to get your free trial today.",
    "We'll be right back after this.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", help="GGUF path; skips embeddings.json when omitted")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pcm = synth()
    with wave.open(os.path.join(OUT, "synth.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    x = pcm.astype(np.float32) / 32768.0
    mid = (x[:, 0] + x[:, 1]) * np.float32(0.5)
    side = (x[:, 0] - x[:, 1]) * np.float32(0.5)
    segs = [{"start": s, "end": e, "n_words": n} for s, e, n in SEGMENTS]
    ref = {"frame_columns": audio.FRAME_NAMES + [f"band{i}" for i in range(audio.N_BANDS)],
           "feature_names": features.FEATURE_NAMES, "segments": segs}
    for name, fr in (("stereo", frames_of(mid, side)), ("mono", frames_of(mid))):
        ref[name] = {"frames": frame_rows(fr),
                     "features": features.segment_features(fr, segs).tolist()}
    with open(os.path.join(OUT, "audio_reference.json"), "w") as fh:
        json.dump(ref, fh)

    # tagger: synthetic inputs in the model's usual range
    info = json.load(open(os.path.join(tagger.MODEL_DIR, "ad_tagger.json")))
    rng = np.random.default_rng(99)
    n = 40
    emb = rng.standard_normal((n, 1024)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    feats = (np.array(info["feature_mean"]) + np.array(info["feature_std"]) * rng.standard_normal((n, 28))).astype(np.float32)
    probs = tagger.Tagger().probs(emb, feats)
    with open(os.path.join(OUT, "tagger_reference.json"), "w") as fh:
        json.dump({"n": n, "emb": b64(emb), "feats": b64(feats), "probs": b64(probs)}, fh)

    # decoding: hand-made label patterns, as one-hot-ish probability rows
    cases = []
    for labels, p_sp in (
            ([0, 1, 1, 1, 0, 0, 0, 1, 1, 0], None),
            ([1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 3, 3, 3], None),
            ([0, 1, 0, 0, 0, 2, 2, 0, 2, 4, 4], None),
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.1, 0.3, 0.4, 0.2, 0.3, 0.26, 0.1, 0.5, 0.6, 0.1]),
            ([0, 1, 1, 0, 0, 0, 0, 0, 0], [0.2, 0.9, 0.9, 0.3, 0.1, 0.3, 0.3, 0.1, 0.4])):
        P = np.full((len(labels), 5), 0.02, np.float32)
        for i, c in enumerate(labels):
            P[i, c] = 0.9
        if p_sp is not None:
            for i, p in enumerate(p_sp):
                if labels[i] == 0:
                    P[i, 1] = p
        segs = [{"start": 3.0 * i, "end": 3.0 * i + 2.5} for i in range(len(labels))]
        spans = tagger.find_spans(P, segs)
        cases.append({"probs": P.tolist(), "segments": segs, "spans": spans,
                      "candidates": tagger.find_candidates(P, spans, segs)})
    with open(os.path.join(OUT, "decode_reference.json"), "w") as fh:
        json.dump({"min_segments": tagger.MIN_SEGMENTS, "max_gap": tagger.MAX_GAP,
                   "cand_min_p": tagger.CAND_MIN_P, "cases": cases}, fh, indent=1)

    if a.embed_model:
        from podcast_ad_tagger.embed import Embedder
        e = Embedder(a.embed_model).embed(SENTENCES)
        with open(os.path.join(OUT, "embedding_reference.json"), "w") as fh:
            json.dump({"note": "compare by cosine similarity; expect >= 0.999",
                       "texts": SENTENCES, "embeddings": b64(e), "dim": int(e.shape[1])}, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
