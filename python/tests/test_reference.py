"""The Python package against ../../testdata (the same files the Swift and Kotlin ports use).

    pytest python/tests
    PODCAST_AD_TAGGER_GGUF=/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf pytest python/tests   # adds the embedder
"""
import base64
import json
import os
import wave

import numpy as np
import pytest

from podcast_ad_tagger import audio, features, tagger

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata")


def load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


def f32(s, shape):
    return np.frombuffer(base64.b64decode(s), "<f4").reshape(shape)


def synth_frames(stereo):
    with wave.open(os.path.join(DATA, "synth.wav")) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2).astype(np.float32) / 32768.0
    mid = (x[:, 0] + x[:, 1]) * np.float32(0.5)
    ex = audio.FrameExtractor()
    ex.feed(mid, (x[:, 0] - x[:, 1]) * np.float32(0.5) if stereo else None)
    return ex.finish()


@pytest.mark.parametrize("mode", ["stereo", "mono"])
def test_audio(mode):
    ref = load("audio_reference.json")
    fr = synth_frames(mode == "stereo")
    rows = np.array([[fr.scalars[n][i] for n in audio.FRAME_NAMES] + list(fr.bands[i]) for i in range(len(fr))])
    want = np.array(ref[mode]["frames"])
    assert rows.shape == want.shape
    f0 = audio.FRAME_NAMES.index("f0_hz")
    other = [c for c in range(rows.shape[1]) if c != f0]
    np.testing.assert_allclose(rows[:, other], want[:, other], atol=1e-3)
    assert (np.abs(rows[:, f0] - want[:, f0]) < 0.5).mean() > 0.98      # pitch is an argmax; rare ties may flip
    got = features.segment_features(fr, ref["segments"])
    np.testing.assert_allclose(got, np.array(ref[mode]["features"]), atol=1e-3)


def test_tagger():
    ref = load("tagger_reference.json")
    n = ref["n"]
    probs = tagger.Tagger().probs(f32(ref["emb"], (n, 1024)), f32(ref["feats"], (n, 28)))
    np.testing.assert_allclose(probs, f32(ref["probs"], (n, 5)), atol=1e-5)


def test_decode():
    for case in load("decode_reference.json")["cases"]:
        p = np.array(case["probs"], np.float32)
        spans = tagger.find_spans(p, case["segments"])
        assert spans == case["spans"]
        assert tagger.find_candidates(p, spans, case["segments"]) == case["candidates"]


@pytest.mark.skipif(not os.environ.get("PODCAST_AD_TAGGER_GGUF"), reason="set PODCAST_AD_TAGGER_GGUF to test the embedder")
def test_embeddings():
    from podcast_ad_tagger.embed import Embedder
    ref = load("embedding_reference.json")
    got = Embedder(os.environ["PODCAST_AD_TAGGER_GGUF"]).embed(ref["texts"])
    want = f32(ref["embeddings"], (len(ref["texts"]), ref["dim"]))
    cos = (got * want).sum(1) / np.linalg.norm(got, axis=1) / np.linalg.norm(want, axis=1)
    assert cos.min() > 0.999
