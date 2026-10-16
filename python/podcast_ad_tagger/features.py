"""The 28 per-segment features the tagger reads, from Frames and the transcript segments.

Order matters: it is the column order of the model's `feats` input (docs/FEATURES.md).
"""
import numpy as np

FEATURE_NAMES = [
    "lk_mean", "lk_std", "rms10max", "rms10min", "flat_mean", "side_mean", "zcr_mean",
    "cpp_mean", "f0_std", "centroid_mean", "gap_before_s", "gap_after_s", "lk_delta_prev",
    "stage0_overlap", "rel_pos", "seg_dur", "n_words", "spk_change",
    "lk_vs_ep", "lk_win_delta", "lra_vs_ep", "side_win_delta", "flux_mean",
    "voiced_frac", "band_dist", "bass_delta", "edge_before", "edge_after",
]
WIN = 50                    # +/- 5 s window (in frames) for the local-baseline features
VALID = -190.0              # frame values at or below this are "no signal"
BASS_BANDS = 4              # bands below 400 Hz


def _valid(a):
    return a[a > VALID]


def _baselines(f):
    """Episode-wide reference values the relative features are measured against."""
    b = {}
    lk = _valid(f.scalars["lk_db"])
    b["lk_med"] = float(np.median(lk)) if len(lk) else -70.0
    b["lra"] = float(np.percentile(lk, 95) - np.percentile(lk, 10)) if len(lk) > 10 else 0.0
    act = f.scalars["rms_db"] > -80.0
    b["act"] = act
    side_c = np.maximum(f.scalars["side_db"], -120.0)
    b["side_c"] = side_c
    b["side_med"] = float(np.median(side_c[act])) if f.stereo and act.sum() > 10 else None
    bc = np.clip(f.bands, -120.0, None)
    prof = bc - bc.mean(axis=1, keepdims=True)
    b["prof"] = prof
    b["prof_med"] = np.median(prof[act], axis=0) if act.sum() > 10 else np.zeros(f.bands.shape[1], np.float32)
    lin = np.power(10.0, np.clip(f.bands, -120, 0) / 10.0)
    bass = lin[:, :BASS_BANDS].sum(axis=1) / (lin.sum(axis=1) + 1e-12)
    b["bass"] = bass
    b["bass_med"] = float(np.median(bass[act])) if act.sum() > 10 else 0.0
    return b


def _edge(lk, i):
    """Loudness step across frame i: |mean of the next 1 s - mean of the previous 1 s|."""
    pre, post = _valid(lk[max(0, i - 10):i]), _valid(lk[i:i + 10])
    return abs(float(post.mean()) - float(pre.mean())) if len(pre) >= 3 and len(post) >= 3 else 0.0


def segment_features(frames, segments):
    """frames: audio.Frames (or None for text-only).  segments: dicts with start, end (seconds),
    and optionally n_words and speaker.  Returns float32 [len(segments), 28]."""
    n = len(segments)
    F = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    if n == 0:
        return F
    have = frames is not None and len(frames) > 0
    if have:
        col = frames.scalars
        b = _baselines(frames)
        nfr = len(frames)
    dur = float(segments[-1]["end"])
    prev_lk = None
    for i, s in enumerate(segments):
        a, e = float(s["start"]), float(s["end"])
        row = dict.fromkeys(FEATURE_NAMES, 0.0)
        if have:
            f0i, f1i = max(0, int(a * 10)), max(int(a * 10) + 1, int(e * 10))
            sl = slice(f0i, min(f1i, nfr))
            lk = _valid(col["lk_db"][sl])
            row["lk_mean"] = float(lk.mean()) if len(lk) else -100.0
            row["lk_std"] = float(lk.std()) if len(lk) else 0.0
            row["rms10max"] = float(col["rms10max_db"][sl].max()) if sl.stop > sl.start else -100.0
            row["rms10min"] = float(col["rms10min_db"][sl].min()) if sl.stop > sl.start else -100.0
            for k, c in (("flat_mean", "flatness_db"), ("side_mean", "side_db"), ("zcr_mean", "zcr"),
                         ("cpp_mean", "cpp_db"), ("centroid_mean", "centroid_hz")):
                v = _valid(col[c][sl])
                row[k] = float(v.mean()) if len(v) else 0.0
            f0v = col["f0_hz"][sl][col["cpp_db"][sl] > 1.0]
            row["f0_std"] = float(f0v.std()) if len(f0v) > 3 else 0.0
            row["gap_before_s"] = float((col["rms_db"][max(0, f0i - 30):f0i] < -60).sum()) / 10.0
            row["gap_after_s"] = float((col["rms_db"][sl.stop:sl.stop + 30] < -60).sum()) / 10.0
            row["lk_delta_prev"] = row["lk_mean"] - prev_lk if prev_lk is not None else 0.0
            prev_lk = row["lk_mean"]
            if len(lk):
                row["lk_vs_ep"] = row["lk_mean"] - b["lk_med"]
            c0 = (f0i + min(f1i, nfr)) // 2
            w = slice(max(0, c0 - WIN), min(nfr, c0 + WIN))
            wlk = _valid(col["lk_db"][w])
            if len(wlk) > 10:
                row["lk_win_delta"] = float(np.median(wlk)) - b["lk_med"]
                row["lra_vs_ep"] = float(np.percentile(wlk, 95) - np.percentile(wlk, 10)) - b["lra"]
            if b["side_med"] is not None and w.stop > w.start:
                row["side_win_delta"] = float(np.median(b["side_c"][w])) - b["side_med"]
            fx = _valid(col["flux_db"][sl])
            row["flux_mean"] = float(fx.mean()) if len(fx) else -90.0
            if sl.stop > sl.start:
                row["voiced_frac"] = float((col["cpp_db"][sl] > 1.0).mean())
            act = b["act"][sl]
            if act.sum() >= 3:
                row["band_dist"] = float(np.abs(np.median(b["prof"][sl][act], axis=0) - b["prof_med"]).mean())
                row["bass_delta"] = float(np.median(b["bass"][sl][act])) - b["bass_med"]
            row["edge_before"] = _edge(col["lk_db"], f0i)
            row["edge_after"] = _edge(col["lk_db"], min(sl.stop, nfr))
        # stage0_overlap (repeated-audio history) stays 0, as it does in production
        row["rel_pos"] = a / max(dur, 1)
        row["seg_dur"] = e - a
        row["n_words"] = float(s.get("n_words") if s.get("n_words") is not None else len(str(s.get("text", "")).split()))
        spk, prev_spk = s.get("speaker"), segments[i - 1].get("speaker") if i else s.get("speaker")
        row["spk_change"] = 1.0 if (spk is not None and prev_spk is not None and spk != prev_spk) else 0.0
        F[i] = [row[k] for k in FEATURE_NAMES]
    return F
