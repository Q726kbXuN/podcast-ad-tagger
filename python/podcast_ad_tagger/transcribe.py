"""Speech to text with faster-whisper, cut into sentence-sized segments.

The tagger was trained on sentence-level segments (a few seconds each, about ten words), so
whatever does the transcription, its segments get split at sentence ends before tagging.
"""
import re

ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "vs.", "etc.", "inc.", "co.", "no.",
                 "u.s.", "u.k.", "e.g.", "i.e.", "a.m.", "p.m.", "mt.", "ft.", "approx."}
SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")
MAX_SEGMENT_S = 15.0


def is_sentence_end(word):
    w = word.strip().lower()
    return bool(SENTENCE_END.search(w)) and w not in ABBREVIATIONS


def split_words(words, max_len=MAX_SEGMENT_S):
    """words: [(start, end, text)] for one recognizer segment -> sentence segments.  A run
    without punctuation is cut at its widest pause once it passes max_len seconds."""
    out, cur = [], []

    def flush(upto=None):
        nonlocal cur
        take, cur = (cur, []) if upto is None else (cur[:upto], cur[upto:])
        if take:
            out.append({"start": take[0][0], "end": take[-1][1],
                        "text": "".join(w[2] for w in take).strip(), "n_words": len(take)})

    for w in words:
        cur.append(w)
        if is_sentence_end(w[2]):
            flush()
        elif cur[-1][1] - cur[0][0] > max_len and len(cur) > 2:
            gaps = [cur[i + 1][0] - cur[i][1] for i in range(len(cur) - 1)]
            flush(gaps.index(max(gaps)) + 1)
    flush()
    return out


def load_16k(path):
    """Whisper wants 16 kHz mono float32; ffmpeg does the decoding."""
    import subprocess
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:a:0", "-ac", "1",
                          "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def transcribe(path, model="small.en", device="auto", compute_type="int8", language="en"):
    """-> [{start, end, text, n_words}] in time order."""
    from faster_whisper import WhisperModel
    m = WhisperModel(model, device=device, compute_type=compute_type)
    segments, _ = m.transcribe(load_16k(path), language=language, word_timestamps=True, beam_size=5,
                               condition_on_previous_text=False)
    out = []
    for s in segments:
        words = [(w.start, w.end, w.word) for w in (s.words or []) if w.end >= w.start]
        if words:
            out.extend(split_words(words))
        elif s.text.strip():
            out.append({"start": s.start, "end": s.end, "text": s.text.strip(),
                        "n_words": len(s.text.split())})
    for s in out:
        if s["end"] <= s["start"]:
            s["end"] = s["start"] + 0.01
    return out
