"""Audio analysis: decode to 24 kHz and compute a row of measurements every 100 ms.

docs/FEATURES.md is the spec; this file is the reference implementation the Swift and
Kotlin ports are tested against (testdata/).
"""
import json
import math
import subprocess

import numpy as np
import scipy.fft as sfft
import scipy.signal as sig

SR = 24000            # analysis rate
HOP = 2400            # 100 ms frame, one output row each
SUB = 240             # 10 ms sub-frames, for the loudest/quietest 10 ms in a frame
NFFT = 2048           # STFT window
SHOP = 1200           # STFT hop, so two spectra per frame
F0_MIN, F0_MAX = 60.0, 400.0
DB_FLOOR = -200.0
PAD_LEFT = NFFT // 2
PAD_RIGHT = NFFT

BAND_EDGES = [0, 60, 120, 250, 400, 630, 800, 1000, 1250, 1600,
              2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12000]
N_BANDS = len(BAND_EDGES) - 1
FRAME_NAMES = ["lk_db", "rms_db", "rms10max_db", "rms10min_db", "zcr", "centroid_hz",
               "flatness_db", "flux_db", "f0_hz", "cpp_db", "side_db"]


class Frames:
    """Per-100 ms measurements for a whole episode."""

    def __init__(self, scalars, bands, stereo):
        self.scalars = scalars          # dict name -> float32 array [n_frames]
        self.bands = bands              # float32 [n_frames, 18], dB
        self.stereo = stereo            # the source had two or more channels

    def __len__(self):
        return len(self.bands)


def db10(power):
    power = np.asarray(power, dtype=np.float64)
    out = np.full(power.shape, DB_FLOOR, dtype=np.float64)
    good = power > 0
    np.log10(power, out=out, where=good)
    out[good] *= 10.0
    return np.maximum(out, DB_FLOOR)


def db20(amplitude):
    return db10(np.asarray(amplitude, dtype=np.float64) ** 2)


def kweighting_sos(sr):
    """ITU-R BS.1770 K-weighting (high shelf + high pass) as two biquads at `sr`."""
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0,
             (vh - vb * k / q + k * k) / a0, 1.0,
             2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    f0b, qb = 38.13547087602444, 0.5003270373238773
    kb = math.tan(math.pi * f0b / sr)
    a0b = 1.0 + kb / qb + kb * kb
    hpf = [1.0, -2.0, 1.0, 1.0,
           2.0 * (kb * kb - 1.0) / a0b, (1.0 - kb / qb + kb * kb) / a0b]
    return np.array([shelf, hpf], dtype=np.float32)


class FrameExtractor:
    """Streaming: feed() 24 kHz mid (and side) samples in any chunk sizes, then finish()."""

    def __init__(self):
        self.sos = kweighting_sos(SR)
        self.zi = np.zeros((2, 2), np.float32)
        self.win = np.hanning(NFFT).astype(np.float32)
        self.freqs = np.fft.rfftfreq(NFFT, 1.0 / SR).astype(np.float32)
        self.n_bins = len(self.freqs)
        self.band_mat = np.zeros((self.n_bins, N_BANDS), dtype=np.float32)
        for i in range(N_BANDS):
            self.band_mat[(self.freqs >= BAND_EDGES[i]) & (self.freqs < BAND_EDGES[i + 1]), i] = 1.0
        # scale so the bins of a frame sum to its mean square
        scale = 1.0 / (float(NFFT) ** 2 * float((self.win.astype(np.float64) ** 2).mean()))
        self.bin_norm = np.full(self.n_bins, 2.0 * scale, dtype=np.float32)
        self.bin_norm[0] = scale
        self.bin_norm[-1] = scale
        self.q_lo = max(1, int(SR / F0_MAX))
        self.q_hi = min(NFFT // 2, int(SR / F0_MIN) + 1)
        self.buf = np.zeros(PAD_LEFT, dtype=np.float32)
        self.buf_side = np.zeros(PAD_LEFT, dtype=np.float32)
        self.have_side = False
        self.prev_mag = None
        self.rows, self.band_rows = [], []

    def feed(self, mid, side=None):
        self.buf = np.concatenate([self.buf, np.asarray(mid, np.float32)])
        if side is not None:
            self.have_side = True
            self.buf_side = np.concatenate([self.buf_side, np.asarray(side, np.float32)])
        self._drain()

    def finish(self):
        pad = np.zeros(PAD_RIGHT + HOP, dtype=np.float32)
        self.buf = np.concatenate([self.buf, pad])
        if self.have_side:
            self.buf_side = np.concatenate([self.buf_side, pad])
        self._drain()
        if not self.rows:
            return Frames({n: np.zeros(0, np.float32) for n in FRAME_NAMES},
                          np.zeros((0, N_BANDS), np.float32), self.have_side)
        rows = np.vstack(self.rows)
        return Frames({n: rows[:, i] for i, n in enumerate(FRAME_NAMES)},
                      np.vstack(self.band_rows), self.have_side)

    def _drain(self):
        nf = (len(self.buf) - PAD_LEFT - PAD_RIGHT) // HOP
        if nf <= 0:
            return
        self._emit(nf)
        self.buf = self.buf[nf * HOP:]
        if self.have_side:
            self.buf_side = self.buf_side[nf * HOP:]

    def _emit(self, nf):
        per = HOP // SHOP
        core = self.buf[PAD_LEFT:PAD_LEFT + nf * HOP]
        frames = core.reshape(nf, HOP)
        weighted, self.zi = sig.sosfilt(self.sos, core, zi=self.zi)
        lk = (weighted.reshape(nf, HOP) ** 2).mean(axis=1)
        rms = (frames ** 2).mean(axis=1)
        sub_ms = (frames.reshape(nf, HOP // SUB, SUB) ** 2).mean(axis=2)
        signs = np.signbit(frames)
        zcr = (signs[:, 1:] != signs[:, :-1]).sum(axis=1) / np.float32(HOP - 1)
        if self.have_side:
            side = self.buf_side[PAD_LEFT:PAD_LEFT + nf * HOP].reshape(nf, HOP)
            side_db = db10((side ** 2).mean(axis=1))
        else:
            side_db = np.full(nf, DB_FLOOR)

        # two Hann-windowed spectra per frame, centred 600 and 1800 samples into it
        starts = np.arange(nf * per, dtype=np.int64) * SHOP + (SHOP // 2)
        block = self.buf[starts[:, None] + np.arange(NFFT, dtype=np.int64)[None, :]] * self.win
        spec = sfft.rfft(block, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2) * self.bin_norm + np.float32(1e-20)
        total = power.sum(axis=1)
        centroid = (power * self.freqs).sum(axis=1) / total
        log_power = np.log(power)
        flatness = np.exp(log_power.mean(axis=1)) / (total / self.n_bins)
        mag = np.sqrt(power)
        prev = mag[:1] if self.prev_mag is None else self.prev_mag[None, :]
        flux = np.maximum(mag - np.vstack([prev, mag[:-1]]), 0.0).sum(axis=1)
        self.prev_mag = mag[-1].copy()
        # cepstral peak in the 60-400 Hz pitch range: voicing strength and pitch
        cep = sfft.irfft(log_power, axis=1)
        window = cep[:, self.q_lo:self.q_hi]
        q_peak = window.argmax(axis=1)
        cpp = (window[np.arange(len(window)), q_peak] - np.median(window, axis=1)) * (10.0 / np.log(10.0))
        f0 = np.float32(SR) / (q_peak + self.q_lo).astype(np.float32)

        def fold(v):
            return v.reshape(nf, per).mean(axis=1)

        cpp_2d = cpp.reshape(nf, per)
        best = cpp_2d.argmax(axis=1)
        r = np.arange(nf)
        cols = {
            "lk_db": db10(lk), "rms_db": db10(rms),
            "rms10max_db": db10(sub_ms.max(axis=1)), "rms10min_db": db10(sub_ms.min(axis=1)),
            "zcr": zcr, "centroid_hz": fold(centroid), "flatness_db": db10(fold(flatness)),
            "flux_db": db20(fold(flux)), "f0_hz": f0.reshape(nf, per)[r, best],
            "cpp_db": cpp_2d[r, best], "side_db": side_db,
        }
        self.rows.append(np.stack([np.asarray(cols[n], np.float32) for n in FRAME_NAMES], axis=1))
        self.band_rows.append(db10((power @ self.band_mat).reshape(nf, per, N_BANDS).mean(axis=1)).astype(np.float32))


def channel_count(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=channels", "-of", "json", path], capture_output=True, text=True, check=True)
    streams = json.loads(out.stdout).get("streams") or [{}]
    return int(streams[0].get("channels") or 1)


def analyze_file(path):
    """Decode any audio file with ffmpeg and return its Frames.  Stereo is split into
    mid = (L+R)/2 and side = (L-R)/2; mono has no side."""
    stereo = channel_count(path) >= 2
    ch = 2 if stereo else 1
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:a:0",
                             "-ac", str(ch), "-ar", str(SR), "-f", "f32le", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 22)
    ex = FrameExtractor()
    step = HOP * 600 * 4 * ch
    left = b""
    try:
        while True:
            raw = proc.stdout.read(step)
            if not raw:
                break
            raw = left + raw
            use = (len(raw) // (4 * ch)) * 4 * ch
            raw, left = raw[:use], raw[use:]
            if not raw:
                continue
            data = np.frombuffer(raw, dtype=np.float32)
            if stereo:
                l, rr = data[0::2], data[1::2]
                ex.feed((l + rr) * np.float32(0.5), (l - rr) * np.float32(0.5))
            else:
                ex.feed(data)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.stderr.close()
        proc.wait()
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed on {path}: {err[:400]}")
    return ex.finish()
