# What the model reads

For each transcript segment, in time order, the tagger wants two things:

- `emb`: a 1024-float text embedding of the segment, and
- `feats`: 28 numbers about the segment, mostly about how its audio sounds.

It returns `probs`, one row of 5 probabilities per segment. This page is the spec for producing `emb` and `feats` on a new platform. `python/podcast_ad_tagger/audio.py` and `features.py` are the reference implementation, and `testdata/` has values to check a port against.

## Text embeddings

[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), specifically the Q8_0 GGUF from [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) (sha256 in `model/ad_tagger.json`), run through llama.cpp:

- The input is the segment's text with surrounding whitespace trimmed, cut to the first 2000 characters, or a single space if that leaves nothing. No instruction prefix, no neighboring text.
- Tokenize with special tokens added (llama.cpp's default for this model appends the end-of-text token), use last-token pooling, and L2-normalize the result.

Other runtimes of the same model work too (the original weights in PyTorch, a Core ML or ONNX conversion) as long as they agree with `testdata/embedding_reference.json` to a cosine similarity of about 0.999 or better. Heavier quantization (4-bit) drifts too far.

## Audio: 100 ms frames

Decode the audio and resample it to 24 kHz. Stereo becomes `mid = (L + R) / 2` and `side = (L - R) / 2`; mono is just `mid` with no side. Everything below runs on `mid` unless it says otherwise.

The model learned from audio that ffmpeg decoded and resampled (`ffmpeg -i file -ac 2 -ar 24000 -f f32le -`, with its default resampler), so the closer a port gets to that, the better. Two things turned out to matter more than you'd expect. Platform resamplers like `AVAudioConverter` shift the spectral numbers noticeably, which is why the iOS package carries a port of ffmpeg's resampler. And MP3s carry encoder delay in their gapless info, which ffmpeg trims (including the 576 samples FFmpeg-written files declare) and some decoders don't; a constant offset of a few milliseconds is harmless, but get it consistent with ffmpeg if you can.

The frame grid: frame `f` covers samples `[2400 f, 2400 (f + 1))`. For an input of `n` samples there are `floor(n / 2400) + 1` frames (the last one is padded with silence). In dB, `db10(x) = 10 log10(x)` and `db20(x) = 20 log10(x)`, both floored at -200 (and -200 for zero or negative input).

Per frame:

| name | how |
|---|---|
| `lk_db` | mean square of the K-weighted signal over the frame, `db10`. K-weighting is the ITU-R BS.1770 filter (high shelf then high pass) as two biquads at 24 kHz; coefficients are in `audio.kweighting_sos`. Run it as one continuous filter over the whole file, starting from zero state. |
| `rms_db` | mean square of the frame, `db10` |
| `rms10max_db`, `rms10min_db` | split the frame into ten 240-sample pieces, take each piece's mean square; `db10` of the largest and of the smallest |
| `zcr` | count of adjacent sample pairs whose sign bits differ, divided by 2399 |
| `side_db` | mean square of `side` over the frame, `db10`; exactly -200 for mono |
| `centroid_hz`, `flatness_db`, `flux_db`, `f0_hz`, `cpp_db`, bands | from two spectra per frame, below |

Spectra: two per frame, 2048-sample Hann windows (numpy's `hanning`, symmetric). Window `k` (0 or 1) of frame `f` covers input samples `[2400 f + 1200 k - 424, 2400 f + 1200 k + 1624)`, so they're centered about 600 and 1800 samples into the frame; samples before the start or after the end are zeros. For each window take the real FFT (1025 bins at `k * 24000 / 2048` Hz) and its power `|X|^2 * norm + 1e-20`, where `norm = 2 / (2048^2 * mean(w^2))` (half that for the first and last bin), so the bins sum to the window's mean square. Then, per spectrum:

- centroid: `sum(power * freq) / sum(power)`
- flatness: `exp(mean(ln power)) / (sum(power) / 1025)`
- flux: `sum(max(mag - prev_mag, 0))` with `mag = sqrt(power)` and `prev_mag` the previous spectrum's (the very first spectrum compares with itself, so 0)
- cepstrum: the inverse real FFT (2048 points) of `ln power`. Look at quefrency indices 60 through 400 inclusive (60-400 Hz). The peak index `q` gives pitch `24000 / q`, and `cpp = (peak value - median of those 341 values) * 10 / ln 10`.
- bands: power summed over bins with `lo <= freq < hi` for the 18 bands with edges 0, 60, 120, 250, 400, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12000 Hz

and per frame: `centroid_hz` is the mean of the two centroids, `flatness_db` is `db10` of the mean flatness, `flux_db` is `db20` of the mean flux, `f0_hz` and `cpp_db` come from whichever of the two spectra has the larger `cpp`, and each band is `db10` of the mean of its two values.

Pitch is an argmax, so a port will now and then pick a different peak than the reference on a frame where two are nearly tied. That's harmless. Likewise, on a stereo file that's really mono, `side` is nothing but decoder and resampler noise, so it can differ by several dB between implementations. Neither changed a single span when I compared ports on real episodes.

## The 28 segment features

Segments need `start` and `end` in seconds, plus a word count, and a speaker label if you have one. A segment covers frames `int(10 start)` up to but not including `max(int(10 start) + 1, int(10 end))`, clipped to the frames that exist. "Valid" frame values are those above -190.

Episode-wide baselines first, over all frames:

- `lk_med`: median of valid `lk_db` (-70 if none); `lra`: 95th minus 10th percentile of valid `lk_db` (0 if 10 or fewer)
- active frames: `rms_db > -80`
- `side_med`: median over active frames of `max(side_db, -120)`, only for stereo with more than 10 active frames
- spectral profile per frame: each band clipped to at least -120, minus that frame's mean over the 18 bands; `prof_med` is the per-band median over active frames (zeros if 10 or fewer)
- bass share per frame: power below 400 Hz (the first 4 bands, converted with `10^(dB/10)` after clipping to -120..0) over the total; `bass_med` is its median over active frames (0 if 10 or fewer)

Medians average the two middle values. Percentiles are numpy's default linear interpolation. Standard deviation is the population one.

Then for each segment, in this order (it's the column order of `feats`):

| # | name | value |
|---|---|---|
| 0 | `lk_mean` | mean of valid `lk_db` in the segment (-100 if none) |
| 1 | `lk_std` | std of the same (0 if none) |
| 2 | `rms10max` | max of `rms10max_db` (-100 if no frames) |
| 3 | `rms10min` | min of `rms10min_db` (-100 if no frames) |
| 4 | `flat_mean` | mean of valid `flatness_db` (0 if none) |
| 5 | `side_mean` | mean of valid `side_db` (0 if none, so always 0 for mono) |
| 6 | `zcr_mean` | mean of `zcr` |
| 7 | `cpp_mean` | mean of valid `cpp_db` |
| 8 | `f0_std` | std of `f0_hz` over frames with `cpp_db > 1` (0 if 3 or fewer) |
| 9 | `centroid_mean` | mean of `centroid_hz` |
| 10 | `gap_before_s` | number of frames with `rms_db < -60` among the 30 before the segment, / 10 |
| 11 | `gap_after_s` | same for the 30 frames after it |
| 12 | `lk_delta_prev` | `lk_mean` minus the previous segment's `lk_mean` (0 for the first) |
| 13 | `stage0_overlap` | always 0 |
| 14 | `rel_pos` | `start / max(episode end, 1)`, where episode end is the last segment's `end` |
| 15 | `seg_dur` | `end - start` |
| 16 | `n_words` | words in the segment |
| 17 | `spk_change` | 1 if the speaker differs from the previous segment's, else 0 (always 0 without speaker labels) |
| 18 | `lk_vs_ep` | `lk_mean - lk_med` (0 if the segment has no valid loudness) |
| 19 | `lk_win_delta` | median of valid `lk_db` in a window of 50 frames each side of the segment's center frame, minus `lk_med` (0 if 10 or fewer valid) |
| 20 | `lra_vs_ep` | 95th minus 10th percentile in that window, minus `lra` (same condition) |
| 21 | `side_win_delta` | median of `max(side_db, -120)` in that window minus `side_med` (0 for mono) |
| 22 | `flux_mean` | mean of valid `flux_db` (-90 if none) |
| 23 | `voiced_frac` | share of the segment's frames with `cpp_db > 1` |
| 24 | `band_dist` | over the segment's active frames (needs at least 3): per-band median of the spectral profile, minus `prof_med`, absolute value, averaged over bands |
| 25 | `bass_delta` | median bass share over the segment's active frames minus `bass_med` (same condition) |
| 26 | `edge_before` | loudness step at the segment's first frame: `abs(mean of valid lk_db in the 10 frames from it - mean of valid lk_db in the 10 before it)`, 0 unless both sides have at least 3 |
| 27 | `edge_after` | the same at the frame just after the segment |

The window in 19-21 is frames `[center - 50, center + 50)` clipped to the file, where `[first, end)` is the segment's frame range from above (before clipping) and `center = (first + min(end, n_frames)) // 2`. Everything that is "0 if none" also applies to segments that fall past the end of the audio.

If there's no audio at all, all audio features are zero, which the model has seen, but expect noticeably worse results. Text alone is not the intended setup.
