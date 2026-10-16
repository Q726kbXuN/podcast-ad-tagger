// node --test web/test   (no dependencies: checks the JS audio, decoding and sentence code
// against ../../testdata, the same reference values the other ports use)
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { analyzeFrames, segmentFeatures, FRAME_NAMES } from '../lib/audio.js';
import { findSpans, findCandidates } from '../lib/spans.js';
import { splitWords } from '../lib/text.js';
import { chunkAtPauses } from '../lib/whisper.js';

const DATA = new URL('../../testdata/', import.meta.url);
const json = (name) => JSON.parse(fs.readFileSync(new URL(name, DATA)));

function synth() {
  const b = fs.readFileSync(new URL('synth.wav', DATA));
  const pcm = new Int16Array(b.buffer, b.byteOffset + 44, (b.length - 44) / 2);
  const n = pcm.length / 2, mid = new Float32Array(n), side = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const l = Math.fround(pcm[2 * i] / 32768), r = Math.fround(pcm[2 * i + 1] / 32768);
    mid[i] = (l + r) * 0.5; side[i] = (l - r) * 0.5;
  }
  return { mid, side };
}

for (const mode of ['stereo', 'mono']) {
  test(`audio frames and features (${mode})`, () => {
    const ref = json('audio_reference.json');
    const { mid, side } = synth();
    const fr = analyzeFrames(mid, mode === 'stereo' ? side : null);
    const want = ref[mode].frames;
    assert.equal(fr.n, want.length);
    let worst = 0, f0Off = 0;
    for (let f = 0; f < fr.n; f++) {
      const row = [...FRAME_NAMES.map((k) => fr.cols[k][f]), ...fr.bands.subarray(f * 18, f * 18 + 18)];
      row.forEach((v, c) => {
        const d = Math.abs(v - want[f][c]);
        if (FRAME_NAMES[c] === 'f0_hz') { if (d > 0.5) f0Off++; } else worst = Math.max(worst, d);
      });
    }
    // the reference was computed in float32, so near-silent frames (-130 dB) differ by a few thousandths
    assert.ok(worst < 1e-2, `frame max |diff| ${worst}`);
    assert.ok(f0Off <= fr.n * 0.02, `${f0Off} pitch frames differ`);
    const got = segmentFeatures(fr, ref.segments);
    ref[mode].features.flat().forEach((v, i) => assert.ok(Math.abs(got[i] - v) < 2e-3, `feature ${i}: ${got[i]} vs ${v}`));
  });
}

test('span decoding', () => {
  for (const c of json('decode_reference.json').cases) {
    const probs = Float32Array.from(c.probs.flat());
    const spans = findSpans(probs, c.segments);
    assert.deepEqual(spans, c.spans);
    assert.deepEqual(findCandidates(probs, spans, c.segments), c.candidates);
  }
});

test('sentence splitting', () => {
  const w = (s, e, t) => ({ start: s, end: e, text: t });
  const segs = splitWords([w(0, 0.4, ' Hello'), w(0.4, 0.9, ' there.'), w(1, 1.3, ' Mr.'), w(1.3, 1.8, ' Smith'),
                           w(1.8, 2.2, ' said'), w(2.2, 2.6, ' hi!')]);
  assert.deepEqual(segs.map((s) => s.text), ['Hello there.', 'Mr. Smith said hi!']);
  assert.deepEqual(segs.map((s) => s.n_words), [2, 4]);
});

test('chunking at pauses', () => {
  // 100 s of "speech" at -20 dB with a quiet dip (-70 dB) every 13 s
  const n = 1000, rms = new Float32Array(n).fill(-20);
  for (let f = 130; f < n; f += 130) rms.fill(-70, f - 1, f + 2);
  const chunks = chunkAtPauses(rms, n);
  assert.equal(chunks[0][0], 0);
  assert.equal(chunks.at(-1)[1], n);
  for (let i = 0; i < chunks.length; i++) {
    const [a, b] = chunks[i];
    assert.ok(b - a <= 295, `chunk ${i} is ${(b - a) / 10} s`);
    if (i) assert.equal(a, chunks[i - 1][1]);                       // no gaps, no overlap
    if (b < n) assert.ok(rms[b] === -70, `chunk ${i} ends at a pause`);
  }
});
