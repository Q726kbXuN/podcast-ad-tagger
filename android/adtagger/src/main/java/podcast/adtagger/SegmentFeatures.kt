package podcast.adtagger

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/** numpy-compatible statistics: median averages the two middle values, percentile is numpy's linear method. */
object Stats {
    fun median(v: DoubleArray): Double {
        v.sort()
        val n = v.size
        return if (n % 2 == 1) v[n / 2] else (v[n / 2 - 1] + v[n / 2]) / 2.0
    }

    /** [sorted] must be ascending. */
    fun percentile(sorted: DoubleArray, q: Double): Double {
        val n = sorted.size
        val vi = q / 100.0 * (n - 1)
        val lo = floor(vi).toInt()
        if (lo + 1 >= n) return sorted[n - 1]
        val t = vi - lo
        val a = sorted[lo]; val b = sorted[lo + 1]
        val d = b - a
        return if (t >= 0.5) b - d * (1 - t) else a + d * t
    }

    fun mean(v: DoubleArray) = v.sum() / v.size

    fun std(v: DoubleArray): Double {
        val m = mean(v)
        return sqrt(v.sumOf { (it - m) * (it - m) } / v.size)
    }
}

/** The 28 per-segment features the tagger reads, in model order (docs/FEATURES.md). */
object SegmentFeatures {
    val NAMES = listOf(
        "lk_mean", "lk_std", "rms10max", "rms10min", "flat_mean", "side_mean", "zcr_mean",
        "cpp_mean", "f0_std", "centroid_mean", "gap_before_s", "gap_after_s", "lk_delta_prev",
        "stage0_overlap", "rel_pos", "seg_dur", "n_words", "spk_change",
        "lk_vs_ep", "lk_win_delta", "lra_vs_ep", "side_win_delta", "flux_mean",
        "voiced_frac", "band_dist", "bass_delta", "edge_before", "edge_after",
    )
    const val COUNT = 28
    private const val WIN = 50
    private const val VALID = -190.0
    private const val BASS_BANDS = 4

    private fun valid(a: FloatArray, from: Int, to: Int): DoubleArray {
        val out = ArrayList<Double>(max(0, to - from))
        for (i in from until to) if (a[i] > VALID) out.add(a[i].toDouble())
        return out.toDoubleArray()
    }

    private class Baselines(f: Frames) {
        val nb = FrameExtractor.N_BANDS
        val lkMed: Double
        val lra: Double
        val act = BooleanArray(f.n) { f.rms[it] > -80.0 }
        val sideC = DoubleArray(f.n) { max(f.side[it].toDouble(), -120.0) }
        val sideMed: Double?
        val prof = DoubleArray(f.n * nb)
        val profMed = DoubleArray(nb)
        val bass = DoubleArray(f.n)
        val bassMed: Double

        init {
            val lk = valid(f.lk, 0, f.n)
            lkMed = if (lk.isNotEmpty()) Stats.median(lk.copyOf()) else -70.0
            lra = if (lk.size > 10) { lk.sort(); Stats.percentile(lk, 95.0) - Stats.percentile(lk, 10.0) } else 0.0
            val nAct = act.count { it }
            sideMed = if (f.stereo && nAct > 10) Stats.median(DoubleArray(nAct).also { o ->
                var j = 0; for (i in 0 until f.n) if (act[i]) o[j++] = sideC[i]
            }) else null
            for (i in 0 until f.n) {
                var s = 0.0
                for (b in 0 until nb) s += max(f.bands[i * nb + b].toDouble(), -120.0)
                val m = s / nb
                var lo = 0.0; var all = 0.0
                for (b in 0 until nb) {
                    val v = f.bands[i * nb + b].toDouble()
                    prof[i * nb + b] = max(v, -120.0) - m
                    val lin = 10.0.pow(min(max(v, -120.0), 0.0) / 10.0)
                    all += lin; if (b < BASS_BANDS) lo += lin
                }
                bass[i] = lo / (all + 1e-12)
            }
            if (nAct > 10) for (b in 0 until nb) profMed[b] = Stats.median(DoubleArray(nAct).also { o ->
                var j = 0; for (i in 0 until f.n) if (act[i]) o[j++] = prof[i * nb + b]
            })
            bassMed = if (nAct > 10) Stats.median(DoubleArray(nAct).also { o ->
                var j = 0; for (i in 0 until f.n) if (act[i]) o[j++] = bass[i]
            }) else 0.0
        }
    }

    private fun edge(lk: FloatArray, i: Int): Double {
        val n = lk.size
        val pre = valid(lk, min(max(0, i - 10), n), min(i, n))
        val post = valid(lk, min(i, n), min(i + 10, n))
        return if (pre.size >= 3 && post.size >= 3) abs(Stats.mean(post) - Stats.mean(pre)) else 0.0
    }

    /** -> row-major [segments.size] x 28.  [frames] may be null for text-only (all audio columns 0). */
    fun compute(frames: Frames?, segments: List<Segment>): FloatArray {
        val n = segments.size
        val out = FloatArray(n * COUNT)
        if (n == 0) return out
        val have = frames != null && frames.n > 0
        val b = if (have) Baselines(frames!!) else null
        val nfr = frames?.n ?: 0
        val nb = FrameExtractor.N_BANDS
        val dur = segments.last().end
        var prevLk: Double? = null
        val row = DoubleArray(COUNT)
        for ((i, s) in segments.withIndex()) {
            row.fill(0.0)
            val a = s.start; val e = s.end
            if (have) {
                val f = frames!!; val bl = b!!
                val f0i = max(0, (a * 10).toInt())
                val f1i = max((a * 10).toInt() + 1, (e * 10).toInt())
                val stop = min(f1i, nfr)
                val hasRange = stop > f0i
                val lk = if (hasRange) valid(f.lk, f0i, stop) else DoubleArray(0)
                val lkMean = if (lk.isNotEmpty()) Stats.mean(lk) else -100.0
                row[0] = lkMean
                row[1] = if (lk.isNotEmpty()) Stats.std(lk) else 0.0
                row[2] = if (hasRange) (f0i until stop).maxOf { f.rms10max[it].toDouble() } else -100.0
                row[3] = if (hasRange) (f0i until stop).minOf { f.rms10min[it].toDouble() } else -100.0
                for ((k, col) in listOf(4 to f.flatness, 5 to f.side, 6 to f.zcr, 7 to f.cpp, 9 to f.centroid)) {
                    val v = if (hasRange) valid(col, f0i, stop) else DoubleArray(0)
                    row[k] = if (v.isNotEmpty()) Stats.mean(v) else 0.0
                }
                val f0v = ArrayList<Double>()
                if (hasRange) for (j in f0i until stop) if (f.cpp[j] > 1.0) f0v.add(f.f0[j].toDouble())
                row[8] = if (f0v.size > 3) Stats.std(f0v.toDoubleArray()) else 0.0
                var gb = 0
                for (j in min(max(0, f0i - 30), nfr) until min(f0i, nfr)) if (f.rms[j] < -60) gb++
                row[10] = gb / 10.0
                var ga = 0
                for (j in stop until min(stop + 30, nfr)) if (f.rms[j] < -60) ga++
                row[11] = ga / 10.0
                row[12] = if (prevLk != null) lkMean - prevLk else 0.0
                prevLk = lkMean
                if (lk.isNotEmpty()) row[18] = lkMean - bl.lkMed
                val c0 = (f0i + min(f1i, nfr)) / 2
                val w0 = max(0, c0 - WIN); val w1 = min(nfr, c0 + WIN)
                val wlk = if (w1 > w0) valid(f.lk, w0, w1) else DoubleArray(0)
                if (wlk.size > 10) {
                    row[19] = Stats.median(wlk.copyOf()) - bl.lkMed
                    wlk.sort()
                    row[20] = Stats.percentile(wlk, 95.0) - Stats.percentile(wlk, 10.0) - bl.lra
                }
                if (bl.sideMed != null && w1 > w0) row[21] = Stats.median(DoubleArray(w1 - w0) { bl.sideC[w0 + it] }) - bl.sideMed
                val fx = if (hasRange) valid(f.flux, f0i, stop) else DoubleArray(0)
                row[22] = if (fx.isNotEmpty()) Stats.mean(fx) else -90.0
                if (hasRange) row[23] = (f0i until stop).count { f.cpp[it] > 1.0 }.toDouble() / (stop - f0i)
                if (hasRange) {
                    val idx = (f0i until stop).filter { bl.act[it] }
                    if (idx.size >= 3) {
                        var dist = 0.0
                        for (band in 0 until nb) {
                            val m = Stats.median(DoubleArray(idx.size) { bl.prof[idx[it] * nb + band] })
                            dist += abs(m - bl.profMed[band])
                        }
                        row[24] = dist / nb
                        row[25] = Stats.median(DoubleArray(idx.size) { bl.bass[idx[it]] }) - bl.bassMed
                    }
                }
                row[26] = edge(f.lk, f0i)
                row[27] = edge(f.lk, min(stop, nfr))
            }
            // [13] stage0_overlap (repeated-audio history) stays 0, as in production
            row[14] = a / max(dur, 1.0)
            row[15] = e - a
            row[16] = s.nWords.toDouble()
            val prev = if (i > 0) segments[i - 1].speaker else s.speaker
            row[17] = if (s.speaker != null && prev != null && s.speaker != prev) 1.0 else 0.0
            for (k in 0 until COUNT) out[i * COUNT + k] = row[k].toFloat()
        }
        return out
    }
}
