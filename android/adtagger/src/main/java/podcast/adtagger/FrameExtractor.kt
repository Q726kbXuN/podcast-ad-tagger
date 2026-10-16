package podcast.adtagger

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Per-100 ms audio measurements for a whole episode (see docs/FEATURES.md).  One value per
 * frame in each column; [bands] is row-major, [n] x [FrameExtractor.N_BANDS], all in dB.
 */
class Frames(
    val n: Int, val stereo: Boolean,
    val lk: FloatArray, val rms: FloatArray, val rms10max: FloatArray, val rms10min: FloatArray,
    val zcr: FloatArray, val centroid: FloatArray, val flatness: FloatArray, val flux: FloatArray,
    val f0: FloatArray, val cpp: FloatArray, val side: FloatArray, val bands: FloatArray,
)

/**
 * Streaming port of the reference extractor (python/podcast_ad_tagger/audio.py).  Feed 24 kHz
 * mid = (L+R)/2 samples, plus side = (L-R)/2 for stereo sources, in any chunk sizes, then call
 * [finish].  Not thread safe; one instance per episode.
 */
class FrameExtractor {
    companion object {
        const val SR = 24000
        const val HOP = 2400
        const val SUB = 240
        const val NFFT = 2048
        const val SHOP = 1200
        const val PAD_LEFT = NFFT / 2
        const val PAD_RIGHT = NFFT
        const val DB_FLOOR = -200.0
        val BAND_EDGES = intArrayOf(0, 60, 120, 250, 400, 630, 800, 1000, 1250, 1600,
            2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12000)
        val N_BANDS = BAND_EDGES.size - 1
        private const val N_BINS = NFFT / 2 + 1
        private val Q_LO = max(1, (SR / 400.0).toInt())
        private val Q_HI = min(NFFT / 2, (SR / 60.0).toInt() + 1)

        fun db10(p: Double): Double = if (p > 0) max(10.0 * log10(p), DB_FLOOR) else DB_FLOOR

        /** ITU-R BS.1770 K-weighting as two biquads [b0, b1, b2, a1, a2], rounded through float like the reference. */
        internal fun kWeighting(sr: Double): Array<DoubleArray> {
            val f0 = 1681.974450955533; val gainDb = 3.999843853973347; val q = 0.7071752369554196
            val k = tan(PI * f0 / sr)
            val vh = Math.pow(10.0, gainDb / 20.0)
            val vb = Math.pow(vh, 0.4996667741545416)
            val a0 = 1.0 + k / q + k * k
            val shelf = doubleArrayOf((vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0,
                (vh - vb * k / q + k * k) / a0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0)
            val f0b = 38.13547087602444; val qb = 0.5003270373238773
            val kb = tan(PI * f0b / sr)
            val a0b = 1.0 + kb / qb + kb * kb
            val hpf = doubleArrayOf(1.0, -2.0, 1.0, 2.0 * (kb * kb - 1.0) / a0b, (1.0 - kb / qb + kb * kb) / a0b)
            return arrayOf(shelf, hpf).map { c -> DoubleArray(5) { c[it].toFloat().toDouble() } }.toTypedArray()
        }
    }

    private val sos = kWeighting(SR.toDouble())
    private val z = DoubleArray(4)
    private val win = DoubleArray(NFFT) { (0.5 - 0.5 * cos(2.0 * PI * it / (NFFT - 1))).toFloat().toDouble() }
    private val freqs = DoubleArray(N_BINS) { (it * SR.toDouble() / NFFT).toFloat().toDouble() }
    private val binNorm: DoubleArray
    private val bandOf = IntArray(N_BINS) { -1 }
    private val fft = Fft(NFFT)
    private val re = DoubleArray(NFFT)
    private val im = DoubleArray(NFFT)
    private val power = DoubleArray(N_BINS)
    private val mag = DoubleArray(N_BINS)
    private var prevMag: DoubleArray? = null
    private val cepWindow = DoubleArray(Q_HI - Q_LO)

    private var buf = FloatArray(1 shl 16)
    private var bufLen = PAD_LEFT
    private var side = FloatArray(1 shl 16)
    private var haveSide = false

    private val cols = Array(11) { FloatList() }
    private val bandRows = FloatList()

    init {
        val scale = 1.0 / (NFFT.toDouble() * NFFT * win.sumOf { it * it } / NFFT)
        binNorm = DoubleArray(N_BINS) { 2.0 * scale }
        binNorm[0] = scale; binNorm[N_BINS - 1] = scale
        for (k in 0 until N_BINS) for (b in 0 until N_BANDS)
            if (freqs[k] >= BAND_EDGES[b] && freqs[k] < BAND_EDGES[b + 1]) bandOf[k] = b
    }

    /** Append [count] samples starting at [offset]; [sideSamples] must be given for every call or none. */
    fun feed(mid: FloatArray, sideSamples: FloatArray? = null, offset: Int = 0, count: Int = mid.size - offset) {
        ensure(bufLen + count)
        System.arraycopy(mid, offset, buf, bufLen, count)
        if (sideSamples != null) {
            haveSide = true
            System.arraycopy(sideSamples, offset, side, bufLen, count)
        }
        bufLen += count
        drain()
    }

    fun finish(): Frames {
        val pad = PAD_RIGHT + HOP
        ensure(bufLen + pad)
        java.util.Arrays.fill(buf, bufLen, bufLen + pad, 0f)
        java.util.Arrays.fill(side, bufLen, bufLen + pad, 0f)
        bufLen += pad
        drain()
        val c = cols.map { it.toArray() }
        return Frames(c[0].size, haveSide, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10],
            bandRows.toArray())
    }

    private fun ensure(size: Int) {
        if (size > buf.size) {
            val cap = max(size, buf.size * 2)
            buf = buf.copyOf(cap); side = side.copyOf(cap)
        }
    }

    private fun drain() {
        val nf = (bufLen - PAD_LEFT - PAD_RIGHT) / HOP
        if (nf <= 0) return
        for (f in 0 until nf) emit(f)
        val drop = nf * HOP
        System.arraycopy(buf, drop, buf, 0, bufLen - drop)
        System.arraycopy(side, drop, side, 0, bufLen - drop)
        bufLen -= drop
    }

    private fun filter(x: Double): Double {
        var v = x
        for (s in 0 until 2) {
            val c = sos[s]
            val y = c[0] * v + z[2 * s]
            z[2 * s] = c[1] * v - c[3] * y + z[2 * s + 1]
            z[2 * s + 1] = c[2] * v - c[4] * y
            v = y
        }
        return v
    }

    private fun emit(f: Int) {
        val base = PAD_LEFT + f * HOP
        var lk = 0.0; var ms = 0.0; var sideMs = 0.0
        var subMax = Double.NEGATIVE_INFINITY; var subMin = Double.POSITIVE_INFINITY
        var crossings = 0
        for (s in 0 until HOP / SUB) {
            var acc = 0.0
            for (t in s * SUB until (s + 1) * SUB) {
                val x = buf[base + t].toDouble()
                acc += x * x
                val y = filter(x)
                lk += y * y
                if (t > 0 && (java.lang.Float.floatToRawIntBits(buf[base + t]) < 0) !=
                    (java.lang.Float.floatToRawIntBits(buf[base + t - 1]) < 0)) crossings++
                if (haveSide) { val sd = side[base + t].toDouble(); sideMs += sd * sd }
            }
            ms += acc
            subMax = max(subMax, acc / SUB); subMin = min(subMin, acc / SUB)
        }
        var centroid = 0.0; var flatness = 0.0; var flux = 0.0
        var bestCpp = Double.NEGATIVE_INFINITY; var bestF0 = 0.0
        val bandSum = DoubleArray(N_BANDS)
        for (k in 0 until HOP / SHOP) {
            val start = (f * (HOP / SHOP) + k) * SHOP + SHOP / 2
            for (i in 0 until NFFT) { re[i] = buf[start + i] * win[i]; im[i] = 0.0 }
            fft.transform(re, im, false)
            var total = 0.0; var weighted = 0.0; var logSum = 0.0
            for (b in 0 until N_BINS) {
                val p = (re[b] * re[b] + im[b] * im[b]) * binNorm[b] + 1e-20
                power[b] = p; total += p; weighted += p * freqs[b]; logSum += ln(p)
                if (bandOf[b] >= 0) bandSum[bandOf[b]] += p
            }
            centroid += weighted / total
            flatness += exp(logSum / N_BINS) / (total / N_BINS)
            var fl = 0.0
            val prev = prevMag
            for (b in 0 until N_BINS) {
                mag[b] = sqrt(power[b])
                if (prev != null) fl += max(mag[b] - prev[b], 0.0)
            }
            flux += fl
            prevMag = (prevMag ?: DoubleArray(N_BINS)).also { System.arraycopy(mag, 0, it, 0, N_BINS) }
            // real cepstrum: inverse FFT of the (symmetric) log power spectrum, scaled by 1/n
            for (b in 0 until N_BINS) { re[b] = ln(power[b]); im[b] = 0.0 }
            for (b in 1 until N_BINS - 1) { re[NFFT - b] = re[b]; im[NFFT - b] = 0.0 }
            fft.transform(re, im, true)
            var peak = 0
            var peakV = Double.NEGATIVE_INFINITY
            for (q in Q_LO until Q_HI) {
                val v = re[q] / NFFT
                cepWindow[q - Q_LO] = v
                if (v > peakV) { peakV = v; peak = q - Q_LO }
            }
            val cpp = (peakV - Stats.median(cepWindow.copyOf())) * (10.0 / ln(10.0))
            if (cpp > bestCpp) { bestCpp = cpp; bestF0 = SR.toDouble() / (peak + Q_LO) }
        }
        val per = (HOP / SHOP).toDouble()
        cols[0].add(db10(lk / HOP)); cols[1].add(db10(ms / HOP))
        cols[2].add(db10(subMax)); cols[3].add(db10(subMin))
        cols[4].add(crossings.toDouble() / (HOP - 1))
        cols[5].add(centroid / per); cols[6].add(db10(flatness / per))
        cols[7].add(db10((flux / per) * (flux / per)))
        cols[8].add(bestF0); cols[9].add(bestCpp)
        cols[10].add(if (haveSide) db10(sideMs / HOP) else DB_FLOOR)
        for (b in 0 until N_BANDS) bandRows.add(db10(bandSum[b] / per))
    }
}

/** A growable float array. */
internal class FloatList(initial: Int = 1024) {
    private var a = FloatArray(initial)
    var size = 0
        private set

    fun add(v: Double) {
        if (size == a.size) a = a.copyOf(a.size * 2)
        a[size++] = v.toFloat()
    }

    fun toArray(): FloatArray = a.copyOf(size)
}
