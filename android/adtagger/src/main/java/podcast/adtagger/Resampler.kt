package podcast.adtagger

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Streaming band-limited resampler (windowed sinc, Kaiser beta 9, cutoff 0.97 of the lower
 * Nyquist, 16 zero crossings a side), close to ffmpeg's default resampler.  Android has no
 * public resampling API, and the features are defined at 24 kHz.  Common rate pairs (44.1 or
 * 48 kHz in, 24 or 16 kHz out) use a precomputed polyphase bank, so it's a plain dot product
 * per output sample.
 */
class Resampler(private val inRate: Int, private val outRate: Int) {
    private val fc = 0.97 * min(inRate, outRate) / (2.0 * inRate)     // cutoff, cycles per input sample
    private val half = ceil(16.0 / (2.0 * fc)).toInt()                // filter half-width, input samples
    private val taps = 2 * half
    private val g = gcd(inRate, outRate)
    private val phases = outRate / g
    private val i0b = besselI0(BETA)
    private val bank: Array<FloatArray>? = if (phases <= 4096) Array(phases) { p ->
        val frac = p.toDouble() * g / outRate
        FloatArray(taps) { j -> h(j - half + 1 - frac).toFloat() }
    } else null

    private var hist = FloatArray(1 shl 15)
    private var histLen = 0
    private var histBase = 0L            // input index of hist[0]
    private var nextOut = 0L             // index of the next output sample

    private fun h(d: Double): Double {
        if (abs(d) >= half) return 0.0
        val x = 2.0 * fc * d
        val sinc = if (x == 0.0) 1.0 else sin(PI * x) / (PI * x)
        val r = d / half
        return 2.0 * fc * sinc * besselI0(BETA * sqrt(max(0.0, 1.0 - r * r))) / i0b
    }

    private val passthrough = inRate == outRate   // like ffmpeg: no filtering when nothing changes

    /** Resample [count] samples of [input] from [offset]; returns whatever output is ready. */
    fun process(input: FloatArray, offset: Int = 0, count: Int = input.size - offset): FloatArray {
        if (passthrough) return input.copyOfRange(offset, offset + count)
        if (histLen + count > hist.size) hist = hist.copyOf(max(hist.size * 2, histLen + count))
        System.arraycopy(input, offset, hist, histLen, count)
        histLen += count
        return produce(Long.MAX_VALUE)
    }

    /** Flush the tail at the end of the stream. */
    fun finish(): FloatArray {
        if (passthrough) return FloatArray(0)
        val end = histBase + histLen
        val pad = half + 1
        if (histLen + pad > hist.size) hist = hist.copyOf(histLen + pad)
        java.util.Arrays.fill(hist, histLen, histLen + pad, 0f)
        histLen += pad
        return produce(end)
    }

    private fun produce(lastInput: Long): FloatArray {
        val out = FloatList(256)
        while (true) {
            // output sample n sits at input position n * inRate / outRate
            val num = nextOut * inRate
            val center = num / outRate
            if (center >= lastInput) break
            if (center + half >= histBase + histLen) break
            val rem = num % outRate
            val base = (center - half + 1 - histBase).toInt()   // hist index of the first tap
            val j0 = if (base < 0) -base else 0
            var acc = 0.0
            val b = bank?.get((rem / g).toInt())
            // the hot loop: keep the common (precomputed table) case free of per-tap branches
            if (b != null) {
                for (j in j0 until taps) acc += hist[base + j].toDouble() * b[j]
            } else {
                val frac = rem.toDouble() / outRate
                for (j in j0 until taps) acc += hist[base + j].toDouble() * h(j - half + 1 - frac)
            }
            out.add(acc)
            nextOut++
        }
        val keepFrom = (nextOut * inRate) / outRate - half - 1
        val drop = (keepFrom - histBase).toInt()
        if (drop in 1 until histLen) {
            System.arraycopy(hist, drop, hist, 0, histLen - drop)
            histLen -= drop; histBase += drop
        }
        return out.toArray()
    }

    companion object {
        private const val BETA = 9.0

        private fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

        private fun besselI0(x: Double): Double {
            var sum = 1.0; var term = 1.0; var k = 1
            while (term > 1e-12 * sum) { term *= (x / (2 * k)) * (x / (2 * k)); sum += term; k++ }
            return sum
        }

        /** Convenience for a whole buffer. */
        fun resample(x: FloatArray, inRate: Int, outRate: Int): FloatArray {
            val r = Resampler(inRate, outRate)
            return r.process(x) + r.finish()
        }
    }
}
