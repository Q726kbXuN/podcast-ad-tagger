package podcast.adtagger

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** In-place radix-2 complex FFT of a fixed power-of-two size.  Forward is e^(-2*pi*i*k*t/n), unscaled. */
internal class Fft(private val n: Int) {
    private val bits = Integer.numberOfTrailingZeros(n)
    private val cosT = DoubleArray(n / 2) { cos(2.0 * PI * it / n) }
    private val sinT = DoubleArray(n / 2) { sin(2.0 * PI * it / n) }
    private val rev = IntArray(n) { Integer.reverse(it) ushr (32 - bits) }

    init {
        require(n >= 2 && n and (n - 1) == 0) { "size must be a power of two" }
    }

    fun transform(re: DoubleArray, im: DoubleArray, inverse: Boolean) {
        for (i in 0 until n) {
            val j = rev[i]
            if (j > i) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
        }
        var size = 2
        while (size <= n) {
            val half = size / 2
            val step = n / size
            var start = 0
            while (start < n) {
                var k = 0
                for (j in start until start + half) {
                    val l = j + half
                    val c = cosT[k]
                    val s = if (inverse) sinT[k] else -sinT[k]
                    val tr = re[l] * c - im[l] * s
                    val ti = re[l] * s + im[l] * c
                    re[l] = re[j] - tr; im[l] = im[j] - ti
                    re[j] += tr; im[j] += ti
                    k += step
                }
                start += size
            }
            size *= 2
        }
    }
}
