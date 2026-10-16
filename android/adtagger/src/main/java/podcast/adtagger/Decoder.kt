package podcast.adtagger

import java.math.BigDecimal
import java.math.RoundingMode

/** Turns the tagger's per-segment probabilities into spans, exactly as the reference does. */
object Decoder {
    val CLASSES = listOf("content", "sponsor_ad", "house_promo", "membership_pitch", "boilerplate")
    const val MIN_SEGMENTS = 2      // a span needs at least this many segments
    const val MAX_GAP = 2           // bridge up to this many "content" segments inside a span
    const val CAND_MIN_P = 0.25     // candidates: sponsor probability at least this ...
    const val CAND_MIN_SEGMENTS = 2 // ... for at least this many segments
    private val K = CLASSES.size

    /** Python-style round(): correctly rounded, half to even on exact ties. */
    internal fun round(x: Double, digits: Int) = BigDecimal(x).setScale(digits, RoundingMode.HALF_EVEN).toDouble()

    fun labels(probs: FloatArray): IntArray = IntArray(probs.size / K) { i ->
        var best = 0
        for (c in 1 until K) if (probs[i * K + c] > probs[i * K + best]) best = c
        best
    }

    /** Smoothed runs of non-content labels: (first, last, class). */
    fun runs(labels: IntArray): List<Triple<Int, Int, Int>> {
        val out = ArrayList<Triple<Int, Int, Int>>()
        val n = labels.size
        var i = 0
        while (i < n) {
            val c = labels[i]
            if (c == 0) { i++; continue }
            var j = i
            while (j + 1 < n && (labels[j + 1] == c ||
                    (j + 1 + MAX_GAP < n && labels[j + 1] == 0 && (1..MAX_GAP).any { labels[j + 1 + it] == c }))) j++
            if (j - i + 1 >= MIN_SEGMENTS) out.add(Triple(i, j, c))
            i = j + 1
        }
        return out
    }

    /** [probs] is row-major [segments.size] x 5, as the tagger returns it. */
    fun findSpans(probs: FloatArray, segments: List<Segment>): List<Span> =
        runs(labels(probs)).map { (i, j, c) ->
            var s = 0.0
            for (r in i..j) s += probs[r * K + c]
            Span(round(segments[i].start, 2), round(segments[j].end, 2), CLASSES[c],
                round(s / (j - i + 1), 3), i, j)
        }

    fun findCandidates(probs: FloatArray, spans: List<Span>, segments: List<Segment>): List<Candidate> {
        val n = probs.size / K
        val inside = BooleanArray(n)
        for (sp in spans) for (r in sp.firstSegment..sp.lastSegment) inside[r] = true
        val hot = BooleanArray(n) { !inside[it] && probs[it * K + 1] >= CAND_MIN_P }
        val out = ArrayList<Candidate>()
        var i = 0
        while (i < n) {
            if (!hot[i]) { i++; continue }
            var j = i
            while (j + 1 < n && (hot[j + 1] || (j + 2 < n && hot[j + 2] && !inside[j + 1]))) j++
            if (j - i + 1 >= CAND_MIN_SEGMENTS) {
                var s = 0.0
                for (r in i..j) s += probs[r * K + 1]
                out.add(Candidate(round(segments[i].start, 2), round(segments[j].end, 2), round(s / (j - i + 1), 3)))
            }
            i = j + 1
        }
        return out
    }
}
