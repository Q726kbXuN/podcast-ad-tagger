package podcast.adtagger

/** A recognized word with its times in seconds; [text] keeps the recognizer's leading space. */
data class Word(val start: Double, val end: Double, val text: String)

/**
 * Cuts a recognizer's output into sentence-sized segments, which is what the tagger was trained
 * on.  Call [split] once per recognizer segment (don't join words across them).
 */
object SentenceSplitter {
    private val ABBREVIATIONS = setOf("mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "vs.", "etc.", "inc.",
        "co.", "no.", "u.s.", "u.k.", "e.g.", "i.e.", "a.m.", "p.m.", "mt.", "ft.", "approx.")
    private val SENTENCE_END = Regex("[.?!][\"')\\]]*$")
    const val MAX_SEGMENT_S = 15.0

    fun isSentenceEnd(word: String): Boolean {
        val w = word.trim().lowercase()
        return SENTENCE_END.containsMatchIn(w) && w !in ABBREVIATIONS
    }

    fun split(words: List<Word>, maxLen: Double = MAX_SEGMENT_S): List<Segment> {
        val out = ArrayList<Segment>()
        var cur = ArrayList<Word>()
        fun flush(upto: Int = cur.size) {
            val take = cur.subList(0, upto).toList()
            cur = ArrayList(cur.subList(upto, cur.size))
            if (take.isNotEmpty()) out.add(Segment(take.first().start, take.last().end,
                take.joinToString("") { it.text }.trim(), take.size))
        }
        for (w in words) {
            cur.add(w)
            if (isSentenceEnd(w.text)) flush()
            else if (cur.last().end - cur.first().start > maxLen && cur.size > 2) {
                var best = 0
                for (k in 1 until cur.size - 1)
                    if (cur[k + 1].start - cur[k].end > cur[best + 1].start - cur[best].end) best = k
                flush(best + 1)
            }
        }
        flush()
        return out
    }
}
