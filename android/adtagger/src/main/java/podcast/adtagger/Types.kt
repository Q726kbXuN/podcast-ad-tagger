package podcast.adtagger

/** One sentence-sized piece of the transcript.  Times in seconds. */
data class Segment(
    val start: Double,
    val end: Double,
    val text: String = "",
    val nWords: Int = text.split(Regex("\\s+")).count { it.isNotEmpty() },
    val speaker: Int? = null,
)

/** A run of segments the tagger put in one class.  [p] is the mean probability of [type] over it. */
data class Span(
    val start: Double, val end: Double, val type: String, val p: Double,
    val firstSegment: Int, val lastSegment: Int,
)

/** A stretch outside every span that might be a sponsor read: show it, don't skip it. */
data class Candidate(val start: Double, val end: Double, val p: Double)

data class AdResult(
    val duration: Double,
    val segments: List<Segment>,
    val spans: List<Span>,
    val candidates: List<Candidate>,
) {
    /** The spans of the given types, e.g. what to skip with the default `setOf("sponsor_ad")`. */
    fun spansOf(types: Set<String> = setOf("sponsor_ad")) = spans.filter { it.type in types }
}
