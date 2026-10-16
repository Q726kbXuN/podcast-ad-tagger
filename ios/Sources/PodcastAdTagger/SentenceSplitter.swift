import Foundation

/// A recognized word with its timing.  `text` is the word as it appears in the transcript,
/// including any leading space and trailing punctuation, so joining words gives the text back.
public struct TimedWord: Sendable {
    public var start: Double
    public var end: Double
    public var text: String
    public init(start: Double, end: Double, text: String) { self.start = start; self.end = end; self.text = text }
}

/// Cuts a recognizer's output into sentence-sized segments, the size the tagger was trained on.
public enum SentenceSplitter {
    static let abbreviations: Set<String> = ["mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "vs.", "etc.", "inc.", "co.",
                                             "no.", "u.s.", "u.k.", "e.g.", "i.e.", "a.m.", "p.m.", "mt.", "ft.", "approx."]
    public static let maxSegmentSeconds = 15.0

    static func isSentenceEnd(_ word: String) -> Bool {
        let w = word.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard w.range(of: #"[.?!]["')\]]*$"#, options: .regularExpression) != nil else { return false }
        return !abbreviations.contains(w)
    }

    /// Words of one recognizer phrase -> sentence segments.  A run with no sentence end is cut
    /// at its widest pause once it passes maxLength seconds.
    public static func split(_ words: [TimedWord], maxLength: Double = maxSegmentSeconds) -> [Segment] {
        var out = [Segment]()
        var cur = [TimedWord]()

        func flush(_ upto: Int? = nil) {
            let take: [TimedWord]
            if let upto { take = Array(cur[..<upto]); cur = Array(cur[upto...]) } else { take = cur; cur = [] }
            guard let first = take.first, let last = take.last else { return }
            let text = take.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines)
            out.append(Segment(start: first.start, end: last.end, text: text, nWords: take.count))
        }

        for w in words {
            cur.append(w)
            if isSentenceEnd(w.text) {
                flush()
            } else if cur.last!.end - cur.first!.start > maxLength && cur.count > 2 {
                let gaps = (0..<(cur.count - 1)).map { cur[$0 + 1].start - cur[$0].end }
                flush(gaps.firstIndex(of: gaps.max()!)! + 1)
            }
        }
        flush()
        return out
    }
}
