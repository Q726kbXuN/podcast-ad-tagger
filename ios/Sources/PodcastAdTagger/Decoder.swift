import Foundation

public enum AdClass: String, Codable, Sendable, CaseIterable {
    case content, sponsorAd = "sponsor_ad", housePromo = "house_promo",
         membershipPitch = "membership_pitch", boilerplate
}

/// A stretch the tagger calls something other than content.
public struct AdSpan: Codable, Sendable, Equatable {
    public var start: Double
    public var end: Double
    public var type: AdClass
    /// mean probability of `type` over the span's segments
    public var p: Double
    public var firstSegment: Int
    public var lastSegment: Int

    enum CodingKeys: String, CodingKey { case start, end, type, p, firstSegment = "first_segment", lastSegment = "last_segment" }
}

/// A stretch outside every span that might be a sponsor read: show it, don't skip it.
public struct AdCandidate: Codable, Sendable, Equatable {
    public var start: Double
    public var end: Double
    public var p: Double
}

/// Per-segment probabilities -> spans and candidates, the same rules as the reference.
public enum Decoder {
    public static let minSegments = 2
    public static let maxGap = 2
    public static let candMinP: Float = 0.25
    public static let candMinSegments = 2

    static func round(_ x: Double, _ places: Int) -> Double {
        let m = pow(10.0, Double(places))
        return (x * m).rounded() / m
    }

    /// class ids -> [(first, last, class)]: content gaps up to maxGap bridged, short runs dropped
    static func runs(_ labels: [Int]) -> [(Int, Int, Int)] {
        var out = [(Int, Int, Int)]()
        var i = 0
        let n = labels.count
        while i < n {
            let c = labels[i]
            if c == 0 { i += 1; continue }
            var j = i
            while j + 1 < n && (labels[j + 1] == c ||
                                (j + 1 + maxGap < n && labels[j + 1] == 0 &&
                                 (1...maxGap).contains { labels[j + 1 + $0] == c })) {
                j += 1
            }
            if j - i + 1 >= minSegments { out.append((i, j, c)) }
            i = j + 1
        }
        return out
    }

    static func argmax(_ row: [Float]) -> Int {
        var k = 0
        for i in 1..<row.count where row[i] > row[k] { k = i }
        return k
    }

    public static func spans(probs: [[Float]], segments: [Segment]) -> [AdSpan] {
        runs(probs.map(argmax)).map { i, j, c in
            let p = (i...j).reduce(0.0) { $0 + Double(probs[$1][c]) } / Double(j - i + 1)
            return AdSpan(start: round(segments[i].start, 2), end: round(segments[j].end, 2),
                          type: AdClass.allCases[c], p: round(p, 3), firstSegment: i, lastSegment: j)
        }
    }

    public static func candidates(probs: [[Float]], spans: [AdSpan], segments: [Segment]) -> [AdCandidate] {
        let n = probs.count
        let pSp = probs.map { $0[1] }
        var inside = [Bool](repeating: false, count: n)
        for s in spans { for k in s.firstSegment...s.lastSegment { inside[k] = true } }
        let hot = (0..<n).map { !inside[$0] && pSp[$0] >= candMinP }
        var out = [AdCandidate]()
        var i = 0
        while i < n {
            if !hot[i] { i += 1; continue }
            var j = i
            while j + 1 < n && (hot[j + 1] || (j + 2 < n && hot[j + 2] && !inside[j + 1])) { j += 1 }
            if j - i + 1 >= candMinSegments {
                let p = (i...j).reduce(0.0) { $0 + Double(pSp[$1]) } / Double(j - i + 1)
                out.append(AdCandidate(start: round(segments[i].start, 2), end: round(segments[j].end, 2), p: round(p, 3)))
            }
            i = j + 1
        }
        return out
    }
}
