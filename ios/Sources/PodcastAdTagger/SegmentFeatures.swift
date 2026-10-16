import Foundation

/// One transcript segment: a sentence-sized piece of speech.
public struct Segment: Codable, Sendable {
    public var start: Double
    public var end: Double
    public var text: String
    /// word count; when nil the words in `text` are counted
    public var nWords: Int?
    /// speaker label from diarization, if the transcriber has one
    public var speaker: Int?

    public init(start: Double, end: Double, text: String, nWords: Int? = nil, speaker: Int? = nil) {
        self.start = start; self.end = end; self.text = text; self.nWords = nWords; self.speaker = speaker
    }

    enum CodingKeys: String, CodingKey { case start, end, text, nWords = "n_words", speaker }
}

/// The 28 per-segment features the tagger reads (docs/FEATURES.md).  Column order is the
/// model's `feats` input order.
public enum SegmentFeatures {
    public static let names = [
        "lk_mean", "lk_std", "rms10max", "rms10min", "flat_mean", "side_mean", "zcr_mean",
        "cpp_mean", "f0_std", "centroid_mean", "gap_before_s", "gap_after_s", "lk_delta_prev",
        "stage0_overlap", "rel_pos", "seg_dur", "n_words", "spk_change",
        "lk_vs_ep", "lk_win_delta", "lra_vs_ep", "side_win_delta", "flux_mean",
        "voiced_frac", "band_dist", "bass_delta", "edge_before", "edge_after",
    ]
    static let win = 50
    static let valid: Float = -190
    static let bassBands = 4

    private static func validOnly(_ a: ArraySlice<Float>) -> [Float] { a.filter { $0 > valid } }

    private struct Baselines {
        var lkMed: Float, lra: Float
        var act: [Bool]
        var sideC: [Float], sideMed: Float?
        var prof: [[Float]], profMed: [Float]
        var bass: [Float], bassMed: Float
    }

    private static func baselines(_ f: Frames) -> Baselines {
        let lkAll = f.column("lk_db").filter { $0 > valid }
        let lkMed = lkAll.isEmpty ? -70 : median(lkAll)
        let lra = lkAll.count > 10 ? percentile(lkAll, 95) - percentile(lkAll, 10) : 0
        let act = f.column("rms_db").map { $0 > -80 }
        let nAct = act.filter { $0 }.count
        let sideC = f.column("side_db").map { max($0, -120) }
        let sideMed: Float? = (f.stereo && nAct > 10) ? median(zip(sideC, act).filter { $0.1 }.map { $0.0 }) : nil
        let nb = FrameExtractor.nBands
        var prof = [[Float]](), bass = [Float]()
        prof.reserveCapacity(f.count); bass.reserveCapacity(f.count)
        for row in f.bands {
            let bc = row.map { max($0, -120) }
            let m = Float(bc.reduce(0.0) { $0 + Double($1) } / Double(nb))
            prof.append(bc.map { $0 - m })
            let lin = row.map { pow(10.0, Double(min(max($0, -120), 0)) / 10.0) }
            let low = lin[0..<bassBands].reduce(0, +), all = lin.reduce(0, +)
            bass.append(Float(low / (all + 1e-12)))
        }
        var profMed = [Float](repeating: 0, count: nb)
        if nAct > 10 {
            let rows = prof.indices.filter { act[$0] }
            for b in 0..<nb { profMed[b] = median(rows.map { prof[$0][b] }) }
        }
        let bassMed: Float = nAct > 10 ? median(bass.indices.filter { act[$0] }.map { bass[$0] }) : 0
        return Baselines(lkMed: lkMed, lra: lra, act: act, sideC: sideC, sideMed: sideMed,
                         prof: prof, profMed: profMed, bass: bass, bassMed: bassMed)
    }

    private static func edge(_ lk: [Float], _ i: Int) -> Float {
        let a = min(max(0, i - 10), lk.count), b = min(max(i, 0), lk.count)
        let pre = validOnly(lk[a..<max(a, b)])
        let post = validOnly(lk[b..<min(i + 10, lk.count)])
        return pre.count >= 3 && post.count >= 3 ? abs(mean(post) - mean(pre)) : 0
    }

    /// frames: nil for text-only (every audio feature stays 0).  Returns one row of 28 per segment.
    public static func compute(frames: Frames?, segments: [Segment]) -> [[Float]] {
        guard let last = segments.last else { return [] }
        let have = (frames?.count ?? 0) > 0
        let f = frames
        let col: [String: [Float]] = have ? Dictionary(uniqueKeysWithValues: Frames.names.map { ($0, f!.column($0)) }) : [:]
        let b = have ? baselines(f!) : nil
        let nfr = f?.count ?? 0
        let dur = last.end
        var prevLk: Float? = nil
        var out = [[Float]]()
        for (i, s) in segments.enumerated() {
            var row = [String: Float]()
            if have, let b {
                let f0i = max(0, Int(s.start * 10)), f1i = max(Int(s.start * 10) + 1, Int(s.end * 10))
                let lo = min(f0i, nfr), hi = max(lo, min(f1i, nfr))
                let sl = lo..<hi
                let lk = validOnly(col["lk_db"]![sl])
                row["lk_mean"] = lk.isEmpty ? -100 : mean(lk)
                row["lk_std"] = lk.isEmpty ? 0 : std(lk)
                row["rms10max"] = sl.isEmpty ? -100 : col["rms10max_db"]![sl].max()!
                row["rms10min"] = sl.isEmpty ? -100 : col["rms10min_db"]![sl].min()!
                for (k, c) in [("flat_mean", "flatness_db"), ("side_mean", "side_db"), ("zcr_mean", "zcr"),
                               ("cpp_mean", "cpp_db"), ("centroid_mean", "centroid_hz")] {
                    let v = validOnly(col[c]![sl])
                    row[k] = v.isEmpty ? 0 : mean(v)
                }
                let cpp = col["cpp_db"]!, f0 = col["f0_hz"]!
                let f0v = sl.filter { cpp[$0] > 1 }.map { f0[$0] }
                row["f0_std"] = f0v.count > 3 ? std(f0v) : 0
                let rmsDb = col["rms_db"]!
                let g0 = min(max(0, f0i - 30), nfr)
                row["gap_before_s"] = Float(rmsDb[g0..<max(g0, min(f0i, nfr))].filter { $0 < -60 }.count) / 10
                row["gap_after_s"] = Float(rmsDb[hi..<min(hi + 30, nfr)].filter { $0 < -60 }.count) / 10
                row["lk_delta_prev"] = prevLk.map { row["lk_mean"]! - $0 } ?? 0
                prevLk = row["lk_mean"]
                if !lk.isEmpty { row["lk_vs_ep"] = row["lk_mean"]! - b.lkMed }
                let c0 = (f0i + min(f1i, nfr)) / 2
                let w = max(0, c0 - win)..<max(max(0, c0 - win), min(nfr, c0 + win))
                let wlk = validOnly(col["lk_db"]![w])
                if wlk.count > 10 {
                    row["lk_win_delta"] = median(wlk) - b.lkMed
                    row["lra_vs_ep"] = (percentile(wlk, 95) - percentile(wlk, 10)) - b.lra
                }
                if let sm = b.sideMed, !w.isEmpty { row["side_win_delta"] = median(Array(b.sideC[w])) - sm }
                let fx = validOnly(col["flux_db"]![sl])
                row["flux_mean"] = fx.isEmpty ? -90 : mean(fx)
                if !sl.isEmpty { row["voiced_frac"] = Float(sl.filter { cpp[$0] > 1 }.count) / Float(sl.count) }
                let actIdx = sl.filter { b.act[$0] }
                if actIdx.count >= 3 {
                    var d = 0.0
                    for k in 0..<FrameExtractor.nBands {
                        d += Double(abs(median(actIdx.map { b.prof[$0][k] }) - b.profMed[k]))
                    }
                    row["band_dist"] = Float(d / Double(FrameExtractor.nBands))
                    row["bass_delta"] = median(actIdx.map { b.bass[$0] }) - b.bassMed
                }
                row["edge_before"] = edge(col["lk_db"]!, f0i)
                row["edge_after"] = edge(col["lk_db"]!, min(hi, nfr))
            }
            // stage0_overlap (repeated-audio history) stays 0, as it does in production
            row["rel_pos"] = Float(s.start / max(dur, 1))
            row["seg_dur"] = Float(s.end - s.start)
            row["n_words"] = Float(s.nWords ?? s.text.split(whereSeparator: { $0.isWhitespace }).count)
            let prev = i > 0 ? segments[i - 1].speaker : s.speaker
            row["spk_change"] = (s.speaker != nil && prev != nil && s.speaker != prev) ? 1 : 0
            out.append(names.map { row[$0] ?? 0 })
        }
        return out
    }
}
