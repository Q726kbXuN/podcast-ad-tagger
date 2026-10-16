import Foundation
import XCTest
@testable import PodcastAdTagger

/// Checks against ../testdata, which the Python reference implementation generates.
final class ReferenceTests: XCTestCase {
    static let repo = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static func json(_ name: String) throws -> [String: Any] {
        let url = repo.appendingPathComponent("testdata").appendingPathComponent(name)
        return try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    }
    static func floats(_ b64: String) -> [Float] {
        let d = Data(base64Encoded: b64)!
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    static func rows(_ any: Any?) -> [[Float]] {
        (any as! [[NSNumber]]).map { $0.map(\.floatValue) }
    }

    /// synth.wav: 16-bit stereo PCM at 24 kHz -> (mid, side) the way the reference builds them.
    static func synthMidSide() throws -> ([Float], [Float]) {
        let d = try Data(contentsOf: repo.appendingPathComponent("testdata/synth.wav"))
        var off = 12
        var pcm = Data()
        while off + 8 <= d.count {
            let id = String(data: d[off..<(off + 4)], encoding: .ascii)!
            let size = Int(d[(off + 4)..<(off + 8)].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
            if id == "data" { pcm = d[(off + 8)..<(off + 8 + size)]; break }
            off += 8 + size + (size & 1)
        }
        let s: [Int16] = pcm.withUnsafeBytes { Array($0.bindMemory(to: Int16.self)) }
        let n = s.count / 2
        var mid = [Float](repeating: 0, count: n), side = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let l = Float(s[2 * i]) / 32768, r = Float(s[2 * i + 1]) / 32768
            mid[i] = (l + r) * 0.5
            side[i] = (l - r) * 0.5
        }
        return (mid, side)
    }

    func testRealFFTMatchesNaiveDFT() {
        let n = 64
        let x = (0..<n).map { Float(sin(Double($0) * 0.7) + 0.3 * cos(Double($0) * 2.1) + Double($0 % 5) * 0.1) }
        let fft = RealFFT(count: n)
        let (re, im) = fft.forward(x)
        for k in 0...(n / 2) {
            var r = 0.0, i = 0.0
            for t in 0..<n {
                r += Double(x[t]) * cos(-2 * Double.pi * Double(k * t) / Double(n))
                i += Double(x[t]) * sin(-2 * Double.pi * Double(k * t) / Double(n))
            }
            XCTAssertEqual(Double(re[k]), r, accuracy: 1e-3, "re[\(k)]")
            XCTAssertEqual(Double(im[k]), (k == 0 || k == n / 2) ? 0 : i, accuracy: 1e-3, "im[\(k)]")
        }
        let back = fft.inverse(re: re, im: im)
        for t in 0..<n { XCTAssertEqual(back[t], x[t], accuracy: 1e-4) }
    }

    func testResamplerSine() {
        // 1 kHz at 44.1 kHz -> 24 kHz: the same sine, and ffmpeg's output length (ceil(N * 80 / 147))
        let n = 44100 * 3
        let x = (0..<n).map { Float(0.5 * sin(2 * Double.pi * 1000 * Double($0) / 44100)) }
        let r = Resampler(inputRate: 44100)!
        XCTAssertEqual([r.length, r.phases, r.step], [62, 80, 147])
        var y = [Float]()
        var i = 0
        while i < n { y += r.process(Array(x[i..<min(i + 10007, n)])); i += 10007 }
        y += r.finish()
        XCTAssertEqual(y.count, Int((Double(n) * 80 / 147).rounded(.up)))
        var worst: Float = 0
        for k in 200..<(y.count - 200) {
            worst = max(worst, abs(y[k] - Float(0.5 * sin(2 * Double.pi * 1000 * Double(k) / 24000))))
        }
        XCTAssertLessThan(worst, 1e-3)
        XCTAssertNil(Resampler(inputRate: 24000))
        XCTAssertEqual(Resampler(inputRate: 48000)!.length, 66)
    }

    func checkAudio(_ key: String, stereo: Bool) throws {
        let ref = try Self.json("audio_reference.json")
        let (mid, side) = try Self.synthMidSide()
        let ex = FrameExtractor()
        var i = 0
        while i < mid.count {           // odd chunk size, like the reference
            let j = min(i + 7919, mid.count)
            ex.feed(mid: Array(mid[i..<j]), side: stereo ? Array(side[i..<j]) : nil)
            i = j
        }
        let fr = ex.finish()
        let want = Self.rows((ref[key] as! [String: Any])["frames"])
        XCTAssertEqual(fr.count, want.count, "frame count")
        let cols = ref["frame_columns"] as! [String]
        var worst = [String: Float]()
        var f0Mismatch = 0
        for r in 0..<min(fr.count, want.count) {
            let got = (0..<Frames.names.count).map { fr.columns[$0][r] } + fr.bands[r]
            for c in 0..<cols.count {
                let d = abs(got[c] - want[r][c])
                if cols[c] == "f0_hz" { if d > 0.5 { f0Mismatch += 1 }; continue }
                worst[cols[c]] = max(worst[cols[c]] ?? 0, d)
            }
        }
        print("[\(key)] frames worst |diff|:", worst.sorted { $0.value > $1.value }.prefix(6).map { "\($0.key)=\($0.value)" },
              "f0 mismatches:", f0Mismatch, "of", fr.count)
        for (c, d) in worst {
            let tol: Float = c == "centroid_hz" ? 0.5 : (c == "zcr" ? 1e-4 : 0.02)
            XCTAssertLessThanOrEqual(d, tol, "\(key) frame column \(c)")
        }
        XCTAssertLessThanOrEqual(Double(f0Mismatch) / Double(fr.count), 0.02, "f0 agreement")

        // per-segment features
        let segs = (ref["segments"] as! [[String: Any]]).map {
            Segment(start: ($0["start"] as! NSNumber).doubleValue, end: ($0["end"] as! NSNumber).doubleValue,
                    text: "", nWords: ($0["n_words"] as! NSNumber).intValue)
        }
        let got = SegmentFeatures.compute(frames: fr, segments: segs)
        let wantF = Self.rows((ref[key] as! [String: Any])["features"])
        var worstF = [String: Float]()
        for s in 0..<segs.count {
            for c in 0..<SegmentFeatures.names.count {
                let d = abs(got[s][c] - wantF[s][c]) / max(1, abs(wantF[s][c]))
                worstF[SegmentFeatures.names[c]] = max(worstF[SegmentFeatures.names[c]] ?? 0, d)
            }
        }
        print("[\(key)] features worst relative |diff|:", worstF.sorted { $0.value > $1.value }.prefix(6).map { "\($0.key)=\($0.value)" })
        for (c, d) in worstF {
            XCTAssertLessThanOrEqual(d, c == "f0_std" ? 0.05 : 0.01, "\(key) feature \(c)")
        }
    }

    func testAudioStereo() throws { try checkAudio("stereo", stereo: true) }
    func testAudioMono() throws { try checkAudio("mono", stereo: false) }

    func testDecodeMatchesExactly() throws {
        let ref = try Self.json("decode_reference.json")
        for (n, c) in (ref["cases"] as! [[String: Any]]).enumerated() {
            let probs = Self.rows(c["probs"])
            let segs = (c["segments"] as! [[String: Any]]).map {
                Segment(start: ($0["start"] as! NSNumber).doubleValue, end: ($0["end"] as! NSNumber).doubleValue, text: "")
            }
            let spans = Decoder.spans(probs: probs, segments: segs)
            let cands = Decoder.candidates(probs: probs, spans: spans, segments: segs)
            let wantSpans = try JSONDecoder().decode([AdSpan].self, from: JSONSerialization.data(withJSONObject: c["spans"]!))
            let wantCands = try JSONDecoder().decode([AdCandidate].self, from: JSONSerialization.data(withJSONObject: c["candidates"]!))
            XCTAssertEqual(spans, wantSpans, "case \(n) spans")
            XCTAssertEqual(cands, wantCands, "case \(n) candidates")
        }
    }

    func testTaggerMatchesReference() throws {
        let ref = try Self.json("tagger_reference.json")
        let n = (ref["n"] as! NSNumber).intValue
        let emb = Self.floats(ref["emb"] as! String), feats = Self.floats(ref["feats"] as! String)
        let want = Self.floats(ref["probs"] as! String)
        let tagger = try Tagger(modelURL: Self.repo.appendingPathComponent("model/ad_tagger.onnx"))
        let got = try tagger.probs(embeddings: (0..<n).map { Array(emb[($0 * 1024)..<($0 * 1024 + 1024)]) },
                                   features: (0..<n).map { Array(feats[($0 * 28)..<($0 * 28 + 28)]) })
        var worst: Float = 0
        for i in 0..<n { for k in 0..<5 { worst = max(worst, abs(got[i][k] - want[i * 5 + k])) } }
        print("tagger worst |probs diff|:", worst)
        XCTAssertLessThan(worst, 1e-4)
    }

    func testSentenceSplitter() {
        let words = [TimedWord(start: 0, end: 0.4, text: " Hello"), TimedWord(start: 0.4, end: 0.9, text: " there."),
                     TimedWord(start: 1.0, end: 1.3, text: " Dr."), TimedWord(start: 1.3, end: 1.8, text: " Smith"),
                     TimedWord(start: 1.8, end: 2.2, text: " agrees!")]
        let segs = SentenceSplitter.split(words)
        XCTAssertEqual(segs.map(\.text), ["Hello there.", "Dr. Smith agrees!"])
        XCTAssertEqual(segs.map { $0.nWords! }, [2, 3])
    }

    func testEmbeddingsMatchReference() throws {
        let path = ProcessInfo.processInfo.environment["PODCAST_AD_TAGGER_GGUF"]
            ?? Self.repo.appendingPathComponent("model/Qwen3-Embedding-0.6B-Q8_0.gguf").path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("no GGUF at \(path); set PODCAST_AD_TAGGER_GGUF to run this one")
        }
        let ref = try Self.json("embedding_reference.json")
        let texts = ref["texts"] as! [String]
        let dim = (ref["dim"] as! NSNumber).intValue
        let want = Self.floats(ref["embeddings"] as! String)
        let got = try Embedder(modelURL: URL(fileURLWithPath: path)).embed(texts)
        var worst = 1.0
        for i in 0..<texts.count {
            let w = want[(i * dim)..<(i * dim + dim)]
            var dot = 0.0, nw = 0.0, ng = 0.0
            for (a, b) in zip(got[i], w) { dot += Double(a) * Double(b); nw += Double(b) * Double(b); ng += Double(a) * Double(a) }
            worst = min(worst, dot / (nw.squareRoot() * ng.squareRoot()))
        }
        print("embedding worst cosine:", worst)
        XCTAssertGreaterThanOrEqual(worst, 0.999)
    }
}
