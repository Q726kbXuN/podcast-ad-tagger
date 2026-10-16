// adtag: find the ads in an audio file with the on-device pipeline, print JSON.
//
//   adtag episode.mp3 --model ../model/ad_tagger.onnx --embed-model Qwen3-Embedding-0.6B-Q8_0.gguf
//
//   --segments FILE      use this transcript instead of transcribing: a JSON list of
//                        {start, end, text} or an object with "segments" (WhisperX style)
//   --segments-out FILE  also write the transcript segments that were tagged
//   --features-out FILE  also write the per-segment features (for comparing implementations)
//   --locale ID          speech locale (default en-US)
import Foundation
import PodcastAdTagger

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(2)
}

func loadSegments(_ path: String) throws -> [Segment] {
    let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path)))
    let list = (obj as? [String: Any])?["segments"] as? [[String: Any]] ?? obj as? [[String: Any]] ?? []
    return list.compactMap { s in
        guard let a = s["start"] as? Double, let b = s["end"] as? Double else { return nil }
        let words = s["words"] as? [[String: Any]]
        return Segment(start: a, end: b, text: (s["text"] as? String) ?? "",
                       nWords: (s["n_words"] as? Int) ?? words?.count,
                       speaker: (s["speaker"] as? Int) ?? (words?.first?["speaker"] as? Int))
    }
}

struct Output: Encodable {
    var duration: Double
    var segments: Int
    var spans: [AdSpan]
    var candidates: [AdCandidate]
    var timings_s: [String: Double]
}

@main
struct AdTag {
    static func main() async {
        var args = Array(CommandLine.arguments.dropFirst())
        func option(_ name: String) -> String? {
            guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
            let v = args[i + 1]
            args.removeSubrange(i...(i + 1))
            return v
        }
        let model = option("--model") ?? "ad_tagger.onnx"
        let embedModel = option("--embed-model") ?? "Qwen3-Embedding-0.6B-Q8_0.gguf"
        let segmentsPath = option("--segments")
        let featuresOut = option("--features-out")
        let segmentsOut = option("--segments-out")
        let locale = Locale(identifier: option("--locale") ?? "en-US")
        guard let audioPath = args.first else {
            fail("usage: adtag <audio> [--model ad_tagger.onnx] [--embed-model gguf] [--segments json] [--segments-out json] [--features-out json] [--locale en-US]")
        }
        let audio = URL(fileURLWithPath: audioPath)
        do {
            var timings = [String: Double]()
            func timed<T>(_ name: String, _ body: () async throws -> T) async rethrows -> T {
                let t = Date()
                let r = try await body()
                timings[name] = (Date().timeIntervalSince(t) * 100).rounded() / 100
                return r
            }
            let core = try await timed("load_models") {
                try AdTagger(taggerModel: URL(fileURLWithPath: model), embeddingModel: URL(fileURLWithPath: embedModel))
            }
            let frames = try await timed("audio_features") { try AudioDecoder.frames(of: audio) }
            let segments: [Segment]
            if let segmentsPath {
                segments = try loadSegments(segmentsPath)
            } else if #available(macOS 26, *) {
                segments = try await timed("transcribe") { try await Transcriber(locale: locale).transcribe(audio) }
            } else {
                fail("transcription needs macOS 26; pass --segments")
            }
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let segmentsOut {
                try enc.encode(segments).write(to: URL(fileURLWithPath: segmentsOut))
            }
            if let featuresOut {
                let f = SegmentFeatures.compute(frames: frames, segments: segments)
                try JSONSerialization.data(withJSONObject: ["names": SegmentFeatures.names, "features": f])
                    .write(to: URL(fileURLWithPath: featuresOut))
            }
            let result = try await timed("embed_and_tag") { try core.tag(frames: frames, segments: segments) }
            let out = Output(duration: result.duration, segments: result.segments, spans: result.spans,
                             candidates: result.candidates, timings_s: timings)
            print(String(data: try enc.encode(out), encoding: .utf8)!)
        } catch {
            fail("adtag: \(error)")
        }
    }
}
