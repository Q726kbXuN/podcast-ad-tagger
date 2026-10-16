import Foundation

public struct AdResult: Codable, Sendable {
    /// episode length in seconds, from the audio analysis
    public var duration: Double
    public var segments: Int
    public var spans: [AdSpan]
    public var candidates: [AdCandidate]
}

/// Embedder + tagger + decoding, for when you already have the frames and segments.  Load it
/// once and reuse it: the embedder is the expensive part to load.
public final class AdTagger {
    public let embedder: Embedder
    public let tagger: Tagger

    public init(taggerModel: URL, embeddingModel: URL, gpuLayers: Int32 = 99) throws {
        tagger = try Tagger(modelURL: taggerModel)
        embedder = try Embedder(modelURL: embeddingModel, gpuLayers: gpuLayers)
    }

    /// frames: nil to run text-only (much less accurate).
    public func tag(frames: Frames?, segments: [Segment]) throws -> AdResult {
        let duration = frames.map { Double($0.count) / 10 } ?? (segments.last?.end ?? 0)
        guard !segments.isEmpty else { return AdResult(duration: duration, segments: 0, spans: [], candidates: []) }
        let emb = try embedder.embed(segments.map(\.text))
        let feats = SegmentFeatures.compute(frames: frames, segments: segments)
        let probs = try tagger.probs(embeddings: emb, features: feats)
        let spans = Decoder.spans(probs: probs, segments: segments)
        return AdResult(duration: duration, segments: segments.count, spans: spans,
                        candidates: Decoder.candidates(probs: probs, spans: spans, segments: segments))
    }
}

/// The whole thing: audio file in, ad spans out, all on device.
@available(iOS 26, macOS 26, *)
public final class AdDetector {
    public let core: AdTagger
    public let transcriber: Transcriber

    public init(taggerModel: URL, embeddingModel: URL, locale: Locale = Locale(identifier: "en-US"),
                gpuLayers: Int32 = 99) throws {
        core = try AdTagger(taggerModel: taggerModel, embeddingModel: embeddingModel, gpuLayers: gpuLayers)
        transcriber = Transcriber(locale: locale)
    }

    /// segments: pass a transcript you already have (sentence-sized) to skip transcription.
    public func detect(audio url: URL, segments: [Segment]? = nil,
                       progress: (@Sendable (String) -> Void)? = nil) async throws -> AdResult {
        progress?("transcribing")
        async let transcript: [Segment] = segments == nil ? transcriber.transcribe(url) : segments!
        progress?("measuring audio")
        let frames = try await Task.detached(priority: .utility) { try AudioDecoder.frames(of: url) }.value
        let segs = try await transcript
        progress?("tagging")
        return try core.tag(frames: frames, segments: segs)
    }
}
