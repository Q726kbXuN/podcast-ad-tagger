import AVFoundation
import Foundation
import Speech

public enum TranscriberError: Error {
    case unavailable
}

/// On-device speech to text with SpeechAnalyzer / SpeechTranscriber, cut into sentence segments.
@available(iOS 26, macOS 26, *)
public struct Transcriber: Sendable {
    public var locale: Locale

    public init(locale: Locale = Locale(identifier: "en-US")) { self.locale = locale }

    /// Downloads the speech model for the locale if the system doesn't have it yet.
    public func prepare() async throws {
        guard await SpeechTranscriber.isAvailable else { throw TranscriberError.unavailable }
        let transcriber = makeTranscriber()
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await request.downloadAndInstall()
        }
    }

    private func makeTranscriber() -> SpeechTranscriber {
        SpeechTranscriber(locale: locale, transcriptionOptions: [], reportingOptions: [],
                          attributeOptions: [.audioTimeRange])
    }

    public func transcribe(_ url: URL) async throws -> [Segment] {
        try await prepare()
        let transcriber = makeTranscriber()
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let file = try AVAudioFile(forReading: url)

        let collector = Task { () -> [Segment] in
            var out = [Segment]()
            for try await result in transcriber.results {
                guard result.isFinal else { continue }
                // Every run of the result's text is kept, so joining the words gives the text
                // back; runs without timing (spaces, punctuation) ride along with the word before.
                var words = [TimedWord]()
                var pending = ""
                for run in result.text.runs {
                    let piece = String(result.text[run.range].characters)
                    if let r = run.audioTimeRange {
                        words.append(TimedWord(start: r.start.seconds, end: r.end.seconds, text: pending + piece))
                        pending = ""
                    } else if words.isEmpty {
                        pending += piece
                    } else {
                        words[words.count - 1].text += piece
                    }
                }
                if words.isEmpty {
                    let text = String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !text.isEmpty {
                        out.append(Segment(start: result.range.start.seconds, end: result.range.end.seconds, text: text))
                    }
                } else {
                    out.append(contentsOf: SentenceSplitter.split(words))
                }
            }
            return out
        }
        if let last = try await analyzer.analyzeSequence(from: file) {
            try await analyzer.finalizeAndFinish(through: last)
        } else {
            await analyzer.cancelAndFinishNow()
        }
        var segments = try await collector.value.sorted { $0.start < $1.start }
        for i in segments.indices where segments[i].end <= segments[i].start {
            segments[i].end = segments[i].start + 0.01
        }
        return segments
    }
}
