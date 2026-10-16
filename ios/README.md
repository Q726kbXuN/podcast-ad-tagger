# iOS

A Swift package that runs the whole thing on the phone: Apple's on-device speech to text, the audio measurements, the embedder (llama.cpp) and the tagger (ONNX Runtime).  Nothing leaves the device.

The transcription part uses `SpeechAnalyzer`, so that needs iOS 26 (or macOS 26).  Everything else works back to iOS 17 if you bring your own transcript.

## Adding it to an app

SwiftPM can't pull a package out of a subfolder of a git repo, so either copy this `ios` folder into your project, or add it as a local package (File > Add Package Dependencies > Add Local).  It pulls in two things on its own: llama.cpp's prebuilt xcframework and ONNX Runtime's Swift package.

Then the models:

* `model/ad_tagger.onnx` is 12 MB, just add it to the app bundle.
* The embedder, `Qwen3-Embedding-0.6B-Q8_0.gguf`, is about 640 MB, so download it on first run rather than shipping it.  `Embedder.ggufURL` and `Embedder.ggufSHA256` have the URL and the hash to check it against.  I'd put it in Application Support and mark it excluded from backup.

Using it:

```swift
import PodcastAdTagger

let detector = try AdDetector(
    taggerModel: Bundle.main.url(forResource: "ad_tagger", withExtension: "onnx")!,
    embeddingModel: downloadedGGUF)

let result = try await detector.detect(audio: episodeFile)
for span in result.spans where span.type == .sponsorAd {
    print("skip \(span.start) - \(span.end)")
}
```

Load the detector once and keep it around; loading the embedder is the slow part.  `result.candidates` are stretches the model wasn't sure about, which make more sense as a "maybe an ad" hint than as something to skip.

If you already have a transcript (from a server, or another recognizer), skip the speech part and use `AdTagger` directly with `AudioDecoder.frames(of:)` and your own `Segment`s.  Cut them into sentences first, `SentenceSplitter` does that if you have word timings.  [TRANSCRIPTION.md](../docs/TRANSCRIPTION.md) has the details.

## Things worth knowing

* The first transcription for a locale downloads Apple's speech model (`Transcriber.prepare()` does it up front), so it needs a network connection once.
* `detect()` is async and does the audio work off the main thread, but embedding and tagging are plain blocking calls.  Run it from a background task, and one episode at a time: the embedder wants around a gigabyte of memory.
* The audio goes through a port of ffmpeg's resampler instead of `AVAudioConverter`, and MP3s written by FFmpeg get their start padding trimmed, since Apple's decoder doesn't do that for them.  Both sound like nitpicks, but they moved the spectral features enough to matter, see [FEATURES.md](../docs/FEATURES.md).  Apple's MP3 decoder also hands back 16-bit samples, which shows up in a few features in near-silent stretches; it didn't change any spans in my tests.
* The prebuilt llama.cpp xcframework has no simulator slice anymore.  To run in the simulator, build your own with llama.cpp's `./build-xcframework.sh ios-sim ios-device macos` and point the `llama` binary target in `Package.swift` at it with `path:`.  (I haven't tried this one.)

## How fast

On an M4 Max Mac, a 29 minute episode takes about 21 seconds start to finish, and a 42 minute one about 33.  Almost all of that is transcription; the audio measurements take 2-6 seconds and embedding plus tagging is under a second on the GPU.  A phone will be slower, I haven't timed it on one yet.

## Testing

`swift test` checks everything against the reference values in `../testdata`.  The embedder test only runs if it can find the GGUF, either at `../model/Qwen3-Embedding-0.6B-Q8_0.gguf` or wherever `PODCAST_AD_TAGGER_GGUF` points.

There's also a small command line tool for the Mac:

```
swift run -c release adtag episode.mp3 --model ../model/ad_tagger.onnx --embed-model Qwen3-Embedding-0.6B-Q8_0.gguf
```

It prints what it found as JSON.  `--segments` takes a transcript instead of running speech to text, and `--segments-out` / `--features-out` dump the intermediate steps, which is handy for comparing against another implementation.
