# Android

This is the whole pipeline as an Android library module: decode the episode, transcribe it, embed the transcript, measure the audio, run the tagger, and turn the result into spans.  It's Kotlin, plus a little C++ to reach whisper.cpp and llama.cpp.

## Adding it to an app

Copy the `adtagger` directory into your project (or point your `settings.gradle.kts` at this one) and depend on it:

```kotlin
// settings.gradle.kts
include(":adtagger")

// app/build.gradle.kts
dependencies {
    implementation(project(":adtagger"))
}
```

You'll need the NDK and CMake from the SDK manager.  The first build downloads and compiles llama.cpp and whisper.cpp at the release tags pinned in `src/main/cpp/CMakeLists.txt`, so it takes a few minutes.  Only `arm64-v8a` is built.  You can add other ABIs to `abiFilters`, but on anything older it's slow enough that I wouldn't bother.

Open the `android` directory in Android Studio, or build from the command line with `./gradlew assembleRelease` (and `./gradlew test` for the unit tests).

## Model files

It needs three files:

* `ad_tagger.onnx` from `../model`.  It's 12 MB, so I just put it in the app's `assets`.
* The text embedder, `Qwen3-Embedding-0.6B-Q8_0.gguf`, about 640 MB.
* A whisper.cpp model.  `ggml-base.en.bin` (148 MB) is the one I'd start with on a phone.  `ggml-small.en.bin` (488 MB) is more accurate and about three times slower.

The big two are better downloaded on first run than shipped in the APK.  `Downloads.fetch()` does that, and checks the hash if you give it one.  That's the only reason the module asks for the `INTERNET` permission, so drop it from the manifest if you get the files some other way.

```kotlin
val dir = File(context.filesDir, "models")
val files = AdDetector.ModelFiles(
    tagger = context.assets.open("ad_tagger.onnx").use { it.readBytes() },
    embedder = Downloads.fetch(Embedder.MODEL_URL, File(dir, "embed.gguf"), Embedder.MODEL_SHA256),
    whisper = Downloads.fetch(Transcriber.BASE_EN_URL, File(dir, "ggml-base.en.bin")),
)
```

(`fetch` blocks too, so that belongs on a background thread as well.)

## Running it

`AdDetector.detect()` takes minutes, so it has to run off the main thread.  Call it from your own worker, or hand it an executor:

```kotlin
val detector = AdDetector(files)
detector.detectAsync(episodePath, Executors.newSingleThreadExecutor()) { result ->
    result.onSuccess { r ->
        for (span in r.spansOf(setOf("sponsor_ad"))) {
            // skip from span.start to span.end (seconds)
        }
    }
}
```

Keep one `AdDetector` around for a queue of episodes.  The models load on first use, which takes a few seconds, and stay loaded until `close()`.  For memory, figure about 650 MB for the embedder, plus the whisper model, plus around 230 MB per hour of episode for the 16 kHz audio while it's being transcribed.  That audio lives in native memory, since the Java heap limit would choke on a long episode, and it's freed as soon as whisper is done.

`r.candidates` are stretches that look like they might be a sponsor read but didn't make the cut.  I show those as "maybe an ad" with a one-tap skip rather than skipping them outright.

If you already have a transcript, pass it in as `detect(path, segments)` and whisper never runs.  The segments need to be sentence-sized; see [TRANSCRIPTION.md](../docs/TRANSCRIPTION.md).

## Timings

I haven't had it on a phone yet.  Running in Android 14 in a container on a fast desktop ARM CPU (release build), a 42 minute stereo episode took about 3 minutes start to finish: decoding plus the audio measurements about 80 seconds, whisper `base.en` (greedy, 4 threads) about 80 seconds, and the embeddings about 10 seconds.  A 29 minute mono one took a minute and a half.  Expect a phone to be somewhat slower, and to want this done in the background, ideally on a charger.  WorkManager is the usual way to do that.

The decoding step is slower than it should be.  About a third of it is MediaCodec handing buffers back and forth, and most of the rest is the resampler, so moving that into native code would be the first thing to try if it matters to you.

## What's been checked

The unit tests in `src/test` check the pure Kotlin parts (audio measurements, segment features, resampler, sentence splitting, span decoding, and the tagger through ONNX Runtime's desktop build) against `../testdata`.

Beyond that, the module was built with Gradle and the NDK and run in Android 14 (arm64) on two real episodes, three ways: decoding plus features compared number for number against the desktop version, the full detector on a server-made transcript, and the whole thing with whisper on the device.  Every sponsor read came out where the desktop version and the original server pipeline put it, within a second or two at the edges.  Android's MP3 decoder only gives 16-bit samples, which nudges a few of the features in near-silent stretches, but didn't change any spans.

It hasn't been run on an actual phone, so phone speed, memory pressure and background behavior are still on you to check.

If you want to port the audio side somewhere else, the details are in [FEATURES.md](../docs/FEATURES.md).
