package podcast.adtagger

import java.io.File
import java.util.concurrent.Executor

/**
 * The whole pipeline: audio file in, ad spans out.  [detect] blocks for a while (most of it is
 * speech to text), so call it from a background thread, or use [detectAsync].  The models load on
 * first use and stay loaded until [close].
 */
class AdDetector(
    private val files: ModelFiles,
    private val threads: Int = Embedder.defaultThreads(),
) : AutoCloseable {

    /** [tagger] is model/ad_tagger.onnx (bytes, so it can come straight from assets). */
    class ModelFiles(val tagger: ByteArray, val embedder: File, val whisper: File)

    private val taggerLazy = lazy { Tagger(files.tagger) }
    private val embedderLazy = lazy { Embedder(files.embedder.path, threads) }
    private val transcriberLazy = lazy { Transcriber(files.whisper.path, threads) }
    private val tagger by taggerLazy
    private val embedder by embedderLazy
    private val transcriber by transcriberLazy

    /**
     * [segments]: pass a transcript you already have (sentence-sized, see docs/TRANSCRIPTION.md)
     * to skip whisper entirely; the audio is still decoded for its features.
     */
    fun detect(audioPath: String, segments: List<Segment>? = null, progress: (String) -> Unit = {}): AdResult {
        progress("Decoding audio")
        val extractor = FrameExtractor()
        val pcm = if (segments == null) PcmBuffer() else null
        val frames = try {
            AudioDecoder.decode(audioPath, extractor, pcm)
            extractor.finish()
        } catch (e: Throwable) {
            pcm?.close()
            throw e
        }
        val segs = segments ?: pcm!!.use {
            progress("Transcribing")
            transcriber.transcribe(it)
        }
        val duration = Decoder.round(frames.n / 10.0, 1)
        if (segs.isEmpty()) return AdResult(duration, segs, emptyList(), emptyList())
        progress("Embedding ${segs.size} segments")
        val emb = embedder.embed(segs.map { it.text })
        progress("Tagging")
        val probs = tagger.probs(emb, SegmentFeatures.compute(frames, segs))
        val spans = Decoder.findSpans(probs, segs)
        return AdResult(duration, segs, spans, Decoder.findCandidates(probs, spans, segs))
    }

    /** Runs [detect] on [executor]; [callback] is called on that executor's thread. */
    fun detectAsync(audioPath: String, executor: Executor, callback: (Result<AdResult>) -> Unit) =
        executor.execute { callback(runCatching { detect(audioPath) }) }

    override fun close() {
        for (l in listOf(taggerLazy, embedderLazy, transcriberLazy))
            if (l.isInitialized()) l.value.close()
    }
}
