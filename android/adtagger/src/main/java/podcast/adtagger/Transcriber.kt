package podcast.adtagger

internal object WhisperJni {
    init { System.loadLibrary("adtagger_whisper") }
    @JvmStatic external fun load(path: String): Long
    @JvmStatic external fun transcribe(handle: Long, pcm: Long, threads: Int, language: String, beamSize: Int): Array<Any>
    @JvmStatic external fun free(handle: Long)
    @JvmStatic external fun pcmNew(capacity: Int): Long
    @JvmStatic external fun pcmAppend(pcm: Long, x: FloatArray, offset: Int, count: Int)
    @JvmStatic external fun pcmFree(pcm: Long)
}

/**
 * 16 kHz mono float samples for whisper, kept in native memory: an hour of audio is about 230 MB,
 * more than an app's Java heap is allowed (direct ByteBuffers count against it on Android).
 * Close it when done, [AdDetector] does.
 */
class PcmBuffer(initialSamples: Int = 16000 * 60 * 10) : AutoCloseable {
    internal var handle = WhisperJni.pcmNew(initialSamples)
        private set
    var size = 0
        private set

    fun append(x: FloatArray, offset: Int = 0, count: Int = x.size - offset) {
        check(handle != 0L) { "closed" }
        WhisperJni.pcmAppend(handle, x, offset, count)
        size += count
    }

    override fun close() {
        if (handle != 0L) { WhisperJni.pcmFree(handle); handle = 0 }
    }
}

/**
 * Speech to text with whisper.cpp, returned as sentence segments (see docs/TRANSCRIPTION.md).
 * English models work best with the tagger: ggml-base.en is a good phone default,
 * ggml-small.en is better and about three times slower.
 */
class Transcriber(
    modelPath: String,
    private val threads: Int = Embedder.defaultThreads(),
    private val language: String = "en",
    private val beamSize: Int = 1,
) : AutoCloseable {
    private var handle = WhisperJni.load(modelPath)

    @Synchronized
    fun transcribe(pcm: PcmBuffer): List<Segment> {
        check(handle != 0L) { "closed" }
        if (pcm.size == 0) return emptyList()
        val r = WhisperJni.transcribe(handle, pcm.handle, threads, language, beamSize)
        val times = r[0] as FloatArray
        val texts = r[1] as Array<*>
        val out = ArrayList<Segment>()
        var group = ArrayList<Word>()
        var current = -1
        for (i in texts.indices) {
            val seg = times[3 * i].toInt()
            if (seg != current && group.isNotEmpty()) { out += SentenceSplitter.split(group); group = ArrayList() }
            current = seg
            val start = times[3 * i + 1].toDouble(); val end = times[3 * i + 2].toDouble()
            if (end >= start) group.add(Word(start, end, String(texts[i] as ByteArray, Charsets.UTF_8)))
        }
        if (group.isNotEmpty()) out += SentenceSplitter.split(group)
        return out.map { if (it.end <= it.start) it.copy(end = it.start + 0.01) else it }
    }

    @Synchronized
    override fun close() {
        if (handle != 0L) { WhisperJni.free(handle); handle = 0 }
    }

    companion object {
        const val BASE_EN_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
        const val SMALL_EN_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"
    }
}
