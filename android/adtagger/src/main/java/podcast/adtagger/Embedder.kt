package podcast.adtagger

import kotlin.math.max
import kotlin.math.min

internal object LlamaJni {
    init { System.loadLibrary("adtagger_llama") }
    @JvmStatic external fun load(path: String, threads: Int): Long
    @JvmStatic external fun dim(handle: Long): Int
    @JvmStatic external fun embed(handle: Long, texts: Array<ByteArray>): FloatArray
    @JvmStatic external fun free(handle: Long)
}

/**
 * Text embeddings: Qwen3-Embedding-0.6B (the Q8_0 GGUF) through llama.cpp, last-token pooled and
 * scaled to unit length.  Loading takes a second or two and about 650 MB; keep one around.
 */
class Embedder(modelPath: String, threads: Int = defaultThreads()) : AutoCloseable {
    private var handle = LlamaJni.load(modelPath, threads)
    val dim = LlamaJni.dim(handle)

    /** -> row-major [texts.size] x [dim]. */
    @Synchronized
    fun embed(texts: List<String>): FloatArray {
        check(handle != 0L) { "closed" }
        val out = FloatArray(texts.size * dim)
        var i = 0
        while (i < texts.size) {
            val chunk = texts.subList(i, min(i + 256, texts.size))
            val v = LlamaJni.embed(handle, Array(chunk.size) { input(chunk[it]).toByteArray(Charsets.UTF_8) })
            System.arraycopy(v, 0, out, i * dim, v.size)
            i += chunk.size
        }
        return out
    }

    @Synchronized
    override fun close() {
        if (handle != 0L) { LlamaJni.free(handle); handle = 0 }
    }

    companion object {
        const val MODEL_URL = "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf"
        const val MODEL_SHA256 = "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439"
        const val MAX_CHARS = 2000

        /** The exact string the model was trained on: trimmed, first 2000 characters, " " if empty. */
        fun input(text: String): String {
            val t = text.trim()
            val cut = if (t.codePointCount(0, t.length) > MAX_CHARS) t.substring(0, t.offsetByCodePoints(0, MAX_CHARS)) else t
            return cut.ifEmpty { " " }
        }

        fun defaultThreads() = max(1, min(4, Runtime.getRuntime().availableProcessors()))
    }
}
