package podcast.adtagger

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer

/**
 * The tagger (model/ad_tagger.onnx) through ONNX Runtime.  Input: one 1024-float unit embedding
 * and one 28-float feature row per segment; output: 5 probabilities per segment, row-major,
 * columns in [Decoder.CLASSES] order.  Small and fast; CPU is fine.
 */
class Tagger(modelBytes: ByteArray) : AutoCloseable {
    constructor(modelFile: File) : this(modelFile.readBytes())

    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(modelBytes, OrtSession.SessionOptions())

    fun probs(emb: FloatArray, feats: FloatArray): FloatArray {
        val n = feats.size / SegmentFeatures.COUNT
        require(emb.size == n * EMB_DIM) { "need $n x $EMB_DIM embeddings, got ${emb.size} floats" }
        if (n == 0) return FloatArray(0)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(emb), longArrayOf(1, n.toLong(), EMB_DIM.toLong())).use { e ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(feats), longArrayOf(1, n.toLong(), SegmentFeatures.COUNT.toLong())).use { f ->
                session.run(mapOf("emb" to e, "feats" to f)).use { r ->
                    val buf = (r.get(0) as OnnxTensor).floatBuffer
                    return FloatArray(buf.remaining()).also { buf.get(it) }
                }
            }
        }
    }

    override fun close() = session.close()

    companion object {
        const val EMB_DIM = 1024
    }
}
