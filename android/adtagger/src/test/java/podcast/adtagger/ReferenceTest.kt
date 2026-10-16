package podcast.adtagger

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64
import kotlin.math.abs
import kotlin.math.sqrt

/** Checks the pure-Kotlin parts against the reference values in ../testdata (plain JVM, no device). */
class ReferenceTest {
    private val root: File = generateSequence(File(System.getProperty("user.dir")).absoluteFile) { it.parentFile }
        .first { File(it, "testdata/audio_reference.json").exists() }

    private fun json(name: String) = JSONObject(File(root, "testdata/$name").readText())

    private fun floats(b64: String): FloatArray {
        val bb = ByteBuffer.wrap(Base64.getDecoder().decode(b64)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(bb.remaining()).also { bb.get(it) }
    }

    /** synth.wav: 16-bit stereo at 24 kHz -> (mid, side), computed in float like the reference. */
    private fun synth(): Pair<FloatArray, FloatArray> {
        val bytes = File(root, "testdata/synth.wav").readBytes()
        var p = 12
        while (String(bytes, p, 4, Charsets.US_ASCII) != "data") p += 8 + ByteBuffer.wrap(bytes, p + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
        val n = ByteBuffer.wrap(bytes, p + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int / 4
        val sb = ByteBuffer.wrap(bytes, p + 8, n * 4).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val mid = FloatArray(n); val side = FloatArray(n)
        for (i in 0 until n) {
            val l = sb.get(2 * i) / 32768f; val r = sb.get(2 * i + 1) / 32768f
            mid[i] = (l + r) * 0.5f; side[i] = (l - r) * 0.5f
        }
        return mid to side
    }

    private fun frames(mid: FloatArray, side: FloatArray?): Frames {
        val ex = FrameExtractor()
        var i = 0
        while (i < mid.size) {           // odd chunk size, like the reference: streaming must not matter
            val c = minOf(7919, mid.size - i)
            ex.feed(mid, side, i, c); i += c
        }
        return ex.finish()
    }

    private fun column(f: Frames, c: Int): FloatArray = when (c) {
        0 -> f.lk; 1 -> f.rms; 2 -> f.rms10max; 3 -> f.rms10min; 4 -> f.zcr; 5 -> f.centroid
        6 -> f.flatness; 7 -> f.flux; 8 -> f.f0; 9 -> f.cpp; 10 -> f.side
        else -> FloatArray(f.n) { f.bands[it * FrameExtractor.N_BANDS + (c - 11)] }
    }

    private fun checkAudio(name: String, side: Boolean) {
        val ref = json("audio_reference.json")
        val (mid, sd) = synth()
        val f = frames(mid, if (side) sd else null)
        val want = ref.getJSONObject(name).getJSONArray("frames")
        assertEquals(want.length(), f.n)
        val cols = ref.getJSONArray("frame_columns")
        for (c in 0 until cols.length()) {
            val col = column(f, c)
            var worst = 0.0; var off = 0
            for (i in 0 until f.n) {
                val d = abs(col[i] - want.getJSONArray(i).getDouble(c))
                worst = maxOf(worst, d)
                if (d > tolerance(cols.getString(c))) off++
            }
            // pitch is an argmax over near-equal cepstral peaks; allow a few frames to land on the other one
            val allowed = if (cols.getString(c) == "f0_hz") f.n / 50 else 0
            assertTrue("$name frame column ${cols.getString(c)}: $off frames off, worst $worst", off <= allowed)
        }
        val segs = ref.getJSONArray("segments").let { a ->
            (0 until a.length()).map { a.getJSONObject(it).let { s -> Segment(s.getDouble("start"), s.getDouble("end"), "", s.getInt("n_words")) } }
        }
        val feats = SegmentFeatures.compute(f, segs)
        val wantF = ref.getJSONObject(name).getJSONArray("features")
        val names = ref.getJSONArray("feature_names")
        for (i in segs.indices) for (k in 0 until SegmentFeatures.COUNT) {
            val w = wantF.getJSONArray(i).getDouble(k)
            val tol = if (names.getString(k) == "f0_std") 2.0 else 1e-3 + 1e-4 * abs(w)
            assertEquals("$name segment $i ${names.getString(k)}", w, feats[i * SegmentFeatures.COUNT + k].toDouble(), tol)
        }
    }

    private fun tolerance(column: String) = when (column) {
        "zcr" -> 1e-5
        "centroid_hz" -> 0.05
        "f0_hz" -> 0.01
        else -> 0.01   // dB columns
    }

    @Test fun audioStereo() = checkAudio("stereo", true)

    @Test fun audioMono() = checkAudio("mono", false)

    @Test fun decoding() {
        val cases = json("decode_reference.json").getJSONArray("cases")
        for (c in 0 until cases.length()) {
            val case = cases.getJSONObject(c)
            val rows = case.getJSONArray("probs")
            val probs = FloatArray(rows.length() * 5) { rows.getJSONArray(it / 5).getDouble(it % 5).toFloat() }
            val segs = case.getJSONArray("segments").let { a ->
                (0 until a.length()).map { a.getJSONObject(it).let { s -> Segment(s.getDouble("start"), s.getDouble("end")) } }
            }
            val spans = Decoder.findSpans(probs, segs)
            val cands = Decoder.findCandidates(probs, spans, segs)
            assertEquals("case $c spans", toJson(case.getJSONArray("spans")), spans.map {
                mapOf("start" to it.start, "end" to it.end, "type" to it.type, "p" to it.p,
                    "first_segment" to it.firstSegment, "last_segment" to it.lastSegment) })
            assertEquals("case $c candidates", toJson(case.getJSONArray("candidates")), cands.map {
                mapOf("start" to it.start, "end" to it.end, "p" to it.p) })
        }
    }

    private fun toJson(a: JSONArray) = (0 until a.length()).map { i ->
        val o = a.getJSONObject(i)
        o.keySet().associateWith { k -> o.get(k).let { v -> if (v is Number && k != "first_segment" && k != "last_segment") v.toDouble() else v } }
    }

    @Test fun sentences() {
        val w = listOf(Word(0.0, 0.4, " Thanks"), Word(0.4, 0.6, " to"), Word(0.6, 1.0, " Dr."), Word(1.0, 1.5, " Smith."),
            Word(1.6, 2.0, " Next"), Word(2.0, 2.2, " up?"))
        val s = SentenceSplitter.split(w)
        assertEquals(listOf("Thanks to Dr. Smith.", "Next up?"), s.map { it.text })
        assertEquals(listOf(4, 2), s.map { it.nWords })
    }

    @Test fun tagger() {
        val model = File(root, "model/ad_tagger.onnx")
        assertTrue("missing $model", model.exists())
        val tagger = try { Tagger(model) } catch (e: Throwable) { null }
        assumeTrue("ONNX Runtime's desktop natives didn't load here; skipping", tagger != null)
        val ref = json("tagger_reference.json")
        val emb = floats(ref.getString("emb")); val feats = floats(ref.getString("feats")); val want = floats(ref.getString("probs"))
        val got = tagger!!.use { it.probs(emb, feats) }
        var worst = 0.0
        for (i in got.indices) worst = maxOf(worst, abs(got[i] - want[i]).toDouble())
        assertTrue("tagger max |diff| $worst", worst < 1e-4)
        assertEquals(Decoder.labels(want).toList(), Decoder.labels(got).toList())
    }

    @Test fun resamplerKeepsATone() {
        val inRate = 44100
        val x = FloatArray(inRate) { kotlin.math.sin(2 * Math.PI * 1000.0 * it / inRate).toFloat() * 0.5f }
        val y = Resampler.resample(x, inRate, 24000)
        assertEquals(24000, y.size)
        var e = 0.0
        for (i in 200 until 23800) {
            val d = y[i] - 0.5 * kotlin.math.sin(2 * Math.PI * 1000.0 * i / 24000)
            e += d * d
        }
        assertTrue("resampled tone rms error ${sqrt(e / 23600)}", sqrt(e / 23600) < 1e-3)
    }
}
