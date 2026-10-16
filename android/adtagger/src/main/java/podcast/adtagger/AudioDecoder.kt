package podcast.adtagger

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.nio.ByteOrder

/**
 * Decodes an audio file or URL with MediaExtractor + MediaCodec and streams it into a
 * [FrameExtractor] (24 kHz mid and side) and, optionally, a 16 kHz mono [PcmBuffer] for whisper.
 */
object AudioDecoder {
    data class Info(val sampleRate: Int, val channels: Int, val durationUs: Long)

    fun decode(path: String, frames: FrameExtractor, pcm16k: PcmBuffer? = null): Info {
        val ex = MediaExtractor()
        ex.setDataSource(path)
        val track = (0 until ex.trackCount).firstOrNull {
            ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: error("no audio track in $path")
        ex.selectTrack(track)
        val format = ex.getTrackFormat(track)
        val duration = if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) else 0L
        var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        var encoding = AudioFormat.ENCODING_PCM_16BIT
        val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
        // Ask for float output: 16-bit samples turn near-silence into exact zeros, which shifts a few
        // of the audio features.  Decoders that can't do it either ignore the request or refuse it.
        try {
            format.setInteger(MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_FLOAT)
            codec.configure(format, null, null, 0)
        } catch (e: Exception) {
            codec.reset()
            codec.configure(ex.getTrackFormat(track), null, null, 0)
        }
        codec.start()

        var toFrames = Resampler(rate, FrameExtractor.SR)
        var toFramesSide = Resampler(rate, FrameExtractor.SR)
        var toWhisper = Resampler(rate, 16000)
        var started = false
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        try {
            while (!outputDone) {
                if (!inputDone) {
                    val i = codec.dequeueInputBuffer(10_000)
                    if (i >= 0) {
                        val n = ex.readSampleData(codec.getInputBuffer(i)!!, 0)
                        if (n < 0) {
                            codec.queueInputBuffer(i, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(i, 0, n, ex.sampleTime, 0)
                            ex.advance()
                        }
                    }
                }
                val o = codec.dequeueOutputBuffer(info, 10_000)
                if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val f = codec.outputFormat
                    val newRate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                    channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    if (f.containsKey(MediaFormat.KEY_PCM_ENCODING)) encoding = f.getInteger(MediaFormat.KEY_PCM_ENCODING)
                    if (newRate != rate && !started) {
                        rate = newRate
                        toFrames = Resampler(rate, FrameExtractor.SR)
                        toFramesSide = Resampler(rate, FrameExtractor.SR)
                        toWhisper = Resampler(rate, 16000)
                    }
                } else if (o >= 0) {
                    val buf = codec.getOutputBuffer(o)!!.order(ByteOrder.nativeOrder())
                    buf.position(info.offset); buf.limit(info.offset + info.size)
                    val stereo = channels >= 2
                    val count = if (encoding == AudioFormat.ENCODING_PCM_FLOAT) info.size / 4 / channels else info.size / 2 / channels
                    if (count > 0) {
                        started = true
                        val mid = FloatArray(count)
                        val side = if (stereo) FloatArray(count) else null
                        if (encoding == AudioFormat.ENCODING_PCM_FLOAT) {
                            val fb = buf.asFloatBuffer()
                            for (k in 0 until count) {
                                val l = fb.get(k * channels)
                                if (stereo) { val r = fb.get(k * channels + 1); mid[k] = (l + r) * 0.5f; side!![k] = (l - r) * 0.5f } else mid[k] = l
                            }
                        } else {
                            val sb = buf.asShortBuffer()
                            for (k in 0 until count) {
                                val l = sb.get(k * channels) / 32768f
                                if (stereo) { val r = sb.get(k * channels + 1) / 32768f; mid[k] = (l + r) * 0.5f; side!![k] = (l - r) * 0.5f } else mid[k] = l
                            }
                        }
                        feed(frames, toFrames.process(mid), side?.let { toFramesSide.process(it) })
                        pcm16k?.append(toWhisper.process(mid))
                    }
                    codec.releaseOutputBuffer(o, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                }
            }
            feed(frames, toFrames.finish(), if (channels >= 2) toFramesSide.finish() else null)
            pcm16k?.append(toWhisper.finish())
        } finally {
            codec.stop(); codec.release(); ex.release()
        }
        return Info(rate, channels, duration)
    }

    private fun feed(frames: FrameExtractor, mid: FloatArray, side: FloatArray?) {
        if (mid.isEmpty()) return
        frames.feed(mid, side?.takeIf { it.size == mid.size })
    }
}
