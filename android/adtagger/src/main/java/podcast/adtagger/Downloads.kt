package podcast.adtagger

import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/** Fetches a model file once (needs the INTERNET permission); call off the main thread. */
object Downloads {
    fun fetch(url: String, dest: File, sha256: String? = null, progress: (Long, Long) -> Unit = { _, _ -> }): File {
        if (dest.exists()) return dest
        dest.parentFile?.mkdirs()
        val part = File(dest.path + ".part")
        val md = MessageDigest.getInstance("SHA-256")
        var conn = URL(url).openConnection() as HttpURLConnection
        conn.instanceFollowRedirects = true
        // Hugging Face redirects to a CDN on another host, which HttpURLConnection won't follow on its own
        var hops = 0
        while (conn.responseCode in 300..399 && hops++ < 5) {
            val next = URL(URL(url), conn.getHeaderField("Location"))
            conn.disconnect()
            conn = next.openConnection() as HttpURLConnection
        }
        val total = conn.contentLengthLong
        conn.inputStream.use { input ->
            part.outputStream().use { output ->
                val buf = ByteArray(1 shl 16)
                var done = 0L
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    output.write(buf, 0, n); md.update(buf, 0, n)
                    done += n
                    progress(done, total)
                }
            }
        }
        conn.disconnect()
        if (sha256 != null) {
            val got = md.digest().joinToString("") { "%02x".format(it) }
            if (!got.equals(sha256, ignoreCase = true)) {
                part.delete()
                error("checksum mismatch for $url")
            }
        }
        if (!part.renameTo(dest)) error("could not move ${part.name} into place")
        return dest
    }
}
