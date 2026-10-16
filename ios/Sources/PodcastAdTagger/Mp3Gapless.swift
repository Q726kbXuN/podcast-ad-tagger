import Foundation

/// MP3 start padding that ffmpeg removes and Apple's decoder doesn't.
///
/// MP3 encoders note their start delay in the Info/Xing frame's LAME extension.  ffmpeg (which
/// decoded the training audio) honors it for LAME, Lavc and Lavf tags.  Apple's decoder honors it
/// for LAME tags but not for the ones FFmpeg writes, which many podcast hosts produce, so those
/// files decode 576 samples (13 ms) late.  Trimming that keeps the 100 ms frames lined up with
/// what the model was trained on.
enum Mp3Gapless {
    /// samples to drop from the start of AVAudioFile's output (at the file's own rate)
    static func extraStartTrim(_ url: URL) -> Int {
        guard let fh = try? FileHandle(forReadingFrom: url) else { return 0 }
        defer { try? fh.close() }
        guard var d = try? fh.read(upToCount: 16) , d.count >= 10 else { return 0 }
        var offset = 0
        if d.prefix(3) == Data("ID3".utf8) {
            let b = [UInt8](d)
            offset = 10 + (Int(b[6]) << 21 | Int(b[7]) << 14 | Int(b[8]) << 7 | Int(b[9]))
            if b[5] & 0x10 != 0 { offset += 10 }   // ID3v2 footer
        }
        try? fh.seek(toOffset: UInt64(offset))
        guard let chunk = try? fh.read(upToCount: 64 * 1024) else { return 0 }
        d = chunk
        let b = [UInt8](d)
        var i = 0
        while i + 4 < b.count && !(b[i] == 0xFF && b[i + 1] & 0xE0 == 0xE0) { i += 1 }
        guard i + 4 < b.count else { return 0 }
        let version = (b[i + 1] >> 3) & 3            // 3 = MPEG-1
        let mono = (b[i + 3] >> 6) & 3 == 3
        let side = version == 3 ? (mono ? 17 : 32) : (mono ? 9 : 17)
        var p = i + 4 + side
        guard p + 8 < b.count else { return 0 }
        let tag = String(bytes: b[p..<(p + 4)], encoding: .ascii)
        guard tag == "Info" || tag == "Xing" else { return 0 }
        p += 4
        let flags = Int(b[p]) << 24 | Int(b[p + 1]) << 16 | Int(b[p + 2]) << 8 | Int(b[p + 3])
        p += 4
        if flags & 1 != 0 { p += 4 }
        if flags & 2 != 0 { p += 4 }
        if flags & 4 != 0 { p += 100 }
        if flags & 8 != 0 { p += 4 }
        guard p + 24 <= b.count else { return 0 }
        let encoder = String(bytes: b[p..<(p + 4)], encoding: .ascii) ?? ""
        guard encoder == "Lavc" || encoder == "Lavf" else { return 0 }
        let v = Int(b[p + 21]) << 16 | Int(b[p + 22]) << 8 | Int(b[p + 23])
        return v >> 12
    }
}
