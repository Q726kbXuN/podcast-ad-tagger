import AVFoundation
import Foundation

public enum AudioDecoderError: Error {
    case cannotConvert(String)
}

/// Decodes an audio file, resamples it to 24 kHz and runs the frame analysis on it.
public enum AudioDecoder {
    /// Stereo sources are split into mid = (L+R)/2 and side = (L-R)/2 after resampling; mono
    /// sources have no side.  The usual sample rates go through `Resampler` (ffmpeg's filter);
    /// odd rates and sources with more than two channels fall back to AVAudioConverter.
    public static func frames(of url: URL) throws -> Frames {
        let file = try AVAudioFile(forReading: url)
        let fmt = file.processingFormat
        let rate = Int(fmt.sampleRate.rounded())
        let channels = Int(fmt.channelCount)
        if channels > 2 || (rate != FrameExtractor.sampleRate && Resampler(inputRate: rate) == nil) {
            return try framesViaConverter(file)
        }
        let rs: [Resampler?] = (0..<channels).map { _ in rate == FrameExtractor.sampleRate ? nil : Resampler(inputRate: rate) }
        let ex = FrameExtractor()
        let chunk: AVAudioFrameCount = 1 << 16
        guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: chunk) else {
            throw AudioDecoderError.cannotConvert(url.lastPathComponent)
        }

        func push(_ chans: [[Float]]) {
            let n = chans.map(\.count).min() ?? 0
            if n == 0 { return }
            if chans.count == 2 {
                let l = chans[0], r = chans[1]
                var mid = [Float](repeating: 0, count: n), side = [Float](repeating: 0, count: n)
                for i in 0..<n { mid[i] = (l[i] + r[i]) * 0.5; side[i] = (l[i] - r[i]) * 0.5 }
                ex.feed(mid: mid, side: side)
            } else {
                ex.feed(mid: Array(chans[0].prefix(n)))
            }
        }

        // Resampled channels can come out a sample apart within a chunk; carry the remainder.
        var carry = [[Float]](repeating: [], count: channels)
        func emit(_ outs: [[Float]], final: Bool) {
            var cs = (0..<channels).map { carry[$0] + outs[$0] }
            let n = final ? (cs.map(\.count).max() ?? 0) : (cs.map(\.count).min() ?? 0)
            if final { for c in 0..<channels where cs[c].count < n { cs[c] += [Float](repeating: 0, count: n - cs[c].count) } }
            push(cs.map { Array($0.prefix(n)) })
            carry = cs.map { Array($0.dropFirst(n)) }
        }

        var skip = file.fileFormat.streamDescription.pointee.mFormatID == kAudioFormatMPEGLayer3
            ? Mp3Gapless.extraStartTrim(url) : 0
        while file.framePosition < file.length {
            // reading at the end of the file fails rather than returning 0 frames, so check first
            try file.read(into: buf)
            let n = Int(buf.frameLength)
            if n == 0 { break }
            let data = buf.floatChannelData!
            let drop = min(skip, n)
            skip -= drop
            if drop == n { continue }
            let ins = (0..<channels).map { Array(UnsafeBufferPointer(start: data[$0] + drop, count: n - drop)) }
            emit((0..<channels).map { c in rs[c].map { $0.process(ins[c]) } ?? ins[c] }, final: false)
        }
        emit((0..<channels).map { c in rs[c]?.finish() ?? [] }, final: true)
        return ex.finish()
    }

    /// AVAudioConverter does the downmix and the rate change: close, but not ffmpeg's numbers.
    static func framesViaConverter(_ file: AVAudioFile) throws -> Frames {
        let src = file.processingFormat
        let channels: AVAudioChannelCount = src.channelCount >= 2 ? 2 : 1
        guard let dst = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(FrameExtractor.sampleRate),
                                      channels: channels, interleaved: false),
              let converter = AVAudioConverter(from: src, to: dst) else {
            throw AudioDecoderError.cannotConvert(file.url.lastPathComponent)
        }
        converter.downmix = true
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        converter.sampleRateConverterQuality = AVAudioQuality.medium.rawValue
        let chunk: AVAudioFrameCount = 1 << 16
        guard let inBuf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: chunk),
              let outBuf = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: chunk) else {
            throw AudioDecoderError.cannotConvert(file.url.lastPathComponent)
        }
        let ex = FrameExtractor()
        var atEnd = false
        var readError: Error?
        while true {
            var err: NSError?
            let status = converter.convert(to: outBuf, error: &err) { _, inStatus in
                if atEnd || file.framePosition >= file.length { atEnd = true; inStatus.pointee = .endOfStream; return nil }
                do { try file.read(into: inBuf) } catch { readError = error; atEnd = true }
                if inBuf.frameLength == 0 { atEnd = true; inStatus.pointee = .endOfStream; return nil }
                inStatus.pointee = .haveData
                return inBuf
            }
            if let err { throw err }
            if let readError { throw readError }
            let n = Int(outBuf.frameLength)
            if n > 0, let data = outBuf.floatChannelData {
                if channels == 2 {
                    let l = data[0], r = data[1]
                    var mid = [Float](repeating: 0, count: n), side = [Float](repeating: 0, count: n)
                    for i in 0..<n { mid[i] = (l[i] + r[i]) * 0.5; side[i] = (l[i] - r[i]) * 0.5 }
                    ex.feed(mid: mid, side: side)
                } else {
                    ex.feed(mid: Array(UnsafeBufferPointer(start: data[0], count: n)))
                }
            }
            if status == .endOfStream || status == .error || (n == 0 && atEnd) { break }
        }
        return ex.finish()
    }
}

