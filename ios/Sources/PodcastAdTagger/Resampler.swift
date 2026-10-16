import Foundation

/// Resampling the way ffmpeg's default resampler (libswresample) does it, because that's what
/// produced the audio the model was trained on: a Kaiser-windowed sinc (beta 9) cut off at 0.97
/// of the output Nyquist, an exact polyphase filter bank, mirror padding at both ends.  Matches
/// ffmpeg's output to float rounding for the usual rates (44.1, 48, 32, 22.05, 16 kHz ...).
/// Platform resamplers differ from it enough to move the spectral features (docs/FEATURES.md).
final class Resampler {
    let length: Int        // taps per output
    let center: Int
    let phases: Int        // output rate / gcd
    let step: Int          // input rate / gcd
    private let bank: [[Float]]
    private var x: [Float] = []        // padded stream, x[0] is absolute position `base`
    private var base = 0
    private var started = false
    private var head: [Float] = []
    private var last: [Float] = []     // the most recent `length` input samples
    private var total = 0
    private var n = 0                  // next output index

    /// nil when the ratio needs more than 1024 phases (odd rates); use a platform resampler then.
    init?(inputRate: Int, outputRate: Int = FrameExtractor.sampleRate) {
        let factor = min(Double(outputRate) * 0.97 / Double(inputRate), 1.0)
        var len = max(Int((32.0 / factor).rounded(.up)), 1)
        if len > 1 { len += len % 2 }
        var a = outputRate, b = inputRate
        while b != 0 { (a, b) = (b, a % b) }
        let p = outputRate / a, d = inputRate / a
        guard p <= 1024, inputRate != outputRate else { return nil }
        length = len; center = (len - 1) / 2; phases = p; step = d
        // filter bank, as libswresample's build_filter
        func i0(_ v: Double) -> Double {
            var s = 1.0, t = 1.0, k = 1.0
            while true { t *= (v / (2 * k)) * (v / (2 * k)); s += t; if t < 1e-17 * s { return s }; k += 1 }
        }
        let phNb = p % 2 == 1 ? p : p / 2 + 1
        var tabs = [[Double]]()
        var norm = 0.0
        for ph in 0..<phNb {
            var tab = [Double](repeating: 0, count: len)
            for i in 0..<len {
                let xv = Double.pi * (Double(i - center) - Double(ph) / Double(p)) * factor
                var y = xv == 0 ? 1.0 : sin(xv) / xv
                let w = 2.0 * xv / (factor * Double(len) * Double.pi)
                y *= i0(9.0 * max(1 - w * w, 0).squareRoot())
                tab[i] = y
                if ph == 0 { norm += y }
            }
            tabs.append(tab)
        }
        var bank = [[Float]](repeating: [Float](repeating: 0, count: len), count: p)
        for ph in 0..<phNb {
            bank[ph] = tabs[ph].map { Float($0 / norm) }
            if p % 2 == 0 && ph > 0 { bank[p - ph] = bank[ph].reversed() }
        }
        self.bank = bank
    }

    private func start() {
        // mirror the opening: x[L-k] = in[k] for k = 1...L
        let h = head
        var front = [Float](repeating: 0, count: length)
        for k in 1...length where k < h.count { front[length - k] = h[k] }
        x = front + h
        base = 0
        head = []
        started = true
    }

    private func produce() -> [Float] {
        var out = [Float]()
        while true {
            let s = (length - center) + (n * step) / phases
            let rel = s - base
            if rel + length > x.count { break }
            let h = bank[(n * step) % phases]
            var even: Float = 0, odd: Float = 0
            x.withUnsafeBufferPointer { xp in
                var i = 0
                while i + 1 < length {
                    even += xp[rel + i] * h[i]
                    odd += xp[rel + i + 1] * h[i + 1]
                    i += 2
                }
                if i < length { even += xp[rel + i] * h[i] }
            }
            out.append(even + odd)
            n += 1
        }
        // drop what no later output needs
        let need = (length - center) + (n * step) / phases - base
        if need > 4096 { x.removeFirst(need); base += need }
        return out
    }

    func process(_ input: [Float]) -> [Float] {
        total += input.count
        last.append(contentsOf: input)
        if last.count > length { last.removeFirst(last.count - length) }
        if !started {
            head.append(contentsOf: input)
            if head.count < length + 1 { return [] }
            start()
        } else {
            x.append(contentsOf: input)
        }
        return produce()
    }

    func finish() -> [Float] {
        if !started { start() }
        let r = (min(total, length) + 1) / 2
        x.append(contentsOf: last.reversed().prefix(r))
        return produce()
    }
}
