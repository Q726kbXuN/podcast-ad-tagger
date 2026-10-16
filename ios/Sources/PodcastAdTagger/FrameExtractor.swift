import Accelerate
import Foundation

/// Per-100 ms measurements for a whole episode (docs/FEATURES.md, "Frames").
public struct Frames: Sendable {
    public static let names = ["lk_db", "rms_db", "rms10max_db", "rms10min_db", "zcr", "centroid_hz",
                               "flatness_db", "flux_db", "f0_hz", "cpp_db", "side_db"]
    /// columns[c][i]: column c (in `names` order) of frame i
    public var columns: [[Float]]
    /// bands[i]: the 18 band levels (dB) of frame i
    public var bands: [[Float]]
    /// the source had two or more channels
    public var stereo: Bool

    public var count: Int { bands.count }
    public func column(_ name: String) -> [Float] { columns[Frames.names.firstIndex(of: name)!] }
}

/// Streaming analyser: feed 24 kHz mid (and side) samples in any chunk sizes, then call finish().
public final class FrameExtractor {
    public static let sampleRate = 24_000
    static let hop = 2400, sub = 240, nfft = 2048, shop = 1200
    static let padLeft = nfft / 2, padRight = nfft
    static let dbFloor = -200.0
    static let bandEdges: [Float] = [0, 60, 120, 250, 400, 630, 800, 1000, 1250, 1600,
                                     2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12000]
    static var nBands: Int { bandEdges.count - 1 }

    private var buf: [Float]
    private var bufSide: [Float]
    private var haveSide = false
    private var biquad: KWeighting
    private let fft: RealFFT
    private let window: [Float]
    private let binNorm: [Float]
    private let freqs: [Float]
    private let bandOf: [Int]          // band index per bin, -1 for none
    private let qLo: Int, qHi: Int
    private var prevMag: [Float]? = nil
    private var columns = [[Float]](repeating: [], count: Frames.names.count)
    private var bands: [[Float]] = []

    public init() {
        let n = Self.nfft
        buf = [Float](repeating: 0, count: Self.padLeft)
        bufSide = [Float](repeating: 0, count: Self.padLeft)
        biquad = KWeighting(sampleRate: Double(Self.sampleRate))
        fft = RealFFT(count: n)
        // numpy.hanning: symmetric
        window = (0..<n).map { Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double($0) / Double(n - 1))) }
        let meanSq = window.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(n)
        let scale = 1.0 / (Double(n) * Double(n) * meanSq)
        var norm = [Float](repeating: Float(2.0 * scale), count: n / 2 + 1)
        norm[0] = Float(scale)
        norm[n / 2] = Float(scale)
        binNorm = norm
        freqs = (0...(n / 2)).map { Float($0) * Float(Self.sampleRate) / Float(n) }
        bandOf = freqs.map { f in
            for i in 0..<(Self.bandEdges.count - 1) where f >= Self.bandEdges[i] && f < Self.bandEdges[i + 1] { return i }
            return -1
        }
        qLo = max(1, Int(Double(Self.sampleRate) / 400.0))
        qHi = min(n / 2, Int(Double(Self.sampleRate) / 60.0) + 1)
    }

    /// mid = (L+R)/2, side = (L-R)/2; pass side = nil for mono sources.
    public func feed(mid: [Float], side: [Float]? = nil) {
        buf.append(contentsOf: mid)
        if let side {
            haveSide = true
            bufSide.append(contentsOf: side)
        }
        drain()
    }

    public func finish() -> Frames {
        let pad = [Float](repeating: 0, count: Self.padRight + Self.hop)
        buf.append(contentsOf: pad)
        if haveSide { bufSide.append(contentsOf: pad) }
        drain()
        return Frames(columns: columns, bands: bands, stereo: haveSide)
    }

    private func drain() {
        let nf = (buf.count - Self.padLeft - Self.padRight) / Self.hop
        guard nf > 0 else { return }
        for j in 0..<nf { emit(frame: j) }
        buf.removeFirst(nf * Self.hop)
        if haveSide { bufSide.removeFirst(nf * Self.hop) }
    }

    private static func db10(_ p: Double) -> Float {
        Float(p > 0 ? max(10.0 * log10(p), dbFloor) : dbFloor)
    }

    private func emit(frame j: Int) {
        let hop = Self.hop, n = Self.nfft
        let core = Self.padLeft + j * hop
        // loudness (K-weighted), level, 10 ms extremes, zero crossings
        var lk = 0.0, rms = 0.0
        var subMax = -Double.infinity, subMin = Double.infinity
        var zc = 0
        for s in 0..<(hop / Self.sub) {
            var acc = 0.0
            for i in 0..<Self.sub {
                let x = buf[core + s * Self.sub + i]
                let y = biquad.process(Double(x))
                lk += y * y
                acc += Double(x) * Double(x)
            }
            rms += acc
            let m = acc / Double(Self.sub)
            subMax = max(subMax, m)
            subMin = min(subMin, m)
        }
        for i in 1..<hop where buf[core + i].sign != buf[core + i - 1].sign { zc += 1 }
        var sideDb = Float(Self.dbFloor)
        if haveSide {
            var acc = 0.0
            for i in 0..<hop { let x = Double(bufSide[core + i]); acc += x * x }
            sideDb = Self.db10(acc / Double(hop))
        }

        // two spectra per frame, centred 600 and 1800 samples into it
        var centroid = 0.0, flatness = 0.0, flux = 0.0
        var best = (cpp: -Float.infinity, f0: Float(0))
        var bandPow = [Double](repeating: 0, count: Self.nBands)
        let nb = n / 2 + 1
        for k in 0..<2 {
            let start = (2 * j + k) * Self.shop + Self.shop / 2
            var block = [Float](repeating: 0, count: n)
            for i in 0..<n { block[i] = buf[start + i] * window[i] }
            let (re, im) = fft.forward(block)
            var power = [Float](repeating: 0, count: nb)
            var total = 0.0, cen = 0.0, logSum = 0.0
            for b in 0..<nb {
                let p = (re[b] * re[b] + im[b] * im[b]) * binNorm[b] + 1e-20
                power[b] = p
                total += Double(p)
                cen += Double(p) * Double(freqs[b])
                logSum += Double(log(p))
                if bandOf[b] >= 0 { bandPow[bandOf[b]] += Double(p) }
            }
            centroid += cen / total
            flatness += exp(logSum / Double(nb)) / (total / Double(nb))
            let mag = power.map { $0.squareRoot() }
            if let prev = prevMag {
                var f = 0.0
                for b in 0..<nb { f += Double(max(mag[b] - prev[b], 0)) }
                flux += f
            }
            prevMag = mag
            // cepstrum: inverse real FFT of the log power spectrum
            let cep = fft.inverse(re: power.map { log($0) }, im: [Float](repeating: 0, count: nb))
            let win = Array(cep[qLo..<qHi])
            var q = 0
            for i in 1..<win.count where win[i] > win[q] { q = i }
            let cpp = (win[q] - median(win)) * Float(10.0 / log(10.0))
            if cpp > best.cpp { best = (cpp, Float(Self.sampleRate) / Float(q + qLo)) }
        }
        let vals: [Float] = [
            Self.db10(lk / Double(hop)), Self.db10(rms / Double(hop)), Self.db10(subMax), Self.db10(subMin),
            Float(zc) / Float(hop - 1), Float(centroid / 2), Self.db10(flatness / 2),
            Self.db10((flux / 2) * (flux / 2)), best.f0, best.cpp, sideDb,
        ]
        for c in 0..<vals.count { columns[c].append(vals[c]) }
        bands.append(bandPow.map { Self.db10($0 / 2) })
    }
}

/// ITU-R BS.1770 K-weighting: high shelf then high pass, as two biquads (direct form II transposed).
struct KWeighting {
    private var sections: [(b: [Double], a: [Double])]
    private var z: [[Double]]

    init(sampleRate sr: Double) {
        let f0 = 1681.974450955533, gainDb = 3.999843853973347, q = 0.7071752369554196
        let k = tan(Double.pi * f0 / sr)
        let vh = pow(10.0, gainDb / 20.0)
        let vb = pow(vh, 0.4996667741545416)
        let a0 = 1.0 + k / q + k * k
        let shelfB = [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
        let shelfA = [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
        let f0b = 38.13547087602444, qb = 0.5003270373238773
        let kb = tan(Double.pi * f0b / sr)
        let a0b = 1.0 + kb / qb + kb * kb
        let hpB = [1.0, -2.0, 1.0]
        let hpA = [1.0, 2.0 * (kb * kb - 1.0) / a0b, (1.0 - kb / qb + kb * kb) / a0b]
        // the coefficients are stored as float32 in the reference implementation
        let f: ([Double]) -> [Double] = { $0.map { Double(Float($0)) } }
        sections = [(f(shelfB), f(shelfA)), (f(hpB), f(hpA))]
        z = [[0, 0], [0, 0]]
    }

    mutating func process(_ x: Double) -> Double {
        var v = x
        for s in 0..<2 {
            let (b, a) = sections[s]
            let y = b[0] * v + z[s][0]
            z[s][0] = b[1] * v - a[1] * y + z[s][1]
            z[s][1] = b[2] * v - a[2] * y
            v = y
        }
        return v
    }
}

/// numpy-compatible real FFT of a power-of-two length, on vDSP.
final class RealFFT {
    let n: Int
    private let fwd: vDSP_DFT_Setup
    private let inv: vDSP_DFT_Setup

    init(count: Int) {
        n = count
        fwd = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(count), .FORWARD)!
        inv = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(count), .INVERSE)!
    }

    deinit {
        vDSP_DFT_DestroySetup(fwd)
        vDSP_DFT_DestroySetup(inv)
    }

    /// numpy.fft.rfft: n/2+1 bins, unscaled.
    func forward(_ x: [Float]) -> (re: [Float], im: [Float]) {
        let h = n / 2
        var ir = [Float](repeating: 0, count: h), ii = [Float](repeating: 0, count: h)
        for i in 0..<h { ir[i] = x[2 * i]; ii[i] = x[2 * i + 1] }
        var or = [Float](repeating: 0, count: h), oi = [Float](repeating: 0, count: h)
        vDSP_DFT_Execute(fwd, ir, ii, &or, &oi)
        // packed output, scaled by 2: or[0] = DC, oi[0] = Nyquist
        var re = [Float](repeating: 0, count: h + 1), im = [Float](repeating: 0, count: h + 1)
        re[0] = or[0] / 2
        re[h] = oi[0] / 2
        for k in 1..<h { re[k] = or[k] / 2; im[k] = oi[k] / 2 }
        return (re, im)
    }

    /// numpy.fft.irfft: real signal of length n from n/2+1 bins (imaginary parts of DC and
    /// Nyquist ignored), scaled by 1/n.
    func inverse(re: [Float], im: [Float]) -> [Float] {
        let h = n / 2
        var ir = [Float](repeating: 0, count: h), ii = [Float](repeating: 0, count: h)
        ir[0] = re[0]
        ii[0] = re[h]
        for k in 1..<h { ir[k] = re[k]; ii[k] = im[k] }
        var or = [Float](repeating: 0, count: h), oi = [Float](repeating: 0, count: h)
        vDSP_DFT_Execute(inv, ir, ii, &or, &oi)
        var out = [Float](repeating: 0, count: n)
        let s = 1.0 / Float(n)
        for i in 0..<h { out[2 * i] = or[i] * s; out[2 * i + 1] = oi[i] * s }
        return out
    }
}
