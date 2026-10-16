import Foundation

// Statistics with numpy's exact semantics, so the features match the reference.

/// numpy.median: the middle value, or the mean of the two middle values.
func median(_ v: [Float]) -> Float {
    if v.isEmpty { return .nan }
    let s = v.sorted()
    let m = s.count / 2
    return s.count % 2 == 1 ? s[m] : Float((Double(s[m - 1]) + Double(s[m])) / 2)
}

/// numpy.percentile with the default linear interpolation.
func percentile(_ v: [Float], _ q: Double) -> Float {
    if v.isEmpty { return .nan }
    let s = v.sorted()
    let pos = q / 100.0 * Double(s.count - 1)
    let lo = Int(pos.rounded(.down)), hi = min(lo + 1, s.count - 1)
    let frac = pos - Double(lo)
    return Float(Double(s[lo]) + (Double(s[hi]) - Double(s[lo])) * frac)
}

func mean(_ v: [Float]) -> Float {
    v.isEmpty ? .nan : Float(v.reduce(0.0) { $0 + Double($1) } / Double(v.count))
}

/// numpy.std (population).
func std(_ v: [Float]) -> Float {
    if v.isEmpty { return .nan }
    let m = v.reduce(0.0) { $0 + Double($1) } / Double(v.count)
    return Float((v.reduce(0.0) { $0 + (Double($1) - m) * (Double($1) - m) } / Double(v.count)).squareRoot())
}
