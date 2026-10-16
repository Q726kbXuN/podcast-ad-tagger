import Foundation
import OnnxRuntimeBindings

/// The tagger model (ad_tagger.onnx) on ONNX Runtime.
public final class Tagger {
    private let env: ORTEnv
    private let session: ORTSession

    public init(modelURL: URL) throws {
        env = try ORTEnv(loggingLevel: .warning)
        session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: nil)
    }

    private static func tensor(_ rows: [[Float]], width: Int) throws -> ORTValue {
        var flat = [Float]()
        flat.reserveCapacity(rows.count * width)
        for r in rows { flat.append(contentsOf: r) }
        let data = flat.withUnsafeBufferPointer { NSMutableData(bytes: $0.baseAddress, length: $0.count * 4) }
        return try ORTValue(tensorData: data, elementType: .float,
                            shape: [1, NSNumber(value: rows.count), NSNumber(value: width)])
    }

    /// embeddings: one 1024-d unit vector per segment; features: 28 raw features per segment.
    /// Returns per-segment probabilities in AdClass.allCases order.
    public func probs(embeddings: [[Float]], features: [[Float]]) throws -> [[Float]] {
        let n = embeddings.count
        if n == 0 { return [] }
        let out = try session.run(withInputs: ["emb": try Self.tensor(embeddings, width: 1024),
                                               "feats": try Self.tensor(features, width: SegmentFeatures.names.count)],
                                  outputNames: ["probs"], runOptions: nil)
        let data = try out["probs"]!.tensorData() as Data
        let k = AdClass.allCases.count
        let flat: [Float] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        return (0..<n).map { Array(flat[($0 * k)..<($0 * k + k)]) }
    }
}
