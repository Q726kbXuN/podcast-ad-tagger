import Foundation
import llama

public enum EmbedderError: Error {
    case loadFailed(String)
    case decodeFailed(Int32)
}

/// Qwen3-Embedding-0.6B (Q8_0 GGUF) through llama.cpp: one unit-length 1024-d vector per text.
public final class Embedder {
    public static let ggufURL = URL(string: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf")!
    public static let ggufSHA256 = "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439"
    static let maxTokens = 8192, maxSequences = 64, maxChars = 2000

    private let model: OpaquePointer
    private let ctx: OpaquePointer
    private let vocab: OpaquePointer
    public let dimension: Int

    /// gpuLayers: 99 runs everything on the GPU (Metal); use 0 on the simulator.
    public init(modelURL: URL, gpuLayers: Int32 = 99) throws {
        _ = Embedder.setUp
        var mp = llama_model_default_params()
        mp.n_gpu_layers = gpuLayers
        guard let m = llama_model_load_from_file(modelURL.path, mp) else {
            throw EmbedderError.loadFailed(modelURL.lastPathComponent)
        }
        var cp = llama_context_default_params()
        cp.n_ctx = UInt32(Self.maxTokens)
        cp.n_batch = UInt32(Self.maxTokens)
        cp.n_ubatch = UInt32(Self.maxTokens)
        cp.n_seq_max = UInt32(Self.maxSequences)
        cp.embeddings = true
        cp.pooling_type = LLAMA_POOLING_TYPE_LAST
        cp.kv_unified = true            // sequences share the context instead of 1/64th each
        guard let c = llama_init_from_model(m, cp) else {
            llama_model_free(m)
            throw EmbedderError.loadFailed("context")
        }
        model = m
        ctx = c
        vocab = llama_model_get_vocab(m)
        dimension = Int(llama_model_n_embd(m))
    }

    deinit {
        llama_free(ctx)
        llama_model_free(model)
    }

    private static let setUp: Void = {
        // only errors reach stderr; llama.cpp is chatty otherwise
        let quiet: ggml_log_callback = { level, text, _ in
            if level == GGML_LOG_LEVEL_ERROR, let text { fputs(text, stderr) }
        }
        ggml_log_set(quiet, nil)
        llama_log_set(quiet, nil)
        llama_backend_init()
    }()

    /// The exact string the tagger was trained on: trimmed, first 2000 characters, " " if empty.
    public static func input(_ text: String) -> String {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let s = String(String.UnicodeScalarView(t.unicodeScalars.prefix(maxChars)))
        return s.isEmpty ? " " : s
    }

    private func tokenize(_ text: String) -> [llama_token] {
        let utf8 = Array(text.utf8CString)
        var tokens = [llama_token](repeating: 0, count: utf8.count + 8)
        var n = llama_tokenize(vocab, utf8, Int32(utf8.count - 1), &tokens, Int32(tokens.count), true, false)
        if n < 0 {
            tokens = [llama_token](repeating: 0, count: Int(-n))
            n = llama_tokenize(vocab, utf8, Int32(utf8.count - 1), &tokens, Int32(tokens.count), true, false)
        }
        return Array(tokens.prefix(Int(max(n, 0))).prefix(Self.maxTokens))
    }

    public func embed(_ texts: [String]) throws -> [[Float]] {
        let toks = texts.map { tokenize(Self.input($0)) }
        var out = [[Float]](repeating: [], count: texts.count)
        var i = 0
        while i < toks.count {
            var group = [Int](), total = 0
            while i < toks.count && group.count < Self.maxSequences && total + toks[i].count <= Self.maxTokens {
                group.append(i); total += toks[i].count; i += 1
            }
            var batch = llama_batch_init(Int32(total), 0, 1)
            defer { llama_batch_free(batch) }
            var k = 0
            for (seq, idx) in group.enumerated() {
                for (pos, t) in toks[idx].enumerated() {
                    batch.token[k] = t
                    batch.pos[k] = Int32(pos)
                    batch.n_seq_id[k] = 1
                    batch.seq_id[k]![0] = Int32(seq)
                    batch.logits[k] = 1
                    k += 1
                }
            }
            batch.n_tokens = Int32(k)
            llama_memory_clear(llama_get_memory(ctx), true)
            let rc = llama_decode(ctx, batch)
            if rc != 0 { throw EmbedderError.decodeFailed(rc) }
            for (seq, idx) in group.enumerated() {
                let p = llama_get_embeddings_seq(ctx, Int32(seq))!
                var v = Array(UnsafeBufferPointer(start: p, count: dimension))
                let norm = v.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
                if norm > 0 { v = v.map { Float(Double($0) / norm) } }
                out[idx] = v
            }
        }
        return out
    }
}
