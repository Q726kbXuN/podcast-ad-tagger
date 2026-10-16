// Text embeddings with llama.cpp for podcast.adtagger.Embedder.
#include <jni.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

namespace {

struct Embedder {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    int dim = 0;
    int n_batch = 0;
    int n_seq = 0;
};

void fail(JNIEnv * env, const std::string & msg) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), msg.c_str());
}

void quiet_log(ggml_log_level level, const char * text, void *) {
    (void) level; (void) text;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_podcast_adtagger_LlamaJni_load(JNIEnv * env, jclass, jstring jpath, jint threads) {
    static bool started = false;
    if (!started) {
        llama_log_set(quiet_log, nullptr);
        llama_backend_init();
        started = true;
    }
    const char * path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!model) {
        fail(env, "could not load the embedding model");
        return 0;
    }
    llama_context_params cp = llama_context_default_params();
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_LAST;
    cp.n_ctx = 8192;
    cp.n_batch = 8192;
    cp.n_ubatch = 8192;
    cp.n_seq_max = 64;
    cp.kv_unified = true;          // the sequences share the context instead of 128 tokens each
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        fail(env, "could not create the embedding context");
        return 0;
    }
    auto * e = new Embedder();
    e->model = model;
    e->ctx = ctx;
    e->dim = llama_model_n_embd(model);
    e->n_batch = (int) cp.n_batch;
    e->n_seq = (int) cp.n_seq_max;
    return reinterpret_cast<jlong>(e);
}

extern "C" JNIEXPORT jint JNICALL
Java_podcast_adtagger_LlamaJni_dim(JNIEnv *, jclass, jlong h) {
    return reinterpret_cast<Embedder *>(h)->dim;
}

// texts -> float[texts.length * dim], each row unit length.  Texts are UTF-8 byte arrays.
extern "C" JNIEXPORT jfloatArray JNICALL
Java_podcast_adtagger_LlamaJni_embed(JNIEnv * env, jclass, jlong h, jobjectArray texts) {
    auto * e = reinterpret_cast<Embedder *>(h);
    const int n = env->GetArrayLength(texts);
    const llama_vocab * vocab = llama_model_get_vocab(e->model);

    std::vector<std::vector<llama_token>> toks(n);
    for (int i = 0; i < n; i++) {
        auto bytes = (jbyteArray) env->GetObjectArrayElement(texts, i);
        const int len = env->GetArrayLength(bytes);
        std::string s(len, '\0');
        env->GetByteArrayRegion(bytes, 0, len, reinterpret_cast<jbyte *>(&s[0]));
        env->DeleteLocalRef(bytes);
        std::vector<llama_token> t(len + 8);
        int k = llama_tokenize(vocab, s.c_str(), len, t.data(), (int) t.size(), true, false);
        if (k < 0) {
            t.resize(-k);
            k = llama_tokenize(vocab, s.c_str(), len, t.data(), (int) t.size(), true, false);
        }
        t.resize(k > 0 ? std::min(k, e->n_batch) : 0);
        toks[i] = std::move(t);
    }

    std::vector<float> out((size_t) n * e->dim, 0.0f);
    llama_batch batch = llama_batch_init(e->n_batch, 0, 1);
    int i = 0;
    while (i < n) {
        llama_memory_clear(llama_get_memory(e->ctx), true);
        batch.n_tokens = 0;
        const int first = i;
        int seq = 0;
        while (i < n && seq < e->n_seq && batch.n_tokens + (int) toks[i].size() <= e->n_batch) {
            for (size_t p = 0; p < toks[i].size(); p++) {
                const int b = batch.n_tokens++;
                batch.token[b] = toks[i][p];
                batch.pos[b] = (llama_pos) p;
                batch.n_seq_id[b] = 1;
                batch.seq_id[b][0] = seq;
                batch.logits[b] = true;
            }
            i++;
            seq++;
        }
        if (seq == 0) {      // empty text: leave a zero row
            i++;
            continue;
        }
        if (llama_decode(e->ctx, batch) != 0) {
            llama_batch_free(batch);
            fail(env, "llama_decode failed");
            return nullptr;
        }
        for (int s = 0; s < seq; s++) {
            const float * v = llama_get_embeddings_seq(e->ctx, s);
            if (!v || toks[first + s].empty()) continue;
            double norm = 0;
            for (int d = 0; d < e->dim; d++) norm += (double) v[d] * v[d];
            const float scale = norm > 0 ? (float) (1.0 / std::sqrt(norm)) : 0.0f;
            float * row = &out[(size_t) (first + s) * e->dim];
            for (int d = 0; d < e->dim; d++) row[d] = v[d] * scale;
        }
    }
    llama_batch_free(batch);

    jfloatArray result = env->NewFloatArray((jsize) out.size());
    env->SetFloatArrayRegion(result, 0, (jsize) out.size(), out.data());
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_podcast_adtagger_LlamaJni_free(JNIEnv *, jclass, jlong h) {
    auto * e = reinterpret_cast<Embedder *>(h);
    if (!e) return;
    llama_free(e->ctx);
    llama_model_free(e->model);
    delete e;
}
