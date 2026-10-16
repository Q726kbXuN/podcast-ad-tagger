// Speech to text with whisper.cpp for podcast.adtagger.Transcriber: words with times.
#include <jni.h>

#include <string>
#include <vector>

#include "whisper.h"

namespace {

void fail(JNIEnv * env, const std::string & msg) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), msg.c_str());
}

void quiet_log(ggml_log_level level, const char * text, void *) {
    (void) level; (void) text;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_podcast_adtagger_WhisperJni_load(JNIEnv * env, jclass, jstring jpath) {
    whisper_log_set(quiet_log, nullptr);
    const char * path = env->GetStringUTFChars(jpath, nullptr);
    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;
    whisper_context * ctx = whisper_init_from_file_with_params(path, cp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!ctx) {
        fail(env, "could not load the whisper model");
        return 0;
    }
    return reinterpret_cast<jlong>(ctx);
}

// The audio buffer (PcmBuffer): 16 kHz mono float32 in native memory, off the Java heap.
extern "C" JNIEXPORT jlong JNICALL
Java_podcast_adtagger_WhisperJni_pcmNew(JNIEnv *, jclass, jint capacity) {
    auto * v = new std::vector<float>();
    v->reserve(capacity > 0 ? (size_t) capacity : 0);
    return reinterpret_cast<jlong>(v);
}

extern "C" JNIEXPORT void JNICALL
Java_podcast_adtagger_WhisperJni_pcmAppend(JNIEnv * env, jclass, jlong pcm, jfloatArray x, jint offset, jint count) {
    auto * v = reinterpret_cast<std::vector<float> *>(pcm);
    const size_t at = v->size();
    v->resize(at + (size_t) count);
    env->GetFloatArrayRegion(x, offset, count, v->data() + at);
}

extern "C" JNIEXPORT void JNICALL
Java_podcast_adtagger_WhisperJni_pcmFree(JNIEnv *, jclass, jlong pcm) {
    delete reinterpret_cast<std::vector<float> *>(pcm);
}

// Returns Object[] { float[] (segment, start, end per word, seconds), byte[][] (each word's UTF-8
// text, leading space kept) }.
extern "C" JNIEXPORT jobjectArray JNICALL
Java_podcast_adtagger_WhisperJni_transcribe(JNIEnv * env, jclass, jlong h, jlong pcm,
                                            jint threads, jstring jlang, jint beam_size) {
    auto * ctx = reinterpret_cast<whisper_context *>(h);
    auto * audio = reinterpret_cast<std::vector<float> *>(pcm);
    const float * samples = audio->data();
    const int n_samples = (int) audio->size();
    const char * lang = env->GetStringUTFChars(jlang, nullptr);
    whisper_full_params p = whisper_full_default_params(
        beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
    p.beam_search.beam_size = beam_size;
    p.n_threads = threads;
    p.language = lang;
    p.translate = false;
    p.no_context = true;           // like condition_on_previous_text=False: fewer runaway repeats
    p.token_timestamps = true;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_timestamps = false;
    p.print_special = false;
    const int rc = whisper_full(ctx, p, samples, n_samples);
    env->ReleaseStringUTFChars(jlang, lang);
    if (rc != 0) {
        fail(env, "whisper_full failed");
        return nullptr;
    }

    const whisper_token eot = whisper_token_eot(ctx);
    std::vector<float> times;
    std::vector<std::string> words;
    const int n_seg = whisper_full_n_segments(ctx);
    for (int s = 0; s < n_seg; s++) {
        std::string cur;
        float t0 = 0, t1 = 0;
        const int n_tok = whisper_full_n_tokens(ctx, s);
        for (int t = 0; t < n_tok; t++) {
            const whisper_token_data d = whisper_full_get_token_data(ctx, s, t);
            if (d.id >= eot) continue;                       // timestamps and other special tokens
            const char * text = whisper_full_get_token_text(ctx, s, t);
            if (!text || !*text) continue;
            if (text[0] == ' ' || cur.empty()) {
                if (!cur.empty()) {
                    times.insert(times.end(), {(float) s, t0, t1});
                    words.push_back(cur);
                }
                cur = text;
                t0 = d.t0 / 100.0f;
            } else {
                cur += text;
            }
            t1 = d.t1 / 100.0f;
        }
        if (!cur.empty()) {
            times.insert(times.end(), {(float) s, t0, t1});
            words.push_back(cur);
        }
    }

    jfloatArray jtimes = env->NewFloatArray((jsize) times.size());
    env->SetFloatArrayRegion(jtimes, 0, (jsize) times.size(), times.data());
    jclass bytes_class = env->FindClass("[B");
    jobjectArray jwords = env->NewObjectArray((jsize) words.size(), bytes_class, nullptr);
    for (size_t i = 0; i < words.size(); i++) {
        jbyteArray b = env->NewByteArray((jsize) words[i].size());
        env->SetByteArrayRegion(b, 0, (jsize) words[i].size(), reinterpret_cast<const jbyte *>(words[i].data()));
        env->SetObjectArrayElement(jwords, (jsize) i, b);
        env->DeleteLocalRef(b);
    }
    jobjectArray result = env->NewObjectArray(2, env->FindClass("java/lang/Object"), nullptr);
    env->SetObjectArrayElement(result, 0, jtimes);
    env->SetObjectArrayElement(result, 1, jwords);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_podcast_adtagger_WhisperJni_free(JNIEnv *, jclass, jlong h) {
    whisper_free(reinterpret_cast<whisper_context *>(h));
}
