"""Text embeddings: Qwen3-Embedding-0.6B (Q8_0 GGUF) through llama.cpp."""
import os
import urllib.request

import numpy as np

GGUF_URL = "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf"
GGUF_SHA256 = "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439"
MAX_CHARS = 2000


def embed_input(text):
    """The exact string the model was trained on: the segment's text, trimmed, capped."""
    return (text or "").strip()[:MAX_CHARS] or " "


def default_model_path():
    return os.path.join(os.path.expanduser("~/.cache/podcast-ad-tagger"), os.path.basename(GGUF_URL))


def fetch_model(path=None):
    """Download the embedder once (about 640 MB) and return its path."""
    path = path or default_model_path()
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading the embedding model to {path} ...", flush=True)
        urllib.request.urlretrieve(GGUF_URL, path + ".part")
        os.replace(path + ".part", path)
    return path


class Embedder:
    def __init__(self, model_path=None, threads=None, gpu_layers=0):
        from llama_cpp import Llama, LLAMA_POOLING_TYPE_LAST
        self.llm = Llama(model_path=model_path or fetch_model(), embedding=True,
                         pooling_type=LLAMA_POOLING_TYPE_LAST, n_ctx=8192, n_batch=8192,
                         n_ubatch=8192, n_seq_max=64, n_threads=threads,
                         n_gpu_layers=gpu_layers, verbose=False)

    def embed(self, texts, batch=64):
        """-> float32 [len(texts), 1024], unit length."""
        out = []
        for i in range(0, len(texts), batch):
            out.extend(self.llm.embed([embed_input(t) for t in texts[i:i + batch]], normalize=True))
        return np.asarray(out, np.float32).reshape(len(texts), -1)
