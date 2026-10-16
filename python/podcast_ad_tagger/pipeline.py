"""Audio file in, ad spans out."""
from . import audio, features
from .tagger import Tagger, find_spans, find_candidates


class Detector:
    """Loads the models once; detect() can then run on any number of files."""

    def __init__(self, embed_model=None, whisper_model="small.en", onnx_path=None, whisper_device="auto"):
        from .embed import Embedder
        self.embedder = Embedder(embed_model)
        self.tagger = Tagger(onnx_path)
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device

    def detect(self, path, segments=None, log=print):
        """segments: optional transcript as [{start, end, text, n_words?, speaker?}], already
        cut into sentences; when omitted the file is transcribed here."""
        log("Measuring the audio ...")
        frames = audio.analyze_file(path)
        if segments is None:
            from .transcribe import transcribe
            log(f"Transcribing with whisper {self.whisper_model} (the slow part) ...")
            segments = transcribe(path, model=self.whisper_model, device=self.whisper_device)
        if not segments:
            return {"duration": len(frames) / 10.0, "segments": 0, "spans": [], "candidates": []}
        log(f"Embedding {len(segments)} segments ...")
        emb = self.embedder.embed([s["text"] for s in segments])
        probs = self.tagger.probs(emb, features.segment_features(frames, segments))
        spans = find_spans(probs, segments)
        return {"duration": round(len(frames) / 10.0, 1), "segments": len(segments), "spans": spans,
                "candidates": find_candidates(probs, spans, segments)}
