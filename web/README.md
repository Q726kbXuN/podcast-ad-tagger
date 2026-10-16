# Browser player

A proof of concept podcast player that does the whole thing in a web page: pick an MP3, and it transcribes it, finds the ads, and plays it back with the transcript scrolling along and the ads marked. Sponsor reads get skipped (there are checkboxes for the other kinds), and clicking any sentence jumps there. Nothing gets uploaded anywhere; the only network traffic is downloading the models, and the browser keeps those for next time.

No build step and no backend. It's a handful of static files plus three libraries pulled from a CDN at pinned versions:

- [transformers.js](https://github.com/huggingface/transformers.js) runs Whisper `base.en` for the speech to text, on the GPU through WebGPU when the browser has it
- [wllama](https://github.com/ngxson/wllama), llama.cpp compiled to WebAssembly, runs the same `Qwen3-Embedding-0.6B-Q8_0.gguf` the other platforms use, also on the GPU when it can
- [onnxruntime-web](https://onnxruntime.ai/) runs `../model/ad_tagger.onnx`

The audio measurements are a JavaScript port in `lib/`, checked against `../testdata` by `node --test web/test`.

The speech to text doesn't use transformers.js's own long-audio pipeline, which works through overlapping 30 second windows one at a time and leaves a GPU mostly idle. `lib/whisper.js` borrows the trick WhisperX uses instead: cut the audio at pauses into chunks of up to 30 seconds, so nothing gets transcribed twice, run several chunks through Whisper at once, and time each word from the model's attention. That made speech to text about three times faster, and the ad spans came out a little better too.

## Putting it online

It needs to be served over HTTPS (service workers and WebGPU insist on it), with `../model/ad_tagger.onnx`, `../model/ad_tagger.json` and `../testdata/embedding_reference.json` next to it the same way they are in this repo. Any static host works. The reference embeddings are there so the page can check the GPU gets the same answers as the CPU before trusting it with the text model; GPUs don't all do the arithmetic the same way, and if the check fails, or the file is missing, it quietly uses the CPU.

The models run a lot faster with more than one thread, and browsers only allow that on pages that are "cross-origin isolated", which normally means the server sends two extra headers. Plenty of static hosts, S3 included, can't do that, so the page carries [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) (MIT): a tiny service worker that adds the headers itself. The very first visit reloads the page once to switch it on.

For S3, from the root of this repo:

```
aws s3 sync . s3://YOUR-BUCKET/podcast-ad-tagger \
  --exclude '*' --include 'web/*' --include 'model/ad_tagger.onnx' --include 'model/ad_tagger.json' \
  --include 'testdata/embedding_reference.json' \
  --exclude 'web/test/*' --exclude 'web/README.md' \
  --acl public-read --cache-control 'max-age=300' --delete
```

Then open `https://YOUR-BUCKET.s3.YOUR-REGION.amazonaws.com/podcast-ad-tagger/web/index.html`. Use that HTTPS address rather than the bucket's static website endpoint, which is plain HTTP. The bucket needs ACLs enabled (object ownership not set to "bucket owner enforced") and Block Public Access turned off for ACLs, or `--acl public-read` gets refused.

## Things to know

- Speed depends mostly on the GPU. In Chrome on an Apple Silicon Mac, a 29 minute episode took a little over a minute start to finish, and a 42 minute one under two, about half of it speech to text. On a Windows PC with an RTX 5080 the same 29 minute episode spent about 80 seconds on speech to text and a minute on the text model; the browser is the bottleneck there, not the card. Without a usable GPU, on a 20-core ARM box, the 29 minute one took about two minutes.
- It's not light on memory: on the Mac the browser peaked around 7 GB for a half-hour episode and nearer 10 GB for the 42 minute one (GPU memory included, since Macs share it), and around 4 GB for the half-hour one on the CPU-only box.
- The first visit downloads about 900 MB: the text model (640 MB) and Whisper (about 200 MB, depending on whether it runs on the GPU or the CPU).
- The whole decoded episode sits in memory, so very long episodes (two hours plus) can run a tab out of memory. Phones' browsers mostly won't manage it at all.
- Browsers decode MP3s themselves, and not all of them the way the training audio was decoded. I haven't seen it change an ad span, but Safari is the one most likely to differ.
- `window.adtag.analyze(file, {segments})` runs the pipeline from a script, and `segments` lets you hand it a transcript you already have instead of running Whisper (see `../docs/TRANSCRIPTION.md`).
