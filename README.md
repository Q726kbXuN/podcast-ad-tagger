# Podcast Ad Tagger

A small model that finds the ads in podcast episodes: sponsor reads, a show plugging its own stuff, the "become a member" pitches, and the recurring boilerplate. It runs locally, on a laptop or a phone, so a podcast player can skip ads without sending anything anywhere. There's also a little command line tool that cuts them out of an MP3, and a proof of concept player that does all of it in a web page.

```
$ adstrip episode.mp3
Measuring the audio ...
Transcribing with whisper small.en (the slow part) ...
Embedding 755 segments ...

5 span(s) found in 42:19:
  cut      0:01 -     1:06  sponsor_ad       p=1.00
  keep     6:11 -     6:23  boilerplate      p=0.90
  cut     21:39 -    23:42  sponsor_ad       p=1.00
  keep    38:45 -    41:42  boilerplate      p=0.84
  cut     41:52 -    42:18  sponsor_ad       p=1.00
Wrote episode.noads.mp3, 3:34 shorter.
```

## How it works, roughly

The episode gets transcribed and cut into sentences. Each sentence gets a text embedding and a couple dozen measurements of how the audio sounds around it, since ads tend to be mastered louder, carry music beds, and get spliced in with hard edits. A small network (12 MB) reads the whole episode's worth of those and says, for each sentence, which of five things it is: content, sponsor ad, house promo, membership pitch, or boilerplate. Runs of the same answer become spans.

The text embeddings come from [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), which is the big piece at about 640 MB. Everything else is small.

## What's here

- `model/`: the tagger as ONNX, plus `ad_tagger.json` with the inputs, outputs, classes, the exact embedding model and how to call it, and the decoding settings
- `python/`: the reference implementation, and `adstrip`, the desktop tool
- `ios/`: a Swift package for iPhone (and the Mac)
- `android/`: a Kotlin library module for Android
- `web/`: a proof of concept podcast player that does all of it in a browser, no backend, a demo is [here](https://q726kbxun.github.io/ad_skipper/web/index.html)
- `docs/FEATURES.md`: exactly what the model reads, for porting it somewhere new
- `docs/TRANSCRIPTION.md`: what the transcript needs to look like, and what recognizer to use where
- `testdata/`: reference inputs and outputs for checking a port

## Cutting ads out of an MP3

You'll need Python 3.10 or newer, and ffmpeg on your path. From the repo:

```
python -m venv .venv
source .venv/bin/activate
pip install -e python
adstrip episode.mp3
```

The first run downloads the embedding model into `~/.cache/podcast-ad-tagger` and a Whisper model for transcription. By default it only cuts sponsor reads; `--strip sponsor,membership,promo,boilerplate` cuts more. `--find-only` just lists what it found, and `--json spans.json` saves everything, including the "unsure" stretches (more on those below). Run `adstrip --help` for the rest.

That episode took about 5 minutes on a 20-core ARM box with no GPU, nearly all of it transcription, which is also the part a GPU speeds up (`--whisper-device cuda`). The cuts get a 15 ms fade on each side so they don't click.

## Using it in an app

The pipeline is the same everywhere:

1. Transcribe the episode into sentence-sized, time-stamped segments ([docs/TRANSCRIPTION.md](docs/TRANSCRIPTION.md)).
2. Measure the audio in 100 ms frames, then work out the 28 numbers per segment ([docs/FEATURES.md](docs/FEATURES.md)).
3. Embed each segment's text.
4. Run `model/ad_tagger.onnx` once over the whole episode: `emb` `[1, N, 1024]` and `feats` `[1, N, 28]` in, `probs` `[1, N, 5]` out.
5. Turn that into spans. Take the most likely class for each segment, bridge gaps of up to two content segments inside a run, and drop runs shorter than two segments. A span starts at its first segment's start and ends at its last segment's end.

The iOS, Android and web samples do all of that; [ios/README.md](ios/README.md), [android/README.md](android/README.md) and [web/README.md](web/README.md) cover using them. For anything else, the Python is short enough to port, and `testdata/` will tell you when you've got it right.

Two things I found useful in a player:

- Each span has `p`, the model's average probability for it, handy for deciding how quietly to skip.
- `candidates` are stretches outside any span where the sponsor probability stays at 0.25 or more for at least two segments: the model's near misses. Rather than skipping those, I show them as "maybe an ad" and let the listener decide with one tap.

## How well it works

I checked it against 240 episodes that people labeled by hand, none of which (or anything close to them) were used for training. For sponsor reads, it gets about 95% of the seconds right (F1 0.95), misses about 4 seconds of sponsor reads per hour, and flags 5 or 6 seconds per hour that aren't ads. On shows it never saw at all the score holds up, though it flags more that isn't an ad, around 12 seconds an hour. I didn't hold the other three types to the same standard, so they're rougher, which is why `adstrip` only cuts sponsor reads unless you ask.

## Mini FAQ

* Why not just ask an LLM?

You can, and they're good at it. But that's a paid API call per episode and the transcript leaves your device. This runs offline, for free, on a phone.

* Does it catch ads the host reads themselves?

Yes. It mostly goes by what's said, so host reads are its bread and butter, and inserted spots get extra help from the audio side.

* Other languages?

No, English only.

* Isn't skipping ads unfair to the people making the podcasts?

Ads pay for a lot of podcasts, and the people making them deserve to get paid. I'm not pretending otherwise. But listeners have been skipping ads since podcast players got a 30-second skip button. This just makes that a little less tedious.

If you like a show, support it directly if you can. Subscribe, donate, buy their merchandise. But ultimately, what you choose to listen to on your own device should be up to you.

## License

MIT, for the code and the model. The embedding model is Qwen's, under Apache 2.0, and gets downloaded separately. Whisper models are MIT.
