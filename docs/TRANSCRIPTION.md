# Transcription

The tagger doesn't care which speech recognizer you use. It cares about what the segments look like.

## What the segments need to be

- **Sentence sized.** One segment per sentence or phrase, usually a few seconds and around ten words. The model was trained on segments averaging about 3 seconds. A recognizer that hands back 30-second chunks needs them cut at sentence ends first (see below). Way too short (single words) is just as bad.
- **Time stamped.** `start` and `end` in seconds for each segment, accurate to a few tenths of a second. The ad spans that come out start and end on segment boundaries, so this is the precision you get.
- **In order, covering the whole episode.** Gaps for silence and music are fine, but don't drop parts of the episode. The model looks at the episode as a sequence, and an ad that never got transcribed can't be found.
- **A word count**, and the text itself, of course.
- **Speaker labels are optional.** If your recognizer does diarization, pass a speaker id per segment. Without them that feature is just zero, and it barely matters.

English only. The model hasn't seen anything else.

## Cutting into sentences

If the recognizer gives word timestamps, split each of its segments after any word ending in `.`, `?` or `!` (allowing for closing quotes and brackets), skipping common abbreviations like "Mr." and "U.S.". If a piece still runs past 15 seconds without punctuation, cut it at its longest pause. `transcribe.split_words` in the Python package does exactly this, and the Swift and Kotlin samples have ports of it. Don't merge across the recognizer's own segment boundaries.

If there are no word timestamps, the recognizer's phrases are usually fine as long as they're in the right size range.

## What I'd use where

- **Desktop:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with `small.en` and word timestamps, which is what `adstrip` does. It's the slow step: figure on several minutes per hour of audio on a CPU, much less with a GPU. A bigger Whisper model is a little more accurate and a lot slower; for finding ads it doesn't buy much. WhisperX works great too.
- **iPhone (iOS 26 and later):** Apple's `SpeechAnalyzer` with `SpeechTranscriber`, on the device, using its phrases. I compared it against Whisper-quality server transcripts: the ad spans came out practically the same (about half a point of F1 lower). The language assets download on first use.
- **Android:** [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with `base.en` or `small.en`, token timestamps on, then the sentence split. Android's own speech APIs aren't built for transcribing files.
- **Server:** if you already transcribe episodes somewhere, send the segments down and do the rest on the device. Nothing in the pipeline has to run in one place.

The audio features don't come from the transcript, so they're computed from the audio file regardless of how you transcribe it.
