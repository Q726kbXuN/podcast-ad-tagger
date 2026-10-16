"""adstrip: remove ads from a podcast episode."""
import argparse
import json
import os
import sys

TYPES = {"sponsor": "sponsor_ad", "promo": "house_promo", "membership": "membership_pitch",
         "boilerplate": "boilerplate"}


def hms(t):
    t = int(round(t))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="adstrip", description="Find the ads in a podcast episode and cut them out.")
    ap.add_argument("input", help="audio file (mp3, m4a, anything ffmpeg reads)")
    ap.add_argument("-o", "--output", help="where to write the ad-free mp3 (default: <input>.noads.mp3)")
    ap.add_argument("--strip", default="sponsor",
                    help="comma list of what to cut: sponsor, promo, membership, boilerplate (default: sponsor)")
    ap.add_argument("--json", metavar="FILE", help="also write everything found (spans and unsure candidates) as JSON")
    ap.add_argument("--find-only", action="store_true", help="just list what was found, don't write audio")
    ap.add_argument("--whisper-model", default="small.en", help="faster-whisper model (default: small.en)")
    ap.add_argument("--whisper-device", default="auto", help="cpu, cuda or auto (default: auto)")
    ap.add_argument("--embed-model", help="path to Qwen3-Embedding-0.6B-Q8_0.gguf (downloaded on first use if omitted)")
    args = ap.parse_args(argv)
    try:
        want = {TYPES[t.strip()] for t in args.strip.split(",") if t.strip()}
    except KeyError as e:
        ap.error(f"unknown type {e}; pick from {', '.join(TYPES)}")

    from .pipeline import Detector
    res = Detector(embed_model=args.embed_model, whisper_model=args.whisper_model,
                   whisper_device=args.whisper_device).detect(args.input)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
    cuts = [s for s in res["spans"] if s["type"] in want]
    print(f"\n{len(res['spans'])} span(s) found in {hms(res['duration'])}:")
    for s in res["spans"]:
        mark = "cut " if s in cuts else "keep"
        print(f"  {mark} {hms(s['start']):>8} - {hms(s['end']):>8}  {s['type']:<16} p={s['p']:.2f}")
    for c in res["candidates"]:
        print(f"  ???? {hms(c['start']):>8} - {hms(c['end']):>8}  maybe a sponsor     p={c['p']:.2f}")
    if args.find_only:
        return 0
    if not cuts:
        print("Nothing to cut.")
        return 0
    from .strip import strip
    out = args.output or os.path.splitext(args.input)[0] + ".noads.mp3"
    removed = strip(args.input, out, [(s["start"], s["end"]) for s in cuts])
    print(f"Wrote {out}, {hms(removed)} shorter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
