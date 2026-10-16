"""Cut spans out of an audio file with ffmpeg."""
import json
import subprocess

FADE = 0.015    # seconds of fade at each join, so the cuts don't click


def merge(cuts, duration):
    out = []
    for a, b in sorted((max(0.0, a), min(duration, b)) for a, b in cuts):
        if b <= a:
            continue
        if out and a <= out[-1][1] + 0.05:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,bit_rate",
                        "-of", "json", path], capture_output=True, text=True, check=True)
    fmt = json.loads(r.stdout).get("format", {})
    return float(fmt.get("duration") or 0), int(float(fmt.get("bit_rate") or 128000))


def strip(src, dst, cuts):
    """Write dst: src without the [start, end] ranges in cuts.  Returns seconds removed."""
    duration, bit_rate = probe(src)
    cuts = merge(cuts, duration)
    keep, t = [], 0.0
    for a, b in cuts:
        if a - t > 0.05:
            keep.append((t, a))
        t = b
    if duration - t > 0.05:
        keep.append((t, duration))
    parts = []
    for i, (a, b) in enumerate(keep):
        f = f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS"
        if i > 0:
            f += f",afade=t=in:d={FADE}"
        if i < len(keep) - 1:
            f += f",afade=t=out:st={max(0.0, b - a - FADE):.3f}:d={FADE}"
        parts.append(f + f"[p{i}]")
    graph = ";".join(parts) + ";" + "".join(f"[p{i}]" for i in range(len(keep))) + \
        f"concat=n={len(keep)}:v=0:a=1[out]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex", graph, "-map", "[out]",
                    "-map_metadata", "0", "-c:a", "libmp3lame", "-b:a", f"{max(64, bit_rate // 1000)}k", dst],
                   check=True)
    return sum(b - a for a, b in cuts)
