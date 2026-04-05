"""
Generate SRT subtitle file from scene cache JSON files.

Usage:
    uv run make_srt.py                     # Uses default video name
    uv run make_srt.py path/to/video.mp4   # Analyze your video
"""

import json
import re
import sys
from pathlib import Path

CACHE_DIR = ".scene_cache"
TIMECODE_RE = re.compile(r"\b\d{2}:\d{2}(?:[,:]\d{3})?\b")


def fmt_srt(value) -> str:
    """Format timestamp as SRT: HH:MM:SS,mmm

    Handles both formats:
    - Integer milliseconds (e.g., 4000)
    - Float seconds (e.g., 4.0)
    """
    if isinstance(value, int) or (isinstance(value, float) and value > 1000):
        # Integer milliseconds
        total_ms = int(value)
    else:
        # Float seconds
        total_ms = round(value * 1000)

    h = total_ms // 3600000
    m = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_description(desc: str) -> str:
    """Clean description for subtitle display."""
    # Remove XML tags
    desc = re.sub(r"<description>|</description>", "", desc)
    # Remove "description:" prefix
    desc = re.sub(r"^description:\s*", "", desc, flags=re.IGNORECASE)
    # Remove markdown
    desc = re.sub(r"\*\*|\*|`", "", desc)
    # Remove leading timestamp
    desc = re.sub(r"^\d{2}:\d{2}-\d{2}:\d{2}[:\s]*", "", desc)
    # Remove "analysis failed" messages
    if "analysis failed" in desc.lower():
        return ""
    if "\n" in desc or TIMECODE_RE.search(desc):
        return ""
    # Remove leading/trailing whitespace and angle brackets
    desc = desc.strip().strip("<>")
    return desc


def merge_adjacent_scenes(scenes: list, gap_tolerance_ms: int = 50) -> list:
    """Merge contiguous scenes that carry the same description."""
    if not scenes:
        return []

    ordered = sorted(scenes, key=lambda item: (item[0], item[1], item[2]))
    merged = [ordered[0]]
    for s, e, d in ordered[1:]:
        ps, pe, pd = merged[-1]
        if d == pd and s <= pe + gap_tolerance_ms:
            merged[-1] = (ps, max(pe, e), pd)
        else:
            merged.append((s, e, d))
    return merged


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"
    video_name = Path(video_path).stem
    cache_dir = Path(CACHE_DIR)

    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        sys.exit(1)

    # Load all cache files
    scenes = []
    cache_files = sorted(cache_dir.glob(f"{video_name}_w*.json"))

    if not cache_files:
        print(f"No cache files found for: {video_name}")
        sys.exit(1)

    print(f"Loading {len(cache_files)} cache files...")

    for cf in cache_files:
        with open(cf) as f:
            data = json.load(f)
            for s, e, d in data.get("scenes", []):
                cleaned = clean_description(d)
                if cleaned:
                    scenes.append((s, e, cleaned))

    # Sort by start time
    scenes.sort(key=lambda x: x[0])

    # Step 1: Resolve overlaps between consecutive scenes
    # When scene N starts before scene N-1 ends, adjust scene N-1's end
    resolved = []
    for s, e, d in scenes:
        if resolved and s < resolved[-1][1]:
            # Overlap: adjust previous scene's end to this scene's start
            ps, pe, pd = resolved[-1]
            resolved[-1] = (ps, s, pd)
        resolved.append((s, e, d))
    scenes = resolved

    # Step 2: Fix zero-duration scenes (start == end) by distributing sequentially
    # Each scene gets at least 1 frame (~33ms at 30fps) or fills gap to next scene
    fixed = []
    for i, (s, e, d) in enumerate(scenes):
        if e <= s:
            # Zero duration — start from previous scene's end
            if fixed:
                s = fixed[-1][1]
            # End at next scene's start, or +1s (1000ms) if last
            if i + 1 < len(scenes) and scenes[i + 1][0] > s:
                e = scenes[i + 1][0]
            else:
                e = s + 1000
        fixed.append((s, e, d))
    scenes = fixed

    # Merge only truly duplicate start times (within 33ms = 1 frame tolerance)
    merged = []
    for s, e, d in scenes:
        if merged and abs(s - merged[-1][0]) < 34:
            # Same frame — merge descriptions
            ps, pe, pd = merged[-1]
            merged[-1] = (ps, max(e, pe), pd + " | " + d)
        else:
            merged.append((s, e, d))
    scenes = merge_adjacent_scenes(merged)

    # Generate SRT
    srt_path = Path(f"{video_name}.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (s, e, d) in enumerate(scenes, 1):
            # End time: use next scene start or current end
            if i < len(scenes):
                next_start = scenes[i][0]
                end_time = max(e, next_start)
            else:
                end_time = e

            f.write(f"{i}\n")
            f.write(f"{fmt_srt(s)} --> {fmt_srt(end_time)}\n")
            f.write(f"{d}\n\n")

    print(f"Generated {srt_path} with {len(scenes)} subtitles")

    # Print preview
    print(f"\n{'=' * 60}")
    print("PREVIEW (first 10):")
    print(f"{'=' * 60}")
    for i, (s, e, d) in enumerate(scenes[:10], 1):
        end_time = scenes[i][0] if i < len(scenes) else e
        print(f"{i}")
        print(f"{fmt_srt(s)} --> {fmt_srt(end_time)}")
        print(f"{d}")
        print()


if __name__ == "__main__":
    main()
