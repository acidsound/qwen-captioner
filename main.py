"""
Scene-cut-aware video analysis with Qwen 3.5 4B on Apple Silicon (MLX).

Optimizations:
1. English prompts for consistency
2. Per-frame explicit timestamps
3. Retry with stronger prompts on failure
4. Cache with resume capability
5. Duplicate/repetition filtering

Usage:
    uv run main.py
    uv run main.py path/to/video.mp4
    uv run main.py path/to/video.mp4 --resume
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from mlx_vlm.video_generate import process_vision_info

# === Tunable Parameters ===
MODEL_ID = "mlx-community/Qwen3.5-4B-4bit"
WINDOW_DURATION = 8  # seconds per analysis window
FRAMES_PER_WINDOW = 12  # frames per window
FRAME_SIZE = (384, 224)  # frame resolution
MAX_TOKENS = 1024  # max output tokens per window
ENABLE_THINKING = False  # disable thinking mode
MAX_RETRIES = 2  # retry attempts on failure
CACHE_DIR = ".scene_cache"
SCENE_LINE_FORMAT = "MM:SS,mmm-MM:SS,mmm description text here"
WINDOW_COVERAGE_TOLERANCE_MS = 1000
# =========================


def extract_frames_from_range(
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int,
    frame_size: tuple,
):
    """Extract evenly spaced frames from a time range.

    Returns: (frames, timestamps) where timestamps are actual frame times in seconds
    """
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    total = end_frame - start_frame

    if total <= 0:
        cap.release()
        return [], []

    if num_frames > 1:
        indices = [
            int(start_frame + i * total / (num_frames - 1)) for i in range(num_frames)
        ]
    else:
        indices = [start_frame + total // 2]
    indices = sorted(set(idx for idx in indices if idx < end_frame))

    frames = []
    timestamps = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize(frame_size, Image.LANCZOS)
        frames.append(img)
        # Get actual frame time in seconds
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

    cap.release()
    return frames, timestamps


def fmt(seconds: float) -> str:
    """Format seconds (float) as MM:SS,mmm for prompt display."""
    total_ms = round(seconds * 1000)
    m = total_ms // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{m:02d}:{s:02d},{ms:03d}"


def fmt_ms(ms: int) -> str:
    """Format integer milliseconds as MM:SS,mmm for display."""
    total = ms
    m = total // 60000
    s = (total % 60000) // 1000
    ms_part = total % 1000
    return f"{m:02d}:{s:02d},{ms_part:03d}"


def get_cache_path(video_path: str, window_idx: int) -> Path:
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{Path(video_path).stem}_w{window_idx}.json"


def load_cache(video_path: str, window_idx: int):
    p = get_cache_path(video_path, window_idx)
    if p.exists():
        with open(p) as f:
            return json.load(f).get("scenes")
    return None


def save_cache(video_path: str, window_idx: int, scenes: list):
    p = get_cache_path(video_path, window_idx)
    with open(p, "w") as f:
        json.dump({"scenes": scenes, "ts": time.time()}, f)


def clip_scenes_to_range(scenes: list, range_start_ms: int, range_end_ms: int) -> list:
    """Clip scenes to an absolute time range."""
    clipped = []
    for s, e, d in scenes:
        cs = max(int(s), range_start_ms)
        ce = min(int(e), range_end_ms)
        if ce > cs:
            clipped.append((cs, ce, d))
    return clipped


def normalize_window_scenes(
    scenes: list, window_start_ms: int, window_end_ms: int
) -> list:
    """Clamp scenes to a window and make coverage contiguous."""
    cleaned = []
    for s, e, d in scenes:
        desc = d.strip()
        if not desc:
            continue
        s_ms = max(window_start_ms, min(int(round(s)), window_end_ms))
        e_ms = max(window_start_ms, min(int(round(e)), window_end_ms))
        if e_ms <= s_ms:
            e_ms = min(window_end_ms, s_ms + 33)
        if e_ms > s_ms:
            cleaned.append((s_ms, e_ms, desc))

    if not cleaned:
        return []

    cleaned.sort(key=lambda item: (item[0], item[1], item[2]))

    fixed = []
    for s_ms, e_ms, desc in cleaned:
        if fixed:
            prev_start, prev_end, prev_desc = fixed[-1]
            if s_ms > prev_end:
                fixed[-1] = (prev_start, s_ms, prev_desc)
            elif s_ms < prev_end:
                s_ms = prev_end
        if e_ms <= s_ms:
            e_ms = min(window_end_ms, s_ms + 33)
        if e_ms > s_ms:
            fixed.append((s_ms, e_ms, desc))

    if not fixed:
        return []

    first_start, first_end, first_desc = fixed[0]
    fixed[0] = (window_start_ms, max(first_end, window_start_ms + 33), first_desc)

    last_start, last_end, last_desc = fixed[-1]
    fixed[-1] = (min(last_start, window_end_ms - 33), window_end_ms, last_desc)

    normalized = []
    for s_ms, e_ms, desc in fixed:
        if normalized and s_ms < normalized[-1][1]:
            prev_start, _, prev_desc = normalized[-1]
            normalized[-1] = (prev_start, s_ms, prev_desc)
        if e_ms <= s_ms:
            e_ms = min(window_end_ms, s_ms + 33)
        if e_ms > s_ms:
            normalized.append((s_ms, e_ms, desc))

    return normalized


def normalize_absolute_scenes(scenes: list) -> list:
    """Sort absolute scenes and remove overlaps without window clipping."""
    cleaned = []
    for s, e, d in scenes:
        desc = d.strip()
        s_ms = int(round(s))
        e_ms = int(round(e))
        if not desc:
            continue
        if e_ms <= s_ms:
            e_ms = s_ms + 33
        cleaned.append((s_ms, e_ms, desc))

    if not cleaned:
        return []

    cleaned.sort(key=lambda item: (item[0], item[1], item[2]))
    normalized = [cleaned[0]]
    for s_ms, e_ms, desc in cleaned[1:]:
        ps, pe, pd = normalized[-1]
        if s_ms < pe:
            s_ms = pe
        if e_ms <= s_ms:
            e_ms = s_ms + 33
        normalized.append((s_ms, e_ms, desc))
    return merge_adjacent_scenes(normalized)


def flatten_scenes(scenes_by_window: dict) -> list:
    flattened = []
    for window_idx in sorted(scenes_by_window):
        for s, e, d in scenes_by_window[window_idx]:
            flattened.append((s, e, d))
    return merge_adjacent_scenes(flattened)


def count_scenes(scenes_by_window: dict) -> int:
    return sum(len(scenes) for scenes in scenes_by_window.values())


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


def is_valid(scenes: list, window_start: float, window_end: float) -> bool:
    """Validate scene analysis output."""
    if not scenes:
        return False

    bad = [
        "scene description",
        "description of the scene",
        "scene analysis",
        "scene transition",
        "scene change",
        "구간 분석",
        "장면 설명",
        "장면 전환",
        "<description>",
        "</description>",
    ]

    ordered = sorted(scenes, key=lambda item: (item[0], item[1]))
    win_start_ms = int(round(window_start * 1000))
    win_end_ms = int(round(window_end * 1000))

    if abs(ordered[0][0] - win_start_ms) > WINDOW_COVERAGE_TOLERANCE_MS:
        return False
    if abs(ordered[-1][1] - win_end_ms) > WINDOW_COVERAGE_TOLERANCE_MS:
        return False

    descriptions = []
    prev_end = win_start_ms
    for s, e, d in ordered:
        dl = d.lower().strip()
        if any(p in dl for p in bad):
            return False
        # Reject placeholder descriptions
        cleaned = dl.strip("#*- ")
        if cleaned in ("description", "description."):
            return False
        if cleaned.startswith("description of"):
            return False
        if len(dl.strip("#*- ")) < 5:
            return False
        if e <= s:
            return False
        if s < win_start_ms - WINDOW_COVERAGE_TOLERANCE_MS:
            return False
        if e > win_end_ms + WINDOW_COVERAGE_TOLERANCE_MS:
            return False
        if s - prev_end > WINDOW_COVERAGE_TOLERANCE_MS:
            return False
        prev_end = max(prev_end, e)
        descriptions.append(dl)

    # Check repetition (>60% identical)
    if len(descriptions) > 2:
        from collections import Counter

        most = Counter(descriptions).most_common(1)[0][1]
        if most / len(descriptions) > 0.6:
            return False

    return True


def build_prompt(frames, frame_timestamps, ws: float, we: float, attempt: int = 0):
    """Build scene analysis prompt with per-frame timestamps."""
    timestamps = frame_timestamps

    frame_labels = "\n".join(
        f"  Frame {i + 1}: {fmt(ts)}" for i, ts in enumerate(timestamps)
    )

    base = (
        f"You are analyzing video frames from {fmt(ws)} to {fmt(we)}.\n"
        f"Each frame's exact timestamp:\n{frame_labels}\n\n"
        f"Task:\n"
        f"1. Identify scene cut points by comparing consecutive frames.\n"
        f"2. Group consecutive frames that belong to the same scene.\n"
        f"3. For each scene, output in this exact format:\n\n"
        f"{SCENE_LINE_FORMAT}\n\n"
        f"IMPORTANT: Use ABSOLUTE timestamps from the frame list above. "
        f"Do NOT use relative timestamps. If frames show 01:12,000, output 01:12,000, NOT 00:12,000.\n"
        f"The first scene MUST start at {fmt(ws)} and the last scene MUST end at {fmt(we)}.\n"
        f"Use millisecond precision. Do NOT round 00:02,000 to 00:03.\n\n"
        f"Rules:\n"
        f"- Use the frame timestamps above to determine exact start/end times.\n"
        f"- Only split when there is a CLEAR scene change (different location, subject, or action).\n"
        f"- Do NOT split for minor camera movements (pan, zoom, angle change).\n"
        f"- Each scene must be at least 2 seconds. Merge shorter segments into the same scene.\n"
        f"- Describe ONLY what is visible. Do NOT hallucinate or guess.\n"
        f"- Keep each description to one concise sentence.\n"
        f"- Include: subject, action, camera angle, lighting, background.\n"
        f"- Start each line with MM:SS,mmm-MM:SS,mmm timestamp range.\n"
        f"- Output ONLY the scene lines. NO preamble, NO explanation, NO markdown, NO XML tags like <description>."
    )

    if attempt >= 1:
        base += (
            "\n\nIMPORTANT: Do NOT use placeholder text like 'scene description'. "
            "Write actual content describing what you see in each frame. "
            "Merge similar frames into the same scene (minimum 2 seconds per scene). "
            f"Keep full coverage from {fmt(ws)} through {fmt(we)}."
        )
    if attempt >= 2:
        base += (
            "\n\nCRITICAL: Previous attempts failed. You MUST describe actual visual "
            "content. Example: '00:00,000-00:02,000 White lingerie hanging on a clothesline "
            "by a sunlit window.'"
        )

    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": f} for f in frames],
                {"type": "text", "text": base},
            ],
        }
    ]

    return messages, timestamps


def analyze(model, processor, config, frames, frame_timestamps, ws: float, we: float):
    """Analyze a window with retry logic."""
    for attempt in range(MAX_RETRIES + 1):
        messages, timestamps = build_prompt(frames, frame_timestamps, ws, we, attempt)
        images, _ = process_vision_info(messages)

        prompt = apply_chat_template(
            processor,
            config,
            messages,
            num_images=len(images) if images else 0,
            enable_thinking=ENABLE_THINKING,
        )

        temp = 0.3 if attempt == 0 else 0.5 + attempt * 0.2

        response = generate(
            model,
            processor,
            prompt,
            image=images,
            temp=temp,
            max_tokens=MAX_TOKENS,
            verbose=False,
        )

        scenes = parse_output(response.text, timestamps, ws, we)

        if is_valid(scenes, ws, we):
            if attempt > 0:
                print(f"    (retry {attempt} OK)")
            return normalize_window_scenes(
                scenes, int(round(ws * 1000)), int(round(we * 1000))
            )
        else:
            print(f"    (attempt {attempt + 1} failed, retrying...)")

    print(f"    ⚠ All {MAX_RETRIES + 1} attempts failed")
    return [(int(round(ws * 1000)), int(round(we * 1000)), f"{fmt(ws)}-{fmt(we)} 분석 실패")]


def reanalyze_same_start(
    model, processor, config, frames, frame_timestamps, ws: float, we: float
):
    """Re-analyze same-start conflict: force single scene output."""
    for attempt in range(MAX_RETRIES + 1):
        # Use actual frame timestamps
        timestamps = frame_timestamps

        frame_labels = "\n".join(
            f"  Frame {i + 1}: {fmt(ts)}" for i, ts in enumerate(timestamps)
        )

        prompt_text = (
            f"You are analyzing video frames from {fmt(ws)} to {fmt(we)}.\n"
            f"Each frame's exact timestamp:\n{frame_labels}\n\n"
            f"Task: These frames form a CONTINUOUS segment. Output a SINGLE scene description.\n\n"
            f"Output format:\n\n"
            f"{SCENE_LINE_FORMAT}\n\n"
            f"IMPORTANT:\n"
            f"- Output EXACTLY ONE scene covering the entire time range.\n"
            f"- Use ABSOLUTE timestamps from the frame list above.\n"
            f"- Start at {fmt(ws)} and end at {fmt(we)}.\n"
            f"- Do NOT split into multiple scenes.\n"
            f"- Describe the overall visual content as one concise sentence.\n"
            f"- Output ONLY the scene line. NO preamble, NO explanation, NO markdown, NO XML tags."
        )

        if attempt >= 1:
            prompt_text += (
                "\n\nCRITICAL: You MUST output exactly ONE scene line. "
                "Do NOT split into multiple scenes."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": f} for f in frames],
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        images, _ = process_vision_info(messages)

        prompt = apply_chat_template(
            processor,
            config,
            messages,
            num_images=len(images) if images else 0,
            enable_thinking=ENABLE_THINKING,
        )

        temp = 0.3 if attempt == 0 else 0.5 + attempt * 0.2

        response = generate(
            model,
            processor,
            prompt,
            image=images,
            temp=temp,
            max_tokens=MAX_TOKENS,
            verbose=False,
        )

        scenes = parse_output(response.text, timestamps, ws, we)

        # Validate: must be exactly 1 scene
        if len(scenes) == 1:
            if attempt > 0:
                print(f"    (retry {attempt} OK)")
            return normalize_window_scenes(
                scenes, int(round(ws * 1000)), int(round(we * 1000))
            )
        else:
            print(
                f"    (attempt {attempt + 1} returned {len(scenes)} scenes, expected 1)"
            )

    # Fallback: return single scene covering full range
    s_ms = int(ws * 1000)
    e_ms = int(we * 1000)
    return [(s_ms, e_ms, f"{fmt(ws)}-{fmt(we)} continuous segment")]


def reanalyze_boundary(
    model, processor, config, frames, frame_timestamps, ws: float, we: float
):
    """Re-analyze window boundary: determine if scene cut exists."""
    for attempt in range(MAX_RETRIES + 1):
        # Use actual frame timestamps
        timestamps = frame_timestamps

        frame_labels = "\n".join(
            f"  Frame {i + 1}: {fmt(ts)}" for i, ts in enumerate(timestamps)
        )

        prompt_text = (
            f"You are analyzing video frames from {fmt(ws)} to {fmt(we)}.\n"
            f"Each frame's exact timestamp:\n{frame_labels}\n\n"
            f"Task: Identify scene cut points and group consecutive frames that belong to the same scene.\n\n"
            f"Output format:\n\n"
            f"{SCENE_LINE_FORMAT}\n\n"
            f"IMPORTANT: Use ABSOLUTE timestamps from the frame list above. "
            f"Do NOT use relative timestamps.\n"
            f"The first scene MUST start at {fmt(ws)} and the last scene MUST end at {fmt(we)}.\n"
            f"Use millisecond precision.\n\n"
            f"Rules:\n"
            f"- Only split when there is a CLEAR scene change (different location, subject, or action).\n"
            f"- Do NOT split for minor camera movements (pan, zoom, angle change).\n"
            f"- Each scene must be at least 2 seconds. Merge shorter segments into the same scene.\n"
            f"- Describe ONLY what is visible. Do NOT hallucinate or guess.\n"
            f"- Keep each description to one concise sentence.\n"
            f"- Start each line with MM:SS,mmm-MM:SS,mmm timestamp range.\n"
            f"- Output ONLY the scene lines. NO preamble, NO explanation, NO markdown, NO XML tags like <description>."
        )

        if attempt >= 1:
            prompt_text += (
                "\n\nIMPORTANT: Do NOT use placeholder text like 'scene description'. "
                "Write actual content describing what you see."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": f} for f in frames],
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        images, _ = process_vision_info(messages)

        prompt = apply_chat_template(
            processor,
            config,
            messages,
            num_images=len(images) if images else 0,
            enable_thinking=ENABLE_THINKING,
        )

        temp = 0.3 if attempt == 0 else 0.5 + attempt * 0.2

        response = generate(
            model,
            processor,
            prompt,
            image=images,
            temp=temp,
            max_tokens=MAX_TOKENS,
            verbose=False,
        )

        scenes = parse_output(response.text, timestamps, ws, we)

        if is_valid(scenes, ws, we):
            if attempt > 0:
                print(f"    (retry {attempt} OK)")
            return normalize_window_scenes(
                scenes, int(round(ws * 1000)), int(round(we * 1000))
            )
        else:
            print(f"    (attempt {attempt + 1} failed, retrying...)")

    # Fallback: return single scene covering full range
    s_ms = int(ws * 1000)
    e_ms = int(we * 1000)
    return [(s_ms, e_ms, f"{fmt(ws)}-{fmt(we)} boundary segment")]


def parse_output(text: str, timestamps: list, ws: float, we: float) -> list:
    """Parse model output into (start_ms, end_ms, desc) tuples.

    Timestamps stored as integer milliseconds.
    Handles MM:SS, MM:SS,mmm, and MM:SS:mmm formats from model output.
    """
    scenes = []
    # Flexible pattern: matches "00:04", "00:04,000", or "00:04:000"
    pattern = r"(\d{2}:\d{2}(?:[,:]\d{3})?)\s*[-–—]\s*(\d{2}:\d{2}(?:[,:]\d{3})?)\*?\*?\s*[:\s]\s*(.+)"

    def parse_ts(ts_str: str) -> int:
        """Parse MM:SS, MM:SS,mmm, or MM:SS:mmm to integer milliseconds."""
        parts = ts_str.split(":")
        m = int(parts[0])

        if len(parts) == 3:
            # Handle MM:SS:mmm format (model sometimes uses colon instead of comma)
            s = int(parts[1])
            ms = int(parts[2])
            return (m * 60 + s) * 1000 + ms
        else:
            rest = parts[1]
            if "," in rest:
                s_str, ms_str = rest.split(",")
                return (m * 60 + int(s_str)) * 1000 + int(ms_str)
            else:
                return (m * 60 + int(rest)) * 1000

    for line in text.strip().split("\n"):
        line = line.strip()
        line = re.sub(r"^[-*•]\s*\*?\*?", "", line).strip()
        line = re.sub(r"^\*\*\d{2}:\d{2}", "", line).strip()
        if not line:
            continue
        m = re.search(pattern, line)
        if m:
            s_str, e_str, desc = m.groups()
            s_ms = parse_ts(s_str)
            e_ms = parse_ts(e_str)

            # Ensure e > s
            if e_ms <= s_ms:
                e_ms = s_ms + 33  # 1 frame at 30fps

            # Clean description
            desc = re.sub(r"<description>|</description>", "", desc).strip()
            desc = re.sub(r"^description:\s*", "", desc, flags=re.IGNORECASE).strip()
            desc = re.sub(r"\*\*|\*|`", "", desc).strip()
            desc = re.sub(r"^[-:]+\s*", "", desc).strip()
            if desc:
                scenes.append((s_ms, e_ms, desc))

    if not scenes:
        # Fallback: use the full requested window as one scene
        s_ms = int(round(ws * 1000))
        e_ms = int(round(we * 1000))
        scenes.append((s_ms, e_ms, text.strip()))

    return scenes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?", default="test_video.mp4")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def reanalyze_window_boundary(
    model,
    processor,
    config,
    video_path: str,
    duration: float,
    prev_idx: int,
    curr_idx: int,
    scenes_by_window: dict,
):
    prev_scenes = scenes_by_window.get(prev_idx, [])
    curr_scenes = scenes_by_window.get(curr_idx, [])
    if not prev_scenes or not curr_scenes:
        return

    last_prev = prev_scenes[-1]
    first_curr = curr_scenes[0]
    if first_curr[0] > last_prev[1] + 200:
        return

    combined_start_sec = last_prev[0] / 1000.0
    combined_end_sec = first_curr[1] / 1000.0
    combined_duration = combined_end_sec - combined_start_sec
    if combined_duration <= 0 or combined_duration > WINDOW_DURATION * 2:
        return

    b_frames, b_frame_timestamps = extract_frames_from_range(
        str(video_path),
        combined_start_sec,
        combined_end_sec,
        FRAMES_PER_WINDOW,
        FRAME_SIZE,
    )
    if not b_frames:
        return

    re_scenes = reanalyze_boundary(
        model,
        processor,
        config,
        b_frames,
        b_frame_timestamps,
        combined_start_sec,
        combined_end_sec,
    )

    scenes_by_window[prev_idx] = normalize_absolute_scenes(prev_scenes[:-1])
    scenes_by_window[curr_idx] = normalize_absolute_scenes(re_scenes + curr_scenes[1:])
    print(
        f"  W{prev_idx + 1}/W{curr_idx + 1}: current cache absorbed "
        f"{len(re_scenes)} boundary scene(s)"
    )


def main():
    args = parse_args()
    video_path = str(Path(args.video_path).expanduser())

    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0
    cap.release()

    print(f"Video: {video_path}")
    print(f"Duration: {duration:.1f}s ({duration / 60:.1f} min)")
    print(f"Model: {MODEL_ID}")
    print(f"Settings: {WINDOW_DURATION}s window, {FRAMES_PER_WINDOW} frames")

    print(f"Loading model: {MODEL_ID}")
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)

    num_w = math.ceil(duration / WINDOW_DURATION) if duration > 0 else 0

    print(f"\nProcessing {num_w} windows...\n")

    t0 = time.time()
    cached = 0
    scenes_by_window = {}

    for i in range(num_w):
        ws = i * WINDOW_DURATION
        we = min((i + 1) * WINDOW_DURATION, duration)
        ws_ms = int(round(ws * 1000))
        we_ms = int(round(we * 1000))

        if args.resume:
            c = load_cache(video_path, i)
            if c:
                scenes_by_window[i] = normalize_absolute_scenes(c)
                cached += 1
                print(f"  ✓ W{i + 1}/{num_w}: {fmt(ws)}-{fmt(we)} (cached)")
                continue

        pct = (i + 1) / num_w * 100
        print(f"  [{pct:5.1f}%] W{i + 1}/{num_w}: {fmt(ws)}-{fmt(we)}")

        processed_windows = i
        if processed_windows > 0:
            elapsed = time.time() - t0
            eta = (num_w - processed_windows) * (elapsed / processed_windows) / 60
            print(f"    ETA: ~{eta:.1f} min")

        print(f"    Scenes so far: {count_scenes(scenes_by_window)}")

        frames, frame_timestamps = extract_frames_from_range(
            str(video_path), ws, we, FRAMES_PER_WINDOW, FRAME_SIZE
        )
        if not frames:
            continue
        print(f"    {len(frames)} frames")

        try:
            scenes = analyze(model, processor, config, frames, frame_timestamps, ws, we)
            scenes_by_window[i] = normalize_window_scenes(scenes, ws_ms, we_ms)
            print(f"    Total scenes collected: {count_scenes(scenes_by_window)}")
            for s, e, d in scenes_by_window[i]:
                print(f"    → {fmt_ms(s)}-{fmt_ms(e)} {d[:60]}...")

            if i > 0:
                reanalyze_window_boundary(
                    model,
                    processor,
                    config,
                    video_path,
                    duration,
                    i - 1,
                    i,
                    scenes_by_window,
                )

            print("    Saving cache...")
            save_cache(video_path, i, scenes_by_window[i])
            if i > 0 and (i - 1) in scenes_by_window:
                save_cache(video_path, i - 1, scenes_by_window[i - 1])

        except Exception as e:
            import traceback

            print(f"    Error: {e}")
            traceback.print_exc()

    print(f"\n  [100.0%] All windows processed.")

    # Verify cache files
    print("\nCache verification:")
    for w_idx in sorted(scenes_by_window.keys()):
        cache_path = get_cache_path(video_path, w_idx)
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            scenes = data.get("scenes", [])
            if scenes:
                last = scenes[-1]
                print(f"  W{w_idx}: last scene {last[0]}-{last[1]} ({last[2][:40]})")

    # Output
    elapsed = time.time() - t0
    all_scenes = flatten_scenes(scenes_by_window)
    print(f"\n{'=' * 70}")
    print(f"SCENE ANALYSIS ({elapsed:.0f}s, {cached} cached)")
    print(f"{'=' * 70}")

    # Deduplicate
    seen = set()
    cleaned = []
    for item in all_scenes:
        s, e, d = item[0], item[1], item[2]
        d = d.strip()
        # Remove leading timestamp prefix
        prefix = f"{fmt_ms(s)}-{fmt_ms(e)}"
        if d.startswith(prefix):
            d = d[len(prefix) :].strip()
        d = re.sub(r"^\d{2}:\d{2}-\d{2}:\d{2}:\s*", "", d).strip()
        key = (s, e, d[:50])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((s, e, d))

    for s, e, d in cleaned:
        print(f"{fmt_ms(s)}-{fmt_ms(e)} {d}")


if __name__ == "__main__":
    main()
