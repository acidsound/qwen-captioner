# Video Scene Analysis Pipeline

## Overview

This project analyzes a video with `mlx-community/Qwen3.5-4B-4bit` and produces
timestamped scene descriptions in milliseconds. The main output path is:

`video -> per-window analysis -> inline boundary re-inference -> cache -> SRT`

The current implementation is optimized for Apple Silicon and stores intermediate
results in `.scene_cache`.

---

## 1. Window Split

- `WINDOW_DURATION = 8`
- `FRAMES_PER_WINDOW = 12`
- `FRAME_SIZE = (384, 224)`

Each 8-second window is sampled into evenly spaced frames. Frame timestamps come
from the actual decoded frame position, not from a synthetic evenly spaced clock.

---

## 2. Prompt / Model Output

The model is prompted with:

- absolute frame timestamps
- a required output format of `MM:SS,mmm-MM:SS,mmm description`
- a rule that the first scene must start at the window start
- a rule that the last scene must end at the window end

`main.py` retries up to 3 total attempts (`MAX_RETRIES = 2`) with stronger prompt
language if the parsed output is invalid.

---

## 3. Parsing And Validation

Parsed timestamps are stored as integer milliseconds.

Accepted timestamp formats:

- `MM:SS`
- `MM:SS,mmm`
- `MM:SS:mmm`

Validation currently checks:

- non-empty output
- placeholder / boilerplate rejection
- start/end coverage of the analyzed window
- no large gaps inside the window
- no excessive repetition

If all retries fail, the code falls back to a single failure segment covering the
whole window in milliseconds.

---

## 4. Boundary Re-inference

Boundary re-inference happens inline while processing window `N`.

It compares:

- the last scene in window `N-1`
- the first scene in window `N`

If they are contiguous or overlapping within 200ms:

1. Re-extract frames from `last_prev.start -> first_curr.end`
2. Re-run a dedicated boundary prompt
3. Remove the previous window's last scene
4. Insert the re-inferred boundary scenes into the current window cache

Important current behavior:

- the current window cache is allowed to absorb a scene that starts before the
  nominal 8-second boundary
- this means cache files are not guaranteed to be clipped to their own window
- example:
  - `W0` may end at `6506-7216`
  - `W1` may begin with `7216-11887`

This is intentional so the cache itself reflects the merged boundary decision.

---

## 5. Cache Format

Cache files live at:

- `.scene_cache/{video_stem}_w{index}.json`

Format:

```json
{
  "scenes": [
    [7216, 11887, "close-up of a person's face in a bath..."],
    [11887, 13000, "close-up of a person's shoulder and back..."]
  ],
  "ts": 1775381288.094456
}
```

Notes:

- timestamps are integer milliseconds
- cache is saved immediately after each window analysis
- cache values are absolute timeline segments
- after boundary re-inference, a cache file can contain a scene that begins before
  that window's nominal start time

`--resume` reloads these cache files and skips already processed windows.

---

## 6. Final Timeline Assembly

After all windows are processed:

1. all cached scenes are flattened
2. adjacent scenes with the same description are merged if they are contiguous

This is what turns a boundary pair such as:

- `7216-8000 same description`
- `8000-11887 same description`

into:

- `7216-11887 same description`

for the final printed timeline.

---

## 7. SRT Generation

`make_srt.py`:

1. loads all cache files for a video
2. cleans descriptions
3. resolves overlaps
4. fixes zero-duration scenes
5. merges duplicate starts
6. merges adjacent identical descriptions
7. writes `{video_stem}.srt`

Because cache files can now hold cross-window scenes, the SRT path also merges
adjacent identical descriptions so boundary-merged scenes remain merged.

---

## 8. Current Status

Implemented:

- millisecond-based parsing and formatting
- full-window coverage validation
- inline boundary re-inference
- boundary scene absorption into the current cache
- final/SRT merge for adjacent identical scenes
- cache resume

Not currently wired into the runtime path:

- same-start re-analysis
- short-scene merge
- any dedicated debug export path

---

## 9. Usage

```bash
uv run main.py path/to/video.mp4
uv run main.py path/to/video.mp4 --resume
uv run make_srt.py path/to/video.mp4
```

---

## 10. Dependencies

```toml
[project]
dependencies = [
    "mlx-vlm>=0.4.3",
    "mlx>=0.22.0",
    "torch",
    "torchvision",
    "opencv-python-headless",
]
```
