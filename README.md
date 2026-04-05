# Qwen Captioner

Scene-cut-aware video analysis pipeline built around `mlx-community/Qwen3.5-4B-4bit`
on Apple Silicon.

It analyzes a video in 8-second windows, writes intermediate absolute scene data to
`.scene_cache`, and can generate an SRT subtitle file from that cache.

## Commands

```bash
uv run main.py path/to/video.mp4
uv run main.py path/to/video.mp4 --resume
uv run make_srt.py path/to/video.mp4
```

## Current Behavior

- timestamps are stored as integer milliseconds
- prompts require `MM:SS,mmm-MM:SS,mmm` output
- descriptions containing embedded timestamps are rejected and retried
- boundary re-inference runs inline while processing each new window
- when a boundary scene is merged, the current window cache absorbs that absolute
  scene, even if it starts before the nominal window boundary
- `--resume` can refresh boundary merges around regenerated windows without
  reprocessing every later window
- final timeline assembly and SRT generation both merge adjacent identical scenes

## Cache

Cache files are written to:

- `.scene_cache/{video_stem}_w{index}.json`

These cache entries are absolute timeline segments, not strictly clip-local window
segments.

## Notes

- `PIPELINE.md` describes the current runtime behavior in more detail
- `make_srt.py` expects cache files for the target video to already exist
