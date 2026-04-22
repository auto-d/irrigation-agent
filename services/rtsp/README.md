# RTSP Service

This directory contains the in-repo RTSP/RTSPS ingest helper used by the agent project's perception layer.

## Layout

- `ingest.py`: ffmpeg-backed frame ingest with optional OpenCV display and sampled debug-frame capture

## CLI Usage

Run from the repository root:

```bash
python -m services.rtsp.ingest --url 'rtsps://camera.example.local:7441/your-stream-path'
python -m services.rtsp.ingest --url 'rtsps://camera.example.local:7441/your-stream-path' --display --debug-max-saved 0
python -m services.rtsp.ingest --url 'rtsps://camera.example.local:7441/your-stream-path' --debug-sample-every 15 --debug-max-saved 20
```

By default the service uses `ffprobe` to infer stream width and height before starting the rawvideo pipe. Override `--width` or `--height` only when stream metadata is missing or incorrect.

To present the classical anomaly pipeline on top of live video, point the helper at your baseline still-image directory and enable display:

```bash
python -m services.rtsp.ingest \
  --url 'rtsps://camera.example.local:7441/your-stream-path' \
  --display \
  --debug-max-saved 0 \
  --detector-baseline-dir vision_baseline \
  --detector-mode baseline_suppressed_combo \
  --detector-refresh-every 5
```

That mode:

- builds the nuisance-suppressed baseline once from the still-image directory
- recomputes the anomaly map every `N` source frames
- overlays the residual heatmap and candidate regions onto the live feed
- highlights the most plausible anomalous region in green

Useful detector options:

- `--detector-mode lab_illumination_intersection` to compare the unsuppressed detector
- `--detector-refresh-every 10` to reduce detector cost relative to stream frame rate
- `--detector-alpha 0.25` to make the heatmap overlay more subtle

## Credentials And URLs

Do not commit real stream URLs, credentials, or camera-specific paths. Pass the feed URL at runtime via `--url` or through your local shell environment only.
