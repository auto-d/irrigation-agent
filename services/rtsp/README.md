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

## Credentials And URLs

Do not commit real stream URLs, credentials, or camera-specific paths. Pass the feed URL at runtime via `--url` or through your local shell environment only.
