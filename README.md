# agent-bakeoff

Working area for perception experiments that ingest a live RTSP/RTSPS camera feed instead of the thermal camera path.

## RTSP ingest

The first entry point is [src/rtsp_ingest.py](/Users/jason/Local/school/590-agents/project2/agent-bakeoff/src/rtsp_ingest.py). It uses `ffmpeg` to decode a live stream into raw `bgr24` frames and hands those frames to an OpenCV loop. The script is intended to be the bridge from shell-level stream validation to agent perception logic.

By default the script uses `ffprobe` to infer the stream width and height before starting the rawvideo pipe. You only need `--width` or `--height` if the stream metadata is wrong or unavailable.

Do not commit real stream URLs, credentials, or camera-specific paths. Pass the feed URL at runtime via `--url` or an environment variable in your local shell session.

From the repo root:

```bash
python3 src/rtsp_ingest.py \
  --url 'rtsps://camera.example.local:7441/your-stream-path' \
  --display
```

Useful options:

- `--max-frames 300` for short smoke tests
- `--print-every 0` to suppress per-frame logging
- `--tls-verify` if your RTSPS endpoint has a certificate you want to validate
- `--ffmpeg-bin /opt/homebrew/bin/ffmpeg` if `ffmpeg` is not already on `PATH`
- `--ffprobe-bin /opt/homebrew/bin/ffprobe` if `ffprobe` is not already on `PATH`
- `--debug-max-saved 0` to disable debug frame writes and keep the stream running continuously

### Debug frame capture

The default debug behavior is now a short validation run that saves 10 JPEGs to `debug_frames/session_01`, taking one every 30 source frames:

```bash
python3 src/rtsp_ingest.py --url 'rtsps://camera.example.local:7441/your-stream-path'
```

You can still override any part of that:

```bash
python3 src/rtsp_ingest.py \
  --url 'rtsps://camera.example.local:7441/your-stream-path' \
  --debug-output-dir debug_frames/session_02 \
  --debug-sample-every 15 \
  --debug-max-saved 20
```

The filenames include both the saved-frame index and the source-frame number, so you can tell how the sampling landed in the live stream.

Current shape:

- `ffmpeg` handles the RTSP/RTSPS session and decode
- Python receives fixed-size `numpy` frames in `BGR`
- the main loop is the place to attach perception or agent-facing logic

## Requirements

- Python 3
- `numpy`
- `opencv-python`
- `ffmpeg`
