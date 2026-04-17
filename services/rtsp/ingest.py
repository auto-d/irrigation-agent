#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np


class FFmpegVideoStream:
    def __init__(
        self,
        url,
        width,
        height,
        transport="tcp",
        tls_verify=False,
        ffmpeg_bin="ffmpeg",
        extra_input_args=None,
    ):
        self.url = url
        self.width = width
        self.height = height
        self.transport = transport
        self.tls_verify = tls_verify
        self.ffmpeg_bin = ffmpeg_bin
        self.extra_input_args = extra_input_args or []
        self.channels = 3
        self.frame_size = self.width * self.height * self.channels
        self.proc = None

    def command(self):
        cmd = [self.ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
        if self.transport:
            cmd.extend(["-rtsp_transport", self.transport])
        if not self.tls_verify:
            cmd.extend(["-tls_verify", "0"])
        cmd.extend(self.extra_input_args)
        cmd.extend(
            [
                "-i",
                self.url,
                "-an",
                "-sn",
                "-dn",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "-",
            ]
        )
        return cmd

    def start(self):
        if self.proc is not None:
            return self

        self.proc = subprocess.Popen(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
        )
        return self

    def read(self):
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("stream has not been started")

        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            stderr = b""
            if self.proc.stderr is not None:
                stderr = self.proc.stderr.read()
            message = stderr.decode("utf-8", errors="replace").strip()
            return None, message

        frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, self.channels))
        return frame, None

    def close(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)
        self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class DebugFrameWriter:
    def __init__(self, output_dir, sample_every, max_saved, jpeg_quality):
        self.output_dir = Path(output_dir)
        self.sample_every = max(1, sample_every)
        self.max_saved = max_saved
        self.jpeg_quality = jpeg_quality
        self.saved_count = 0

    def enabled(self):
        return self.max_saved != 0

    def should_save(self, frame_count):
        if not self.enabled():
            return False
        if self.max_saved > 0 and self.saved_count >= self.max_saved:
            return False
        return frame_count % self.sample_every == 0

    def save(self, frame, frame_count, elapsed):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = self.output_dir / f"frame_{self.saved_count:04d}_src_{frame_count:06d}.jpg"
        ok = cv2.imwrite(
            str(filename),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            raise RuntimeError(f"failed to write debug frame to {filename}")
        self.saved_count += 1
        print(f"saved_debug_frame path={filename} src_frame={frame_count} elapsed={elapsed:.2f}s")


def ffprobe_stream_dimensions(url, transport="tcp", tls_verify=False, ffprobe_bin="ffprobe"):
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
    ]
    if transport:
        cmd.extend(["-rtsp_transport", transport])
    if not tls_verify:
        cmd.extend(["-tls_verify", "0"])
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"ffprobe failed to inspect stream dimensions: {stderr or 'unknown error'}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned malformed JSON while probing stream dimensions") from exc

    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe found no video streams in the RTSP source")

    width = streams[0].get("width")
    height = streams[0].get("height")
    if not width or not height:
        raise RuntimeError("ffprobe did not return width/height for the first video stream")

    return int(width), int(height)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read RTSP/RTSPS video frames via ffmpeg and expose them to an OpenCV perception loop."
    )
    parser.add_argument("--url", required=True, help="RTSP or RTSPS stream URL.")
    parser.add_argument("--width", type=int, help="Override decoded frame width. Defaults to ffprobe output.")
    parser.add_argument("--height", type=int, help="Override decoded frame height. Defaults to ffprobe output.")
    parser.add_argument(
        "--transport",
        choices=["tcp", "udp", "udp_multicast", "http", "https"],
        default="tcp",
        help="RTSP transport passed to ffmpeg.",
    )
    parser.add_argument(
        "--tls-verify",
        action="store_true",
        help="Enable TLS certificate verification for RTSPS sources.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="Path to the ffmpeg executable.",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="Path to the ffprobe executable used for stream dimension detection.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for short test runs. 0 means unlimited.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display frames in an OpenCV window.",
    )
    parser.add_argument(
        "--window-name",
        default="rtsp-frame",
        help="OpenCV window name when --display is enabled.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Log one line of frame timing every N frames. 0 disables logging.",
    )
    parser.add_argument(
        "--debug-output-dir",
        default="debug_frames/session_01",
        help="Directory used for sampled debug frame writes.",
    )
    parser.add_argument(
        "--debug-sample-every",
        type=int,
        default=30,
        help="Write one debug frame every N source frames.",
    )
    parser.add_argument(
        "--debug-max-saved",
        type=int,
        default=10,
        help="Maximum number of sampled debug frames to save. 0 disables debug writes.",
    )
    parser.add_argument(
        "--debug-jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved debug frames.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    width = args.width
    height = args.height
    if width is None or height is None:
        width, height = ffprobe_stream_dimensions(
            url=args.url,
            transport=args.transport,
            tls_verify=args.tls_verify,
            ffprobe_bin=args.ffprobe_bin,
        )
        print(f"stream_dimensions width={width} height={height}")

    frame_count = 0
    started_at = time.monotonic()
    debug_writer = DebugFrameWriter(
        output_dir=args.debug_output_dir,
        sample_every=args.debug_sample_every,
        max_saved=args.debug_max_saved,
        jpeg_quality=args.debug_jpeg_quality,
    )

    with FFmpegVideoStream(
        url=args.url,
        width=width,
        height=height,
        transport=args.transport,
        tls_verify=args.tls_verify,
        ffmpeg_bin=args.ffmpeg_bin,
    ) as stream:
        while True:
            frame, error = stream.read()
            if frame is None:
                if error:
                    print(f"stream ended: {error}", file=sys.stderr)
                else:
                    print("stream ended before a full frame was read", file=sys.stderr)
                return 1

            frame_count += 1
            elapsed = max(time.monotonic() - started_at, 1e-6)
            fps = frame_count / elapsed

            if args.print_every and frame_count % args.print_every == 0:
                print(
                    f"frame={frame_count} shape={frame.shape} dtype={frame.dtype} approx_fps={fps:.2f}"
                )

            if debug_writer.should_save(frame_count):
                debug_writer.save(frame, frame_count, elapsed)

            # Placeholder for downstream perception work. Replace this block with model inference.
            if args.display:
                overlay = frame.copy()
                cv2.putText(
                    overlay,
                    f"frame {frame_count}  {fps:.2f} fps",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(args.window_name, overlay)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if args.max_frames and frame_count >= args.max_frames:
                break
            if debug_writer.enabled() and debug_writer.max_saved > 0:
                if debug_writer.saved_count >= debug_writer.max_saved:
                    break

    if args.display:
        cv2.destroyAllWindows()

    print(f"completed frame_count={frame_count} saved_debug_frames={debug_writer.saved_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
