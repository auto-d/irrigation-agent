"""Neural camera perception backed by the shared LLM backend."""

from __future__ import annotations

import base64
import datetime as dt
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import cv2

from llm_backend import Backend
from planner import Event
from services.rtsp.ingest import FFmpegVideoStream, ffprobe_stream_info


def _now() -> dt.datetime:
    """Return the current UTC timestamp."""
    return dt.datetime.now(dt.timezone.utc)


class NeuralSecurityCameraPerceptor:
    """Classify one lawn-camera still with a multimodal model."""

    DEFAULT_MODEL = os.getenv("IRRIGATION_VISION_MODEL", "gpt-5.4-mini")
    OUTPUT_SCHEMA: Dict[str, Any] = {
        "name": "lawn_camera_classification",
        "schema": {
            "type": "object",
            "properties": {
                "scene_summary": {"type": "string"},
                "person_detected": {"type": "boolean"},
                "animal_detected": {"type": "boolean"},
                "lawn_mower_active": {"type": "boolean"},
                "obstacle_detected": {"type": "boolean"},
                "watering_safe": {"type": "boolean"},
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "scene_summary",
                "person_detected",
                "animal_detected",
                "lawn_mower_active",
                "obstacle_detected",
                "watering_safe",
                "confidence",
                "evidence",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }

    def __init__(self, *, model: str | None = None) -> None:
        self.model = model or self.DEFAULT_MODEL
        self._backend: Backend | None = None

    def probe_raw(
        self,
        url: str | None = None,
        *,
        sample_frames: int = 1,
        save_frame_path: str | None = None,
        score_image_path: str | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Return the captured still plus the raw model classification."""
        frame_path, cleanup_path, image_meta = self._resolve_frame_input(
            url=url,
            sample_frames=sample_frames,
            save_frame_path=save_frame_path,
            score_image_path=score_image_path,
        )
        try:
            classification = self._classify_image(frame_path)
        finally:
            if cleanup_path is not None:
                Path(cleanup_path).unlink(missing_ok=True)
                image_meta["saved_frame_path"] = None
        return {
            "mode": "neural_classification",
            "camera_backend": "neural",
            "image": image_meta,
            "classification": classification,
            "model": self.model,
        }

    def perceive(
        self,
        url: str | None = None,
        *,
        sample_frames: int = 1,
        save_frame_path: str | None = None,
        score_image_path: str | None = None,
        **_: Any,
    ) -> Event:
        """Return a normalized planner-facing scene event."""
        raw = self.probe_raw(
            url,
            sample_frames=sample_frames,
            save_frame_path=save_frame_path,
            score_image_path=score_image_path,
        )
        classification = raw["classification"]
        payload = {
            "camera_backend": "neural",
            "model": raw["model"],
            "saved_frame_path": raw["image"].get("saved_frame_path"),
            "image_width": raw["image"].get("width"),
            "image_height": raw["image"].get("height"),
            "scene_summary": classification["scene_summary"],
            "person_detected": classification["person_detected"],
            "animal_detected": classification["animal_detected"],
            "lawn_mower_active": classification["lawn_mower_active"],
            "obstacle_detected": classification["obstacle_detected"],
            "watering_safe": classification["watering_safe"],
            "confidence": classification["confidence"],
            "evidence": classification["evidence"],
            "interpretation": "multimodal_scene_classification",
        }
        return Event(timestamp=_now(), source="camera", type="scene_activity", payload=payload)

    def _resolve_frame_input(
        self,
        *,
        url: str | None,
        sample_frames: int,
        save_frame_path: str | None,
        score_image_path: str | None,
    ) -> tuple[str, str | None, Dict[str, Any]]:
        """Resolve a local image path either from disk or from the RTSP feed."""
        if score_image_path:
            image_path = Path(score_image_path).expanduser()
            if not image_path.is_file():
                raise RuntimeError(f"camera image not found: {image_path}")
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"unable to read camera image: {image_path}")
            height, width = frame.shape[:2]
            return str(image_path), None, {
                "input_source": "still_image",
                "saved_frame_path": str(image_path),
                "width": width,
                "height": height,
            }

        if not url:
            raise RuntimeError("camera URL or still image required")

        stream_info = ffprobe_stream_info(url)
        frame = self._capture_frame(
            url,
            width=int(stream_info["width"]),
            height=int(stream_info["height"]),
            sample_frames=max(1, sample_frames),
        )
        if self._save_frame_requested(save_frame_path):
            saved_path = self._write_frame(frame, save_frame_path)
            cleanup_path = None
        else:
            saved_path, cleanup_path = self._write_temp_frame(frame)
        image_meta = {
            "input_source": "rtsp",
            "saved_frame_path": saved_path,
            "temporary_frame": cleanup_path is not None,
            "width": int(stream_info["width"]),
            "height": int(stream_info["height"]),
        }
        return saved_path, cleanup_path, image_meta

    def _classify_image(self, image_path: str) -> Dict[str, Any]:
        """Ask the multimodal model for a constrained scene classification."""
        image_data_url = self._image_as_data_url(image_path)
        classification = self._backend_from_env().structured_image_completion(
            system_prompt=(
                "You inspect a lawn camera for irrigation safety. "
                "Classify whether a person, animal, mower, or other obstacle is present on the lawn or in the watering path. "
                "A garden hose supplying water to the sprinkler and the watering tripod or sprinkler head itself are normal parts "
                "of the irrigation setup and should not be treated as hazards on their own. "
                "Ignore normal fixed background context such as patios, pools, patio furniture, fences, trees, and landscaping "
                "unless they indicate an active hazard or clearly occupy the irrigated lawn area. "
                "Set watering_safe to false whenever watering would be unsafe or materially ambiguous."
            ),
            user_prompt=(
                "Inspect this lawn still and return only the requested structured classification. "
                "Report obstacle_detected as true only when something materially relevant to watering is on the lawn, crossing the spray path, "
                "or would make watering unsafe right now. "
                "Do not mark obstacle_detected true for ordinary fixed scene context at the perimeter or patio area. "
                "Treat humans, pets, wildlife, active mower activity, toys left on the grass, ladders, "
                "or movable equipment on the grass as obstacles when they materially block safe watering. "
                "Do not treat the irrigation hose, the sprinkler tripod, or the sprinkler head as obstacles merely because they are visible; "
                "those are expected irrigation components unless they appear damaged, misplaced, or create a distinct trip or entanglement hazard."
            ),
            image_url=image_data_url,
            schema=self.OUTPUT_SCHEMA,
        )
        if not isinstance(classification.get("evidence"), list):
            classification["evidence"] = []
        classification["confidence"] = float(classification.get("confidence", 0.0))
        return classification

    def _backend_from_env(self) -> Backend:
        """Initialize the shared LLM backend on demand."""
        if self._backend is not None:
            return self._backend
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for neural camera perception")
        self._backend = Backend(api_key=api_key, model=self.model)
        return self._backend

    @staticmethod
    def _capture_frame(url: str, *, width: int, height: int, sample_frames: int) -> Any:
        """Capture one representative frame after advancing through the stream."""
        last_frame = None
        with FFmpegVideoStream(url, width, height).start() as stream:
            for _ in range(sample_frames):
                frame, error = stream.read()
                if frame is None:
                    raise RuntimeError(f"Unable to read RTSP frame: {error or 'unknown error'}")
                last_frame = frame
        if last_frame is None:
            raise RuntimeError("Unable to capture camera frame")
        return last_frame

    @staticmethod
    def _write_frame(frame: Any, path: str) -> str:
        """Write one captured frame to disk."""
        output = Path(path).expanduser()
        if not output.suffix:
            output = output.with_suffix(".jpg")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), frame):
            raise RuntimeError(f"failed to write captured frame to {output}")
        return str(output)

    @staticmethod
    def _save_frame_requested(path: str | None) -> bool:
        """Treat common planner sentinels as requests not to persist a frame."""
        if path is None:
            return False
        normalized = path.strip().lower()
        return normalized not in {"", "none", "no", "never", "off", "false", "null"}

    @classmethod
    def _write_temp_frame(cls, frame: Any) -> tuple[str, str]:
        """Persist one temporary frame for model submission."""
        with NamedTemporaryFile(prefix="irrigation-agent-camera-", suffix=".jpg", delete=False) as handle:
            temp_path = handle.name
        return cls._write_frame(frame, temp_path), temp_path

    @staticmethod
    def _image_as_data_url(path: str) -> str:
        """Encode a local image as a data URL for multimodal input."""
        image_path = Path(path).expanduser()
        suffix = image_path.suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
