"""Planner contracts, runtime executor, and replay helpers.

This module defines the narrow runtime boundary around planners.

The intended execution model is planner-driven rather than event-pushed:
- the executor wakes the planner once per tick
- the planner solicits only the perceptions it needs through ``PlannerProxy``
- the planner emits normalized action requests through that same proxy
- the proxy records perception/action request-result pairs so the episode can
  be replayed later against the same planner interface

The core primitives here are therefore not an attempt at a universal world
model. They are the minimum normalized residue needed to:
- compare different planners against the same live services
- run those planners through the same surface in live and replay modes
- inspect, log, and regression-test planner behavior without coupling planner
  code directly to B-hyve, NWS, RTSP, or notification implementation details

The replay path is intentionally strict and planner-facing. It is well-suited
to structured scenario variation over normalized inputs, but it does not claim
to simulate the full world. In practice, this means playback is used to stress
planner behavior, while live mode and human inspection remain the higher-fidelity
way to vet the integrated system when camera realism matters.

``Event`` and ``Decision`` capture planner-facing observations and outputs.
``PerceptionRequest`` / ``PerceptionResult`` and ``ActionRequest`` /
``ActionResult`` capture the interaction boundary the executor can log and
replay. ``PlannerExecutor`` owns the tick loop; the live and replay proxies
make planner code agnostic to whether inputs are coming from real services or a
recorded trajectory.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Protocol


ActionName = Literal["water_on", "water_off", "no_op", "notify"]

_LOG = logging.getLogger(__name__)


def _summarize_kwargs(kwargs: Dict[str, Any]) -> str:
    """Render compact human-readable call arguments for runtime logs."""
    if not kwargs:
        return ""
    parts = [f"{key}={value!r}" for key, value in sorted(kwargs.items())]
    return ", ".join(parts)


def _perceptor_label(name: str) -> str:
    """Map perceptor ids to short human-facing labels."""
    return {
        "weather": "weather forecast",
        "precipitation": "recent rainfall",
        "irrigation": "irrigation status",
        "camera": "lawn camera",
    }.get(name, name)


def _actor_label(name: str, action: str) -> str:
    """Map actor/action pairs to short human-facing labels."""
    if name == "irrigation":
        return {
            "water_on": "start watering",
            "water_off": "stop watering",
            "cycle": "cycle watering",
        }.get(action, f"irrigation:{action}")
    if name == "notification":
        return "send Discord message"
    return f"{name}:{action}"


def _sanitize_for_log(value: Any) -> Any:
    """Remove device identifiers and precise location data from JSONL artifacts."""
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            if key in {"device_id", "id"}:
                continue
            if key in {"latitude", "longitude", "lat", "lon"}:
                continue
            if key == "coordinates":
                continue
            sanitized[key] = _sanitize_for_log(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]
    return value


@dataclass
class Event:
    """Canonical external observation unit exchanged with planners."""

    timestamp: datetime
    source: str
    type: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class Decision:
    """High-level planner decision summary for one planning episode."""

    timestamp: datetime
    planner: str
    action: ActionName
    rationale: str
    duration_seconds: int | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptionRequest:
    """Planner request for one perceptor call."""

    timestamp: datetime
    planner: str
    perceptor: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptionResult:
    """Recorded result of one perceptor call."""

    timestamp: datetime
    planner: str
    perceptor: str
    event: Event


@dataclass
class ActionRequest:
    """Normalized action request for one actor call."""

    timestamp: datetime
    planner: str
    actor: str
    action: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Recorded result of one actor call."""

    timestamp: datetime
    actor: str
    action: str
    status: Literal["dry_run", "executed", "failed"]
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerRunResult:
    """Residue from one planner episode."""

    timestamp: datetime
    planner: str
    decision: Decision
    trace: List[str] = field(default_factory=list)
    perception_count: int = 0
    action_count: int = 0


class PlannerProxy(Protocol):
    """Runtime surface planners use to solicit perception and emit actions."""

    async def perceive(self, perceptor: str, **kwargs: Any) -> Event:
        """Fetch one normalized perception event."""

    async def act(self, actor: str, action: str, **kwargs: Any) -> ActionResult:
        """Execute one actor action."""


class Actor(Protocol):
    """Executable action target used by the runtime."""

    name: str

    async def execute(self, action: str, **kwargs: Any) -> ActionResult:
        """Execute one named action."""


class BasePlanner(ABC):
    """Common planner base for one tick-driven planning episode."""

    name: str

    def __init__(self, *, tick_interval: timedelta = timedelta(minutes=1)) -> None:
        self.tick_interval = tick_interval
        self._trace: List[str] = []
        self.daylight_start_hour = int(os.getenv("IRRIGATION_DAYLIGHT_START_HOUR", "6"))
        self.daylight_end_hour = int(os.getenv("IRRIGATION_DAYLIGHT_END_HOUR", "18"))

    def log(self, message: str) -> None:
        self._trace.append(message)

    def _consume_trace(self) -> List[str]:
        trace = list(self._trace)
        self._trace.clear()
        return trace

    def _is_within_watering_window(self, now: datetime) -> bool:
        """Return true when local wall time is inside the daylight watering window."""
        local_now = now.astimezone()
        local_time = local_now.timetz().replace(tzinfo=None)
        start = time(self.daylight_start_hour, 0)
        end = time(self.daylight_end_hour, 0)
        if start <= end:
            return start <= local_time <= end
        return local_time >= start or local_time <= end

    def _watering_window_summary(self, now: datetime) -> Dict[str, Any]:
        """Return compact local wall-time context for planner decisions."""
        local_now = now.astimezone()
        return {
            "local_time": local_now.isoformat(),
            "daylight_start_hour": self.daylight_start_hour,
            "daylight_end_hour": self.daylight_end_hour,
            "within_watering_window": self._is_within_watering_window(now),
        }

    @abstractmethod
    async def run(self, proxy: PlannerProxy, *, now: datetime) -> PlannerRunResult:
        """Run one planning episode for the current tick."""


def event_to_dict(event: Event) -> Dict[str, Any]:
    """Serialize an Event for JSONL logging."""
    payload = asdict(event)
    payload["timestamp"] = event.timestamp.isoformat()
    return payload


def decision_to_dict(decision: Decision) -> Dict[str, Any]:
    """Serialize a Decision for JSONL logging."""
    payload = asdict(decision)
    payload["timestamp"] = decision.timestamp.isoformat()
    return payload


def perception_request_to_dict(request: PerceptionRequest) -> Dict[str, Any]:
    """Serialize a PerceptionRequest for JSONL logging."""
    payload = asdict(request)
    payload["timestamp"] = request.timestamp.isoformat()
    return payload


def perception_result_to_dict(result: PerceptionResult) -> Dict[str, Any]:
    """Serialize a PerceptionResult for JSONL logging."""
    payload = {
        "timestamp": result.timestamp.isoformat(),
        "planner": result.planner,
        "perceptor": result.perceptor,
        "event": event_to_dict(result.event),
    }
    return payload


def action_request_to_dict(request: ActionRequest) -> Dict[str, Any]:
    """Serialize an ActionRequest for JSONL logging."""
    payload = asdict(request)
    payload["timestamp"] = request.timestamp.isoformat()
    return payload


def action_result_to_dict(result: ActionResult) -> Dict[str, Any]:
    """Serialize an ActionResult for JSONL logging."""
    payload = asdict(result)
    payload["timestamp"] = result.timestamp.isoformat()
    return payload


def planner_run_result_to_dict(result: PlannerRunResult) -> Dict[str, Any]:
    """Serialize a PlannerRunResult for JSONL logging."""
    return {
        "timestamp": result.timestamp.isoformat(),
        "planner": result.planner,
        "decision": decision_to_dict(result.decision),
        "trace": result.trace,
        "perception_count": result.perception_count,
        "action_count": result.action_count,
    }


def event_from_dict(payload: Dict[str, Any]) -> Event:
    """Deserialize a logged event."""
    data = dict(payload)
    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
    return Event(**data)


def append_jsonl(path: str | Path, record_type: str, payload: Dict[str, Any]) -> None:
    """Append one record to a JSONL log, creating parent directories as needed."""
    record = _sanitize_for_log({"record_type": record_type, **payload})
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass(frozen=True)
class PerceptionConfig:
    """Shared configuration for service-backed perceptors."""

    lat: float | None = None
    lon: float | None = None
    infer_location: bool = False
    forecast_hours: int | None = None
    forecast_days: int | None = None
    precipitation_window: str = "24H"
    camera_url: str | None = None
    sample_frames: int = 3
    save_frame: str | None = None
    baseline_image_dir: str | None = None
    baseline_output: str | None = None
    baseline_visualization_output: str | None = None
    experiment_image_dir: str | None = None
    experiment_output_dir: str | None = None
    score_image: str | None = None
    score_image_dir: str | None = None
    score_visualization_output: str | None = None
    camera_backend: str = "classic"
    notification_message: str | None = None


class ServicePerceptionProxy:
    """Service-backed probe/perception surface for the CLI."""

    def __init__(self, config: PerceptionConfig) -> None:
        self._config = config

    async def probe_services(self, target: str) -> Dict[str, Any]:
        from perception import (
            ForecastPerceptor,
            HistoricalPrecipitationPerceptor,
            IrrigationPerceptor,
            camera_perceptor_for_backend,
        )
        from services.notification import DiscordWebhookClient

        results: Dict[str, Any] = {}
        location = await self._resolve_lat_lon()

        if target in {"weather", "all"}:
            if location is None:
                results["weather"] = {"error": "latitude/longitude unavailable"}
            else:
                results["weather"] = await ForecastPerceptor().probe_raw(
                    *location,
                    hours=self._config.forecast_hours,
                    days=self._config.forecast_days,
                )

        if target in {"precipitation", "all"}:
            results["precipitation"] = await HistoricalPrecipitationPerceptor().probe_raw(
                self._config.precipitation_window
            )

        if target in {"irrigation", "all"}:
            results["irrigation"] = await IrrigationPerceptor().probe_raw()

        if target in {"camera", "all"}:
            if (
                not self._config.camera_url
                and not self._config.baseline_image_dir
                and not self._config.score_image
                and not self._config.score_image_dir
            ):
                results["camera"] = {"error": "camera URL required"}
            else:
                results["camera"] = camera_perceptor_for_backend(self._config.camera_backend).probe_raw(
                    self._config.camera_url,
                    sample_frames=self._config.sample_frames,
                    save_frame_path=self._config.save_frame,
                    baseline_image_dir=self._config.baseline_image_dir,
                    baseline_output_path=self._config.baseline_output,
                    baseline_visualization_path=self._config.baseline_visualization_output,
                    experiment_image_dir=self._config.experiment_image_dir,
                    experiment_output_dir=self._config.experiment_output_dir,
                    score_image_path=self._config.score_image,
                    score_image_dir=self._config.score_image_dir,
                    score_visualization_path=self._config.score_visualization_output,
                )

        if target in {"notification", "all"}:
            try:
                async with DiscordWebhookClient.from_env() as client:
                    results["notification"] = await client.probe(
                        message=self._config.notification_message or "Notification probe from irrigation-agent."
                    )
            except Exception as err:
                results["notification"] = {"error": f"{type(err).__name__}: {err}"}

        return results

    async def probe_perception(self, target: str) -> Dict[str, Any]:
        """Run one or more perceptors and return normalized events."""
        results: Dict[str, Any] = {}
        for perceptor in self._targets_to_perceptors(target):
            try:
                event = await self.perceive(perceptor)
            except Exception as err:
                results[perceptor] = {"error": f"{type(err).__name__}: {err}"}
            else:
                results[perceptor] = event_to_dict(event)
        return results

    async def perceive(self, perceptor: str, **kwargs: Any) -> Event:
        """Dispatch one named perceptor."""
        from perception import (
            ForecastPerceptor,
            HistoricalPrecipitationPerceptor,
            IrrigationPerceptor,
            camera_perceptor_for_backend,
        )

        merged = self._merge_kwargs(kwargs)
        if perceptor == "weather":
            location = await self._resolve_lat_lon()
            if location is None:
                raise RuntimeError("latitude/longitude unavailable")
            return await ForecastPerceptor().perceive(
                *location,
                hours=merged["forecast_hours"],
                days=merged["forecast_days"],
            )
        if perceptor == "precipitation":
            return await HistoricalPrecipitationPerceptor().perceive(merged["window"])
        if perceptor == "irrigation":
            return await IrrigationPerceptor().perceive()
        if perceptor == "camera":
            camera_url = merged["camera_url"]
            if (
                not camera_url
                and not merged["baseline_image_dir"]
                and not merged["score_image"]
                and not merged["score_image_dir"]
            ):
                raise RuntimeError("camera URL required")
            return camera_perceptor_for_backend(merged["camera_backend"]).perceive(
                camera_url,
                sample_frames=merged["sample_frames"],
                save_frame_path=merged["save_frame"],
                baseline_image_dir=merged["baseline_image_dir"],
                baseline_output_path=merged["baseline_output"],
                baseline_visualization_path=merged["baseline_visualization_output"],
                experiment_image_dir=merged["experiment_image_dir"],
                experiment_output_dir=merged["experiment_output_dir"],
                score_image_path=merged["score_image"],
                score_image_dir=merged["score_image_dir"],
                score_visualization_path=merged["score_visualization_output"],
            )
        raise ValueError(f"Unknown perceptor: {perceptor}")

    def _merge_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Overlay per-call overrides on top of the shared config."""
        return {
            "forecast_hours": kwargs.get("forecast_hours", self._config.forecast_hours),
            "forecast_days": kwargs.get("forecast_days", self._config.forecast_days),
            "window": kwargs.get("window", self._config.precipitation_window),
            "camera_url": kwargs.get("camera_url", self._config.camera_url),
            "sample_frames": kwargs.get("sample_frames", self._config.sample_frames),
            "save_frame": kwargs.get("save_frame", self._config.save_frame),
            "baseline_image_dir": kwargs.get("baseline_image_dir", self._config.baseline_image_dir),
            "baseline_output": kwargs.get("baseline_output", self._config.baseline_output),
            "baseline_visualization_output": kwargs.get(
                "baseline_visualization_output", self._config.baseline_visualization_output
            ),
            "experiment_image_dir": kwargs.get("experiment_image_dir", self._config.experiment_image_dir),
            "experiment_output_dir": kwargs.get("experiment_output_dir", self._config.experiment_output_dir),
            "score_image": kwargs.get("score_image", self._config.score_image),
            "score_image_dir": kwargs.get("score_image_dir", self._config.score_image_dir),
            "score_visualization_output": kwargs.get(
                "score_visualization_output", self._config.score_visualization_output
            ),
            "camera_backend": kwargs.get("camera_backend", self._config.camera_backend),
        }

    def _targets_to_perceptors(self, target: str) -> List[str]:
        """Expand CLI target names to concrete perceptors."""
        if target == "all":
            return ["weather", "precipitation", "irrigation", "camera"]
        return [target]

    async def _resolve_lat_lon(self) -> tuple[float, float] | None:
        """Resolve coordinates from flags or the configured B-hyve device."""
        if self._config.lat is not None and self._config.lon is not None:
            return (self._config.lat, self._config.lon)

        if self._config.infer_location:
            from services.bhyve.controller import controller_from_env

            async with controller_from_env() as controller:
                sprinkler = controller.list_sprinkler_devices()[0]
                location = sprinkler.get("location", {})
                coords = location.get("coordinates", [])
                if len(coords) == 2:
                    return (float(coords[1]), float(coords[0]))

        return None


class IrrigationActor:
    """Actor that controls the B-hyve hose timer."""

    name = "irrigation"

    async def execute(self, action: str, **kwargs: Any) -> ActionResult:
        now = dt.datetime.now(dt.timezone.utc)
        from services.bhyve.controller import controller_from_env

        if action == "water_on":
            duration_seconds = float(kwargs.get("duration_seconds") or 300)
            _LOG.info("Starting watering for %.0f seconds", duration_seconds)
            async with controller_from_env() as controller:
                sprinkler = controller.list_sprinkler_devices()[0]
                result = await controller.turn_on(
                    str(sprinkler["id"]),
                    seconds=duration_seconds,
                )
            _LOG.info("Watering command sent")
            return ActionResult(timestamp=now, actor=self.name, action=action, status="executed", detail={"result": result})

        if action == "water_off":
            _LOG.info("Stopping watering")
            async with controller_from_env() as controller:
                sprinkler = controller.list_sprinkler_devices()[0]
                result = await controller.turn_off(str(sprinkler["id"]))
            _LOG.info("Stop-watering command sent")
            return ActionResult(timestamp=now, actor=self.name, action=action, status="executed", detail={"result": result})

        if action == "cycle":
            duration_seconds = float(kwargs.get("duration_seconds") or 5)
            _LOG.info("Cycling watering for %.0f seconds", duration_seconds)
            async with controller_from_env() as controller:
                sprinkler = controller.list_sprinkler_devices()[0]
                result = await controller.cycle(
                    str(sprinkler["id"]),
                    seconds=duration_seconds,
                )
            _LOG.info("Cycle-watering command sent")
            return ActionResult(timestamp=now, actor=self.name, action=action, status="executed", detail={"result": result})

        raise ValueError(f"Unsupported irrigation action: {action}")


class NotificationActor:
    """Actor that delivers planner notifications through Discord."""

    name = "notification"

    async def execute(self, action: str, **kwargs: Any) -> ActionResult:
        now = dt.datetime.now(dt.timezone.utc)
        if action != "send":
            raise ValueError(f"Unsupported notification action: {action}")

        from services.notification import DiscordWebhookClient

        message = str(kwargs.get("message") or "").strip()
        if not message:
            raise ValueError("NotificationActor.send requires a non-empty message")

        metadata = kwargs.get("metadata")
        _LOG.info("Sending Discord message: %s", message[:120])
        async with DiscordWebhookClient.from_env() as client:
            result = await client.send(message, metadata=metadata if isinstance(metadata, dict) else None)
        _LOG.info("Discord message delivered")

        return ActionResult(
            timestamp=now,
            actor=self.name,
            action=action,
            status="executed",
            detail={"result": result},
        )


class LiveExecutorProxy:
    """Proxy handed to planners during live execution."""

    def __init__(
        self,
        *,
        planner_name: str,
        perceptor_registry: Dict[str, ServicePerceptionProxy],
        actor_registry: Dict[str, Actor],
        log_jsonl: str | None,
        execute_actions: bool,
        now: datetime,
    ) -> None:
        self._planner_name = planner_name
        self._perceptors = perceptor_registry
        self._actors = actor_registry
        self._log_jsonl = log_jsonl
        self._execute_actions = execute_actions
        self._now = now
        self.perception_count = 0
        self.action_count = 0

    async def perceive(self, perceptor: str, **kwargs: Any) -> Event:
        if perceptor not in self._perceptors:
            raise ValueError(f"Unknown perceptor: {perceptor}")
        details = _summarize_kwargs(kwargs)
        if details:
            _LOG.info("Asking for %s (%s)", _perceptor_label(perceptor), details)
        else:
            _LOG.info("Asking for %s", _perceptor_label(perceptor))
        _LOG.debug("Perception request: planner=%s perceptor=%s kwargs=%s", self._planner_name, perceptor, kwargs)
        request = PerceptionRequest(
            timestamp=dt.datetime.now(dt.timezone.utc),
            planner=self._planner_name,
            perceptor=perceptor,
            kwargs=kwargs,
        )
        self._append("perception_request", {"perception_request": perception_request_to_dict(request)})
        # The planner asks for a named perceptor; the live proxy records the
        # request/result pair so the same episode can be replayed later.
        event = await self._perceptors[perceptor].perceive(perceptor, **kwargs)
        result = PerceptionResult(
            timestamp=dt.datetime.now(dt.timezone.utc),
            planner=self._planner_name,
            perceptor=perceptor,
            event=event,
        )
        self._append("perception_result", {"perception_result": perception_result_to_dict(result)})
        self.perception_count += 1
        _LOG.debug(
            "Perception completed: planner=%s perceptor=%s event=%s/%s",
            self._planner_name,
            perceptor,
            event.source,
            event.type,
        )
        return event

    async def act(self, actor: str, action: str, **kwargs: Any) -> ActionResult:
        if actor not in self._actors:
            raise ValueError(f"Unknown actor: {actor}")
        details = _summarize_kwargs(kwargs)
        if details:
            _LOG.info("Action: %s (%s)", _actor_label(actor, action), details)
        else:
            _LOG.info("Action: %s", _actor_label(actor, action))
        _LOG.debug(
            "Action request: planner=%s actor=%s action=%s kwargs=%s",
            self._planner_name,
            actor,
            action,
            kwargs,
        )
        request = ActionRequest(
            timestamp=dt.datetime.now(dt.timezone.utc),
            planner=self._planner_name,
            actor=actor,
            action=action,
            kwargs=kwargs,
        )
        self._append("action_request", {"action_request": action_request_to_dict(request)})
        if not self._execute_actions:
            result = ActionResult(
                timestamp=dt.datetime.now(dt.timezone.utc),
                actor=actor,
                action=action,
                status="dry_run",
                detail={"reason": "action emission disabled", "request": kwargs},
            )
        else:
            result = await self._actors[actor].execute(action, **kwargs)
        self._append("action_result", {"action_result": action_result_to_dict(result)})
        self.action_count += 1
        _LOG.debug(
            "Action completed: planner=%s actor=%s action=%s status=%s",
            self._planner_name,
            actor,
            action,
            result.status,
        )
        return result

    def _append(self, record_type: str, payload: Dict[str, Any]) -> None:
        if self._log_jsonl:
            append_jsonl(self._log_jsonl, record_type, payload)


class ReplayExecutorProxy:
    """Proxy that replays previously logged perception/action calls.

    Replay is planner-facing rather than transcript-facing: a planner may ask
    for recorded observations in a different order than the live run that
    produced them. We therefore serve matching recorded interactions from a
    pool, refreshing timestamps as they are replayed so the planner sees a
    coherent current episode rather than a stale historical sequence.
    """

    def __init__(
        self,
        *,
        planner_name: str,
        records: List[Dict[str, Any]],
        execute_actions: bool,
        now: datetime,
    ) -> None:
        self._planner_name = planner_name
        self._execute_actions = execute_actions
        self.perception_count = 0
        self.action_count = 0
        self._replay_time = now
        self._perception_pairs = self._collect_pairs(records, "perception_request", "perception_result")
        self._action_pairs = self._collect_pairs(records, "action_request", "action_result")

    async def perceive(self, perceptor: str, **kwargs: Any) -> Event:
        pair = self._consume_perception_pair(perceptor, kwargs)
        result = pair["result"]["perception_result"]
        self.perception_count += 1
        event = event_from_dict(result["event"])
        event.timestamp = self._next_timestamp()
        return event

    async def act(self, actor: str, action: str, **kwargs: Any) -> ActionResult:
        pair = self._consume_action_pair(actor, action, kwargs)
        result = dict(pair["result"]["action_result"])
        result["timestamp"] = self._next_timestamp()
        if not self._execute_actions:
            result["status"] = "dry_run"
            result["detail"] = {"reason": "replay action emission disabled", "recorded": result["detail"]}
        self.action_count += 1
        return ActionResult(**result)

    @staticmethod
    def _collect_pairs(records: List[Dict[str, Any]], request_type: str, result_type: str) -> List[Dict[str, Any]]:
        """Pair recorded requests with their corresponding results."""
        pairs: List[Dict[str, Any]] = []
        pending_request: Dict[str, Any] | None = None
        for record in records:
            record_type = record.get("record_type")
            if record_type == request_type:
                pending_request = record
            elif record_type == result_type:
                if pending_request is None:
                    continue
                pairs.append({"request": pending_request, "result": record})
                pending_request = None
        return pairs

    def _consume_perception_pair(self, perceptor: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Return one matching recorded perception interaction."""
        return self._consume_pair(
            pool=self._perception_pairs,
            matcher=lambda pair: pair["request"]["perception_request"]["perceptor"] == perceptor,
            strict_matcher=lambda pair: (
                pair["request"]["perception_request"]["perceptor"] == perceptor
                and self._normalize_kwargs(pair["request"]["perception_request"].get("kwargs", {}))
                == self._normalize_kwargs(kwargs)
            ),
            label=f"perception {perceptor}",
        )

    def _consume_action_pair(self, actor: str, action: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Return one matching recorded action interaction."""
        return self._consume_pair(
            pool=self._action_pairs,
            matcher=lambda pair: (
                pair["request"]["action_request"]["actor"] == actor
                and pair["request"]["action_request"]["action"] == action
            ),
            strict_matcher=lambda pair: (
                pair["request"]["action_request"]["actor"] == actor
                and pair["request"]["action_request"]["action"] == action
                and self._normalize_kwargs(pair["request"]["action_request"].get("kwargs", {}))
                == self._normalize_kwargs(kwargs)
            ),
            label=f"action {actor}.{action}",
        )

    def _consume_pair(
        self,
        *,
        pool: List[Dict[str, Any]],
        matcher: Any,
        strict_matcher: Any,
        label: str,
    ) -> Dict[str, Any]:
        """Consume one matching request/result pair from a replay pool."""
        for index, pair in enumerate(pool):
            if strict_matcher(pair):
                return pool.pop(index)

        fallback_matches = [index for index, pair in enumerate(pool) if matcher(pair)]
        if len(fallback_matches) == 1:
            return pool.pop(fallback_matches[0])
        if len(fallback_matches) > 1:
            raise RuntimeError(f"Replay is ambiguous for {label}: multiple recorded candidates remain")
        raise RuntimeError(f"Replay log has no recorded candidate for {label}")

    @staticmethod
    def _normalize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize kwargs for replay matching."""
        return _sanitize_for_log(kwargs)

    def _next_timestamp(self) -> datetime:
        """Issue a fresh monotonic timestamp for replayed interactions."""
        self._replay_time = self._replay_time + timedelta(milliseconds=1)
        return self._replay_time


class PlannerExecutor:
    """Owns planner ticks, planner-driven proxy calls, action emission, and logging."""

    def __init__(
        self,
        *,
        planner: BasePlanner,
        perception_proxy: ServicePerceptionProxy | None = None,
        actor_registry: Dict[str, Actor] | None = None,
        log_jsonl: str | None = None,
    ) -> None:
        self._planner = planner
        self._perception_proxy = perception_proxy
        self._actor_registry = actor_registry or {
            "irrigation": IrrigationActor(),
            "notification": NotificationActor(),
        }
        self._log_jsonl = log_jsonl
        self._log_initialized = False

    def _prepare_log(self) -> None:
        """Start each live run with a fresh JSONL log file."""
        if not self._log_jsonl or self._log_initialized:
            return
        path = Path(self._log_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        self._log_initialized = True

    async def tick(self, tick_index: int, *, execute_actions: bool) -> Dict[str, Any]:
        if self._perception_proxy is None:
            raise RuntimeError("Live ticks require a perception proxy.")
        self._prepare_log()

        tick_time = dt.datetime.now(dt.timezone.utc)
        _LOG.info(
            "Tick %s started with %s planner%s",
            tick_index,
            self._planner.name,
            "" if execute_actions else " (actions disabled)",
        )
        tick_event = Event(
            timestamp=tick_time,
            source="runtime",
            type="tick",
            payload={"tick_index": tick_index},
        )
        self._append("tick", {"event": event_to_dict(tick_event)})

        proxy = LiveExecutorProxy(
            planner_name=self._planner.name,
            perceptor_registry={name: self._perception_proxy for name in ["weather", "precipitation", "irrigation", "camera"]},
            actor_registry=self._actor_registry,
            log_jsonl=self._log_jsonl,
            execute_actions=execute_actions,
            now=tick_time,
        )
        result = await self._planner.run(proxy, now=tick_time)
        self._append("planner_run", {"planner_run": planner_run_result_to_dict(result)})
        _LOG.info(
            "Tick %s decision: %s after %s perception call(s) and %s action(s): %s",
            tick_index,
            result.decision.action,
            result.perception_count,
            result.action_count,
            result.decision.rationale,
        )
        return planner_run_result_to_dict(result)

    async def run(
        self,
        *,
        tick_count: int,
        tick_seconds: float,
        execute_actions: bool,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for tick_index in range(tick_count):
            results.append(await self.tick(tick_index, execute_actions=execute_actions))
            if tick_index != tick_count - 1:
                _LOG.debug("Sleeping between ticks for %.3f seconds", tick_seconds)
                await asyncio.sleep(tick_seconds)
        return results

    async def replay(self, log_jsonl: str, *, execute_actions: bool) -> List[Dict[str, Any]]:
        """Replay each logged tick as one planner episode."""
        _LOG.info(
            "Replaying %s with %s planner%s",
            log_jsonl,
            self._planner.name,
            "" if execute_actions else " (actions disabled)",
        )
        records = [
            json.loads(line)
            for line in Path(log_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        episodes: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for record in records:
            if record.get("record_type") == "tick":
                if current:
                    episodes.append(current)
                current = [record]
            elif current:
                current.append(record)
        if current:
            episodes.append(current)

        results: List[Dict[str, Any]] = []
        for episode in episodes:
            tick_record = episode[0]
            tick_event = event_from_dict(tick_record["event"])
            _LOG.info("Replay tick at %s", tick_event.timestamp.isoformat())
            proxy = ReplayExecutorProxy(
                planner_name=self._planner.name,
                records=episode[1:],
                execute_actions=execute_actions,
                now=tick_event.timestamp,
            )
            result = await self._planner.run(proxy, now=tick_event.timestamp)
            results.append(planner_run_result_to_dict(result))
            _LOG.info(
                "Replay decision: %s after %s perception call(s) and %s action(s): %s",
                result.decision.action,
                result.perception_count,
                result.action_count,
                result.decision.rationale,
            )
        _LOG.info("Replay completed: %s episode(s)", len(results))
        return results

    def _append(self, record_type: str, payload: Dict[str, Any]) -> None:
        if self._log_jsonl:
            append_jsonl(self._log_jsonl, record_type, payload)
