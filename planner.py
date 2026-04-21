"""Planner contracts, runtime executor, and replay helpers."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Protocol


ActionName = Literal["water_on", "water_off", "no_op", "notify"]


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

    def log(self, message: str) -> None:
        self._trace.append(message)

    def _consume_trace(self) -> List[str]:
        trace = list(self._trace)
        self._trace.clear()
        return trace

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
    record = {"record_type": record_type, **payload}
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
            SecurityCameraPerceptor,
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
            if not self._config.camera_url:
                results["camera"] = {"error": "camera URL required"}
            else:
                results["camera"] = SecurityCameraPerceptor().probe_raw(
                    self._config.camera_url,
                    sample_frames=self._config.sample_frames,
                    save_frame_path=self._config.save_frame,
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
            SecurityCameraPerceptor,
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
            if not camera_url:
                raise RuntimeError("camera URL required")
            return SecurityCameraPerceptor().perceive(
                camera_url,
                sample_frames=merged["sample_frames"],
                save_frame_path=merged["save_frame"],
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
            async with controller_from_env() as controller:
                sprinkler = controller.list_sprinkler_devices()[0]
                result = await controller.turn_on(
                    str(sprinkler["id"]),
                    seconds=float(kwargs.get("duration_seconds") or 300),
                )
            return ActionResult(timestamp=now, actor=self.name, action=action, status="executed", detail={"result": result})

        if action == "water_off":
            async with controller_from_env() as controller:
                sprinkler = controller.list_sprinkler_devices()[0]
                result = await controller.turn_off(str(sprinkler["id"]))
            return ActionResult(timestamp=now, actor=self.name, action=action, status="executed", detail={"result": result})

        if action == "cycle":
            async with controller_from_env() as controller:
                sprinkler = controller.list_sprinkler_devices()[0]
                result = await controller.cycle(
                    str(sprinkler["id"]),
                    seconds=float(kwargs.get("duration_seconds") or 5),
                )
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
        async with DiscordWebhookClient.from_env() as client:
            result = await client.send(message, metadata=metadata if isinstance(metadata, dict) else None)

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
        return event

    async def act(self, actor: str, action: str, **kwargs: Any) -> ActionResult:
        if actor not in self._actors:
            raise ValueError(f"Unknown actor: {actor}")
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
        return result

    def _append(self, record_type: str, payload: Dict[str, Any]) -> None:
        if self._log_jsonl:
            append_jsonl(self._log_jsonl, record_type, payload)


class ReplayExecutorProxy:
    """Proxy that replays previously logged perception/action calls."""

    def __init__(self, *, planner_name: str, records: List[Dict[str, Any]], execute_actions: bool) -> None:
        self._planner_name = planner_name
        self._records = records
        self._execute_actions = execute_actions
        self._index = 0
        self.perception_count = 0
        self.action_count = 0

    async def perceive(self, perceptor: str, **kwargs: Any) -> Event:
        # Replay is strict about call order so planner behavior remains
        # comparable between live execution and offline evaluation.
        request_record = self._next_record("perception_request")
        request = request_record["perception_request"]
        if request["perceptor"] != perceptor:
            raise RuntimeError(f"Replay mismatch: expected perceptor {request['perceptor']}, got {perceptor}")
        result_record = self._next_record("perception_result")
        result = result_record["perception_result"]
        self.perception_count += 1
        return event_from_dict(result["event"])

    async def act(self, actor: str, action: str, **kwargs: Any) -> ActionResult:
        request_record = self._next_record("action_request")
        request = request_record["action_request"]
        if request["actor"] != actor or request["action"] != action:
            raise RuntimeError(
                f"Replay mismatch: expected {request['actor']}.{request['action']}, got {actor}.{action}"
            )
        result_record = self._next_record("action_result")
        result = dict(result_record["action_result"])
        result["timestamp"] = datetime.fromisoformat(result["timestamp"])
        if not self._execute_actions:
            result["status"] = "dry_run"
            result["detail"] = {"reason": "replay action emission disabled", "recorded": result["detail"]}
        self.action_count += 1
        return ActionResult(**result)

    def _next_record(self, record_type: str) -> Dict[str, Any]:
        while self._index < len(self._records):
            record = self._records[self._index]
            self._index += 1
            if record.get("record_type") == record_type:
                return record
        raise RuntimeError(f"Replay log exhausted while looking for {record_type}")


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

    async def tick(self, tick_index: int, *, execute_actions: bool) -> Dict[str, Any]:
        if self._perception_proxy is None:
            raise RuntimeError("Live ticks require a perception proxy.")

        tick_time = dt.datetime.now(dt.timezone.utc)
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
                await asyncio.sleep(tick_seconds)
        return results

    async def replay(self, log_jsonl: str, *, execute_actions: bool) -> List[Dict[str, Any]]:
        """Replay each logged tick as one planner episode."""
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
            proxy = ReplayExecutorProxy(
                planner_name=self._planner.name,
                records=episode[1:],
                execute_actions=execute_actions,
            )
            result = await self._planner.run(proxy, now=tick_event.timestamp)
            results.append(planner_run_result_to_dict(result))
        return results

    def _append(self, record_type: str, payload: Dict[str, Any]) -> None:
        if self._log_jsonl:
            append_jsonl(self._log_jsonl, record_type, payload)
