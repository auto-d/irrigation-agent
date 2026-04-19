#!/usr/bin/env python3
"""Unified MVP CLI for probing services, probing perception, running, and replaying."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from classical_agent import ClassicalPlanner
from neural_agent import NeuralMvpPlanner
from perception import (
    ForecastPerceptor,
    HistoricalPrecipitationPerceptor,
    IrrigationPerceptor,
    SecurityCameraPerceptor,
)
from planner import (
    ActionResult,
    Decision,
    Event,
    action_result_to_dict,
    append_jsonl,
    decision_to_dict,
    event_from_dict,
    event_to_dict,
)
from services.bhyve.controller import controller_from_env


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


async def _resolve_lat_lon(args: argparse.Namespace) -> tuple[float, float] | None:
    if args.lat is not None and args.lon is not None:
        return (args.lat, args.lon)

    if getattr(args, "infer_location", False):
        async with controller_from_env() as controller:
            sprinkler = controller.list_sprinkler_devices()[0]
            location = sprinkler.get("location", {})
            coords = location.get("coordinates", [])
            if len(coords) == 2:
                return (float(coords[1]), float(coords[0]))
    return None


def _build_planner(name: str):
    if name == "classical":
        return ClassicalPlanner()
    if name == "neural":
        return NeuralMvpPlanner()
    raise ValueError(f"Unsupported planner: {name}")


async def _probe_service(args: argparse.Namespace) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    location = await _resolve_lat_lon(args)

    if args.target in {"weather", "all"}:
        if location is None:
            results["weather"] = {"error": "latitude/longitude unavailable"}
        else:
            results["weather"] = await ForecastPerceptor().probe_raw(
                *location,
                hours=args.forecast_hours,
                days=args.forecast_days,
            )

    if args.target in {"precipitation", "all"}:
        results["precipitation"] = await HistoricalPrecipitationPerceptor().probe_raw(args.window)

    if args.target in {"irrigation", "all"}:
        results["irrigation"] = await IrrigationPerceptor().probe_raw()

    if args.target in {"camera", "all"}:
        if not args.camera_url:
            results["camera"] = {"error": "camera URL required"}
        else:
            results["camera"] = SecurityCameraPerceptor().probe_raw(
                args.camera_url,
                sample_frames=args.sample_frames,
                save_frame_path=args.save_frame,
            )

    return results


async def _probe_perception(args: argparse.Namespace) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    location = await _resolve_lat_lon(args)

    if args.target in {"weather", "all"}:
        if location is None:
            results["weather"] = {"error": "latitude/longitude unavailable"}
        else:
            event = await ForecastPerceptor().perceive(
                *location,
                hours=args.forecast_hours,
                days=args.forecast_days,
            )
            results["weather"] = event_to_dict(event)

    if args.target in {"precipitation", "all"}:
        event = await HistoricalPrecipitationPerceptor().perceive(args.window)
        results["precipitation"] = event_to_dict(event)

    if args.target in {"irrigation", "all"}:
        event = await IrrigationPerceptor().perceive()
        results["irrigation"] = event_to_dict(event)

    if args.target in {"camera", "all"}:
        if not args.camera_url:
            results["camera"] = {"error": "camera URL required"}
        else:
            event = SecurityCameraPerceptor().perceive(
                args.camera_url,
                sample_frames=args.sample_frames,
                save_frame_path=args.save_frame,
            )
            results["camera"] = event_to_dict(event)

    return results


async def _collect_perception_events(args: argparse.Namespace) -> List[Event]:
    events: List[Event] = []
    location = await _resolve_lat_lon(args)

    if location is not None:
        events.append(
            await ForecastPerceptor().perceive(
                *location,
                hours=args.forecast_hours,
                days=args.forecast_days,
            )
        )

    events.append(await HistoricalPrecipitationPerceptor().perceive(args.window))
    events.append(await IrrigationPerceptor().perceive())

    if args.camera_url:
        events.append(SecurityCameraPerceptor().perceive(args.camera_url, sample_frames=args.sample_frames))

    return events


async def _emit_action(decision: Decision, *, execute_actions: bool) -> ActionResult:
    now = dt.datetime.now(dt.timezone.utc)

    if not execute_actions:
        return ActionResult(
            timestamp=now,
            action=decision.action,
            status="dry_run",
            detail={"reason": "action emission disabled"},
        )

    if decision.action == "water_on":
        async with controller_from_env() as controller:
            sprinkler = controller.list_sprinkler_devices()[0]
            result = await controller.turn_on(
                str(sprinkler["id"]),
                seconds=float(decision.duration_seconds or 300),
            )
        return ActionResult(
            timestamp=now,
            action=decision.action,
            status="executed",
            detail={"result": result},
        )

    if decision.action == "water_off":
        async with controller_from_env() as controller:
            sprinkler = controller.list_sprinkler_devices()[0]
            result = await controller.turn_off(str(sprinkler["id"]))
        return ActionResult(
            timestamp=now,
            action=decision.action,
            status="executed",
            detail={"result": result},
        )

    return ActionResult(
        timestamp=now,
        action=decision.action,
        status="dry_run",
        detail={"reason": "no external action required"},
    )


def _append_records(log_path: str | None, records: Iterable[tuple[str, Dict[str, Any]]]) -> None:
    if not log_path:
        return
    for record_type, payload in records:
        append_jsonl(log_path, record_type, payload)


async def _run_live(args: argparse.Namespace) -> None:
    planner = _build_planner(args.planner)
    tick_count = args.ticks if args.ticks is not None else max(1, int(args.duration_seconds / args.tick_seconds))

    for tick_index in range(tick_count):
        events = await _collect_perception_events(args)
        tick_event = Event(
            timestamp=dt.datetime.now(dt.timezone.utc),
            source="runtime",
            type="tick",
            payload={"tick_index": tick_index},
        )
        events.append(tick_event)

        for event in events:
            planner.observe(event)
            _append_records(args.log_jsonl, [("event", {"event": event_to_dict(event)})])

        decision = planner.decide(tick_event.timestamp)
        print(json.dumps({"decision": decision_to_dict(decision)}, indent=2, sort_keys=True))
        _append_records(args.log_jsonl, [("decision", {"decision": decision_to_dict(decision)})])

        result = await _emit_action(decision, execute_actions=args.execute_actions)
        print(json.dumps({"action_result": action_result_to_dict(result)}, indent=2, sort_keys=True))
        _append_records(
            args.log_jsonl,
            [("action_result", {"action_result": action_result_to_dict(result)})],
        )

        if tick_index != tick_count - 1:
            await asyncio.sleep(args.tick_seconds)


async def _run_replay(args: argparse.Namespace) -> None:
    planner = _build_planner(args.planner)
    records = Path(args.log_jsonl).read_text(encoding="utf-8").splitlines()
    for line in records:
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record_type") != "event":
            continue
        event = event_from_dict(record["event"])
        planner.observe(event)
        if event.type == "tick":
            decision = planner.decide(event.timestamp)
            print(json.dumps({"decision": decision_to_dict(decision)}, indent=2, sort_keys=True))
            result = await _emit_action(decision, execute_actions=args.execute_actions)
            print(json.dumps({"action_result": action_result_to_dict(result)}, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Examples:\n"
            "  python agent_cli.py probe service all --infer-location\n"
            "  python agent_cli.py probe perception irrigation\n"
            "  python agent_cli.py run --planner classical --infer-location --log-jsonl trajectory.jsonl\n"
            "  python agent_cli.py replay --planner neural --log-jsonl trajectory.jsonl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    probe = subparsers.add_parser("probe")
    probe_sub = probe.add_subparsers(dest="probe_target", required=True)

    service = probe_sub.add_parser("service")
    service.add_argument("target", choices=["all", "weather", "precipitation", "irrigation", "camera"])
    service.add_argument("--lat", type=float)
    service.add_argument("--lon", type=float)
    service.add_argument("--infer-location", action="store_true")
    service.add_argument("--forecast-hours", type=int)
    service.add_argument("--forecast-days", type=int)
    service.add_argument("--window", default="24H")
    service.add_argument("--camera-url")
    service.add_argument("--sample-frames", type=int, default=1)
    service.add_argument("--save-frame")

    perception = probe_sub.add_parser("perception")
    perception.add_argument("target", choices=["all", "weather", "precipitation", "irrigation", "camera"])
    perception.add_argument("--lat", type=float)
    perception.add_argument("--lon", type=float)
    perception.add_argument("--infer-location", action="store_true")
    perception.add_argument("--forecast-hours", type=int)
    perception.add_argument("--forecast-days", type=int)
    perception.add_argument("--window", default="24H")
    perception.add_argument("--camera-url")
    perception.add_argument("--sample-frames", type=int, default=3)
    perception.add_argument("--save-frame")

    run = subparsers.add_parser("run")
    run.add_argument("--planner", choices=["classical", "neural"], required=True)
    run.add_argument("--lat", type=float)
    run.add_argument("--lon", type=float)
    run.add_argument("--infer-location", action="store_true")
    run.add_argument("--forecast-hours", type=int)
    run.add_argument("--forecast-days", type=int)
    run.add_argument("--window", default="24H")
    run.add_argument("--camera-url")
    run.add_argument("--sample-frames", type=int, default=3)
    run.add_argument("--tick-seconds", type=float, default=60.0)
    run.add_argument("--ticks", type=int)
    run.add_argument("--duration-seconds", type=float, default=300.0)
    run.add_argument("--log-jsonl")
    run.add_argument("--execute-actions", action="store_true")

    replay = subparsers.add_parser("replay")
    replay.add_argument("--planner", choices=["classical", "neural"], required=True)
    replay.add_argument("--log-jsonl", required=True)
    replay.add_argument("--execute-actions", action="store_true")

    return parser


async def _main_async(args: argparse.Namespace) -> int:
    if hasattr(args, "forecast_hours") and hasattr(args, "forecast_days"):
        if args.forecast_hours is not None and args.forecast_days is not None:
            raise SystemExit("Specify either --forecast-hours or --forecast-days, not both.")
    if args.command == "probe" and args.probe_target == "service":
        print(json.dumps(_jsonable(await _probe_service(args)), indent=2, sort_keys=True))
        return 0
    if args.command == "probe" and args.probe_target == "perception":
        print(json.dumps(_jsonable(await _probe_perception(args)), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        await _run_live(args)
        return 0
    if args.command == "replay":
        await _run_replay(args)
        return 0
    raise SystemExit(f"Unsupported command: {args.command}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 2
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
