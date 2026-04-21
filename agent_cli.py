#!/usr/bin/env python3
"""CLI entrypoint for service probes, planner runs, and replay."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from classical_agent import ClassicalPlanner
from neural_agent import NeuralPlanner
from planner import PerceptionConfig, PlannerExecutor, ServicePerceptionProxy


def _jsonable(value: Any) -> Any:
    """Normalize dataclasses and datetimes for JSON output."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _build_planner(name: str):
    """Construct a planner from its CLI name."""
    if name == "classical":
        return ClassicalPlanner()
    if name == "neural":
        return NeuralPlanner()
    raise ValueError(f"Unsupported planner: {name}")


def _perception_config_from_args(args: argparse.Namespace) -> PerceptionConfig:
    """Translate CLI flags into shared perceptor configuration."""
    return PerceptionConfig(
        lat=args.lat,
        lon=args.lon,
        infer_location=getattr(args, "infer_location", False),
        forecast_hours=getattr(args, "forecast_hours", None),
        forecast_days=getattr(args, "forecast_days", None),
        precipitation_window=getattr(args, "window", "24H"),
        camera_url=getattr(args, "camera_url", None),
        sample_frames=getattr(args, "sample_frames", 3),
        save_frame=getattr(args, "save_frame", None),
        notification_message=getattr(args, "message", None),
    )


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
    service.add_argument("target", choices=["all", "weather", "precipitation", "irrigation", "camera", "notification"])
    service.add_argument("--lat", type=float)
    service.add_argument("--lon", type=float)
    service.add_argument("--infer-location", action="store_true")
    service.add_argument("--forecast-hours", type=int)
    service.add_argument("--forecast-days", type=int)
    service.add_argument("--window", default="24H")
    service.add_argument("--camera-url")
    service.add_argument("--sample-frames", type=int, default=1)
    service.add_argument("--save-frame")
    service.add_argument("--message")

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
    """Dispatch one CLI command."""
    if hasattr(args, "forecast_hours") and hasattr(args, "forecast_days"):
        if args.forecast_hours is not None and args.forecast_days is not None:
            raise SystemExit("Specify either --forecast-hours or --forecast-days, not both.")
    if args.command == "probe" and args.probe_target == "service":
        proxy = ServicePerceptionProxy(_perception_config_from_args(args))
        print(json.dumps(_jsonable(await proxy.probe_services(args.target)), indent=2, sort_keys=True))
        return 0
    if args.command == "probe" and args.probe_target == "perception":
        proxy = ServicePerceptionProxy(_perception_config_from_args(args))
        print(json.dumps(_jsonable(await proxy.probe_perception(args.target)), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        planner = _build_planner(args.planner)
        executor = PlannerExecutor(
            planner=planner,
            perception_proxy=ServicePerceptionProxy(_perception_config_from_args(args)),
            log_jsonl=args.log_jsonl,
        )
        tick_count = args.ticks if args.ticks is not None else max(1, int(args.duration_seconds / args.tick_seconds))
        results = await executor.run(
            tick_count=tick_count,
            tick_seconds=args.tick_seconds,
            execute_actions=args.execute_actions,
        )
        for result in results:
            print(json.dumps({"planner_run": result}, indent=2, sort_keys=True))
        return 0
    if args.command == "replay":
        planner = _build_planner(args.planner)
        executor = PlannerExecutor(planner=planner)
        results = await executor.replay(args.log_jsonl, execute_actions=args.execute_actions)
        for result in results:
            print(json.dumps({"planner_run": result}, indent=2, sort_keys=True))
        return 0
    raise SystemExit(f"Unsupported command: {args.command}")


def main() -> int:
    """Run the CLI synchronously."""
    parser = _build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 2
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
