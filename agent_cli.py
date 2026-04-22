#!/usr/bin/env python3
"""CLI entrypoint for service probes, planner runs, and replay."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from classical_agent import ClassicalPlanner
from neural_agent import NeuralPlanner
from planner import PerceptionConfig, PlannerExecutor, ServicePerceptionProxy

_LOG = logging.getLogger(__name__)


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


def _planner_names(selection: str) -> list[str]:
    """Expand CLI planner selection into concrete planner names."""
    if selection == "both":
        return ["classical", "neural"]
    return [selection]


def _perception_config_from_args(args: argparse.Namespace) -> PerceptionConfig:
    """Translate CLI flags into shared perceptor configuration."""
    camera_backend = getattr(args, "camera_backend", None)
    if camera_backend is None and getattr(args, "planner", None) == "neural":
        camera_backend = "neural"
    if camera_backend is None:
        camera_backend = "classic"
    return PerceptionConfig(
        lat=args.lat,
        lon=args.lon,
        infer_location=getattr(args, "infer_location", False),
        suppress_irrigation_state=getattr(args, "suppress_irrigation_state", False),
        forecast_hours=getattr(args, "forecast_hours", None),
        forecast_days=getattr(args, "forecast_days", None),
        precipitation_window=getattr(args, "window", "24H"),
        camera_url=getattr(args, "camera_url", None),
        sample_frames=getattr(args, "sample_frames", 3),
        save_frame=getattr(args, "save_frame", None),
        baseline_image_dir=getattr(args, "baseline_image_dir", None),
        baseline_output=getattr(args, "baseline_output", None),
        baseline_visualization_output=getattr(args, "baseline_viz_output", None),
        experiment_image_dir=getattr(args, "experiment_image_dir", None),
        experiment_output_dir=getattr(args, "experiment_output_dir", None),
        score_image=getattr(args, "score_image", None),
        score_image_dir=getattr(args, "score_image_dir", None),
        score_visualization_output=getattr(args, "score_viz_output", None),
        camera_backend=camera_backend,
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
    service.add_argument("--suppress-irrigation-state", action="store_true")
    service.add_argument("--forecast-hours", type=int)
    service.add_argument("--forecast-days", type=int)
    service.add_argument("--window", default="24H")
    service.add_argument("--camera-url")
    service.add_argument("--camera-backend", choices=["classic", "neural"])
    service.add_argument("--sample-frames", type=int, default=1)
    service.add_argument("--save-frame")
    service.add_argument("--baseline-image-dir")
    service.add_argument("--baseline-output")
    service.add_argument("--baseline-viz-output")
    service.add_argument("--experiment-image-dir")
    service.add_argument("--experiment-output-dir")
    service.add_argument("--score-image")
    service.add_argument("--score-image-dir")
    service.add_argument("--score-viz-output")
    service.add_argument("--message")

    perception = probe_sub.add_parser("perception")
    perception.add_argument("target", choices=["all", "weather", "precipitation", "irrigation", "camera"])
    perception.add_argument("--lat", type=float)
    perception.add_argument("--lon", type=float)
    perception.add_argument("--infer-location", action="store_true")
    perception.add_argument("--suppress-irrigation-state", action="store_true")
    perception.add_argument("--forecast-hours", type=int)
    perception.add_argument("--forecast-days", type=int)
    perception.add_argument("--window", default="24H")
    perception.add_argument("--camera-url")
    perception.add_argument("--camera-backend", choices=["classic", "neural"])
    perception.add_argument("--sample-frames", type=int, default=3)
    perception.add_argument("--save-frame")
    perception.add_argument("--baseline-image-dir")
    perception.add_argument("--baseline-output")
    perception.add_argument("--baseline-viz-output")
    perception.add_argument("--experiment-image-dir")
    perception.add_argument("--experiment-output-dir")
    perception.add_argument("--score-image")
    perception.add_argument("--score-image-dir")
    perception.add_argument("--score-viz-output")

    run = subparsers.add_parser("run")
    run.add_argument("--planner", choices=["classical", "neural"], required=True)
    run.add_argument("--lat", type=float)
    run.add_argument("--lon", type=float)
    run.add_argument("--infer-location", action="store_true")
    run.add_argument("--suppress-irrigation-state", action="store_true")
    run.add_argument("--forecast-hours", type=int)
    run.add_argument("--forecast-days", type=int)
    run.add_argument("--window", default="24H")
    run.add_argument("--camera-url")
    run.add_argument("--camera-backend", choices=["classic", "neural"])
    run.add_argument("--sample-frames", type=int, default=3)
    run.add_argument("--tick-seconds", type=float, default=60.0)
    run.add_argument("--ticks", type=int)
    run.add_argument("--duration-seconds", type=float, default=300.0)
    run.add_argument("--log-jsonl")
    run.add_argument("--execute-actions", action="store_true")
    run.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))

    replay = subparsers.add_parser("replay")
    replay.add_argument("--planner", choices=["classical", "neural"], required=True)
    replay.add_argument("--log-jsonl", required=True)
    replay.add_argument("--execute-actions", action="store_true")
    replay.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--planner", choices=["classical", "neural", "both"], default="both")
    evaluate.add_argument("--eval-dir", default="eval_cases")
    evaluate.add_argument("--execute-actions", action="store_true")
    evaluate.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))

    return parser


def _load_expected_label(path: Path) -> dict[str, Any]:
    """Infer the expected watering outcome from a recorded eval case."""
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    planner_runs = [record["planner_run"] for record in records if record.get("record_type") == "planner_run"]
    if not planner_runs:
        raise RuntimeError(f"Eval case is missing planner_run residue: {path}")
    decision = planner_runs[-1]["decision"]
    expected_action = decision["action"]
    return {
        "expected_action": expected_action,
        "should_water": expected_action == "water_on",
    }


def _decision_should_water(decision: dict[str, Any]) -> bool:
    """Collapse planner actions to a binary watering decision."""
    return decision.get("action") == "water_on"


def _empty_metrics() -> dict[str, Any]:
    """Create one metrics accumulator."""
    return {
        "cases_total": 0,
        "cases_evaluated": 0,
        "error_count": 0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }


def _finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add derived accuracy and precision to an accumulator."""
    evaluated = metrics["cases_evaluated"]
    total = metrics["cases_total"]
    predicted_positive = metrics["tp"] + metrics["fp"]
    # Replay/runtime errors are failures to classify and should therefore count
    # against accuracy even though they do not land inside the 2x2 confusion
    # matrix of successfully scored cases.
    accuracy = ((metrics["tp"] + metrics["tn"]) / total) if total else None
    precision = (metrics["tp"] / predicted_positive) if predicted_positive else None
    return {
        **metrics,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
    }


def _render_confusion_matrix(
    *,
    planner_name: str,
    metrics: dict[str, Any],
    output_path: Path,
) -> str | None:
    """Render a scikit-learn-style confusion matrix image to disk."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as err:
        _LOG.warning(
            "Unable to render confusion matrix for %s: %s: %s",
            planner_name,
            type(err).__name__,
            err,
        )
        return None

    matrix = np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        dtype=float,
    )
    labels = np.array([["TN", "FP"], ["FN", "TP"]], dtype=object)
    total = float(np.sum(matrix))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Predicted No Water", "Predicted Water"],
        yticklabels=["Expected No Water", "Expected Water"],
        xlabel="Predicted label",
        ylabel="True label",
        title=(
            f"{planner_name.capitalize()} Planner Confusion Matrix\n"
            f"scored={metrics['cases_evaluated']} total={metrics['cases_total']} errors={metrics['error_count']}"
        ),
    )

    threshold = (np.max(matrix) / 2.0) if total else 0.0
    for row in range(2):
        for col in range(2):
            count = int(matrix[row, col])
            pct = (count / total * 100.0) if total else 0.0
            ax.text(
                col,
                row,
                f"{labels[row, col]}\n{count}\n{pct:.1f}%",
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontsize=11,
                fontweight="bold" if count > 0 else None,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


async def _run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Replay all known eval cases against one or more planners."""
    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = Path(__file__).resolve().parent / eval_dir
    case_paths = sorted(eval_dir.glob("*.jsonl"))
    if not case_paths:
        raise RuntimeError(f"No eval cases found in {eval_dir}")

    results: dict[str, Any] = {
        "eval_dir": str(eval_dir),
        "case_count": len(case_paths),
        "confusion_matrix_dir": str(eval_dir / "confusion_matrices"),
        "planners": {},
    }
    confusion_matrix_dir = eval_dir / "confusion_matrices"

    for planner_name in _planner_names(args.planner):
        metrics = _empty_metrics()
        case_results = []
        for path in case_paths:
            metrics["cases_total"] += 1
            expectation = _load_expected_label(path)
            executor = PlannerExecutor(planner=_build_planner(planner_name))
            try:
                replay_results = await executor.replay(str(path), execute_actions=args.execute_actions)
                if not replay_results:
                    raise RuntimeError("Replay produced no planner_run results")
                planner_run = replay_results[-1]
                actual_decision = planner_run["decision"]
                should_water = expectation["should_water"]
                predicted_water = _decision_should_water(actual_decision)
                outcome = "tn"
                if should_water and predicted_water:
                    outcome = "tp"
                elif should_water and not predicted_water:
                    outcome = "fn"
                elif (not should_water) and predicted_water:
                    outcome = "fp"
                metrics["cases_evaluated"] += 1
                metrics[outcome] += 1
                case_results.append(
                    {
                        "case": path.name,
                        "status": "ok",
                        "expected_action": expectation["expected_action"],
                        "expected_should_water": should_water,
                        "actual_action": actual_decision["action"],
                        "actual_should_water": predicted_water,
                        "outcome": outcome,
                        "rationale": actual_decision.get("rationale"),
                    }
                )
            except Exception as err:
                metrics["error_count"] += 1
                case_results.append(
                    {
                        "case": path.name,
                        "status": "error",
                        "expected_action": expectation["expected_action"],
                        "expected_should_water": expectation["should_water"],
                        "error": f"{type(err).__name__}: {err}",
                    }
                )

        finalized_metrics = _finalize_metrics(metrics)
        confusion_matrix_path = _render_confusion_matrix(
            planner_name=planner_name,
            metrics=finalized_metrics,
            output_path=confusion_matrix_dir / f"{planner_name}_confusion_matrix.png",
        )

        results["planners"][planner_name] = {
            "metrics": finalized_metrics,
            "confusion_matrix_path": confusion_matrix_path,
            "cases": case_results,
        }

    return results


def _configure_logging(level_name: str) -> None:
    """Configure process-wide runtime logging."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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
        try:
            results = await executor.run(
                tick_count=tick_count,
                tick_seconds=args.tick_seconds,
                execute_actions=args.execute_actions,
            )
        except asyncio.CancelledError:
            _LOG.info("Planner run interrupted by user; shutting down cleanly.")
            return 130
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
    if args.command == "eval":
        print(json.dumps(_jsonable(await _run_eval(args)), indent=2, sort_keys=True))
        return 0
    raise SystemExit(f"Unsupported command: {args.command}")


def main() -> int:
    """Run the CLI synchronously."""
    parser = _build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 2
    _configure_logging(getattr(args, "log_level", os.getenv("LOG_LEVEL", "INFO")))
    _LOG.debug("CLI arguments parsed: %s", vars(args))
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        _LOG.info("Interrupted by user; exiting.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
