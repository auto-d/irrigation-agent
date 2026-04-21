"""Thin perception adapters that turn service outputs into constrained events."""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from planner import Event
from services.bhyve.controller import BHyveController, controller_from_env
from services.precipitation import PrecipitationClient, PrecipitationSummary
from services.rtsp.ingest import FFmpegVideoStream, ffprobe_stream_info
from services.weather import Forecast, WeatherClient


def _now() -> dt.datetime:
    """Return the current UTC timestamp."""
    return dt.datetime.now(dt.timezone.utc)


def _bhyve_day(now: dt.datetime) -> int:
    """Convert Python weekday to Orbit/B-hyve weekday numbering."""
    return (now.weekday() + 1) % 7


class ForecastPerceptor:
    """Fetch and condense forecast data into a planner-facing event."""

    async def probe_raw(
        self,
        latitude: float,
        longitude: float,
        *,
        hours: int | None = 12,
        days: int | None = None,
    ) -> Dict[str, Any]:
        """Return raw forecast payloads plus the selected planning window."""
        async with WeatherClient.from_env() as client:
            point = await client.fetch_point(latitude, longitude)
            forecast = await self._fetch_forecast(
                client,
                latitude,
                longitude,
                hours=hours,
                days=days,
            )
        selected_periods = self._select_periods(forecast, hours=hours, days=days)
        return {
            "point": asdict(point),
            "forecast": asdict(forecast),
            "window": self._window_metadata(forecast, hours=hours, days=days),
            "selected_periods": [asdict(period) for period in selected_periods],
        }

    async def perceive(
        self,
        latitude: float,
        longitude: float,
        *,
        hours: int | None = 12,
        days: int | None = None,
    ) -> Event:
        """Return one normalized weather event."""
        async with WeatherClient.from_env() as client:
            forecast = await self._fetch_forecast(
                client,
                latitude,
                longitude,
                hours=hours,
                days=days,
            )
        return self._forecast_event(forecast, hours=hours, days=days)

    async def _fetch_forecast(
        self,
        client: WeatherClient,
        latitude: float,
        longitude: float,
        *,
        hours: int | None,
        days: int | None,
    ) -> Forecast:
        if hours is not None and days is not None:
            raise ValueError("Specify either hours or days for the forecast window, not both.")
        if days is not None:
            return await client.fetch_forecast(latitude, longitude)
        return await client.fetch_hourly_forecast(latitude, longitude)

    def _forecast_event(
        self,
        forecast: Forecast,
        *,
        hours: int | None,
        days: int | None,
    ) -> Event:
        # Integrate precipitation probability over the requested window rather
        # than reducing the forecast to a single point estimate.
        next_periods = self._select_periods(forecast, hours=hours, days=days)
        precip_probs = [
            period.probability_of_precipitation
            for period in next_periods
            if period.probability_of_precipitation is not None
        ]
        temps = [period.temperature for period in next_periods if period.temperature is not None]
        precip_probability_hours = round(
            sum(self._period_duration_hours(period) * ((period.probability_of_precipitation or 0) / 100.0) for period in next_periods),
            2,
        )
        hours_above_40pct = round(
            sum(
                self._period_duration_hours(period)
                for period in next_periods
                if (period.probability_of_precipitation or 0) >= 40
            ),
            2,
        )
        payload = {
            "forecast_type": forecast.forecast_type,
            "point": {
                "latitude": forecast.point.latitude,
                "longitude": forecast.point.longitude,
                "city": forecast.point.city,
                "state": forecast.point.state,
                "timezone": forecast.point.timezone,
            },
            "window": self._window_metadata(forecast, hours=hours, days=days),
            "periods_considered": len(next_periods),
            "max_precip_probability": max(precip_probs) if precip_probs else None,
            "avg_precip_probability": round(sum(precip_probs) / len(precip_probs), 1) if precip_probs else None,
            "precip_probability_hours": precip_probability_hours,
            "hours_above_40pct": hours_above_40pct,
            "avg_temperature": round(sum(temps) / len(temps), 1) if temps else None,
            "headline": next_periods[0].short_forecast if next_periods else None,
            "rain_expected_in_window": precip_probability_hours >= 1.5 or hours_above_40pct >= 2.0,
            "rain_expected_soon": precip_probability_hours >= 1.5 or hours_above_40pct >= 2.0,
        }
        return Event(timestamp=_now(), source="weather", type="forecast_summary", payload=payload)

    def _select_periods(
        self,
        forecast: Forecast,
        *,
        hours: int | None,
        days: int | None,
    ) -> List[Any]:
        if hours is not None and days is not None:
            raise ValueError("Specify either hours or days for the forecast window, not both.")
        if hours is None and days is None:
            hours = 12

        now = dt.datetime.now(dt.timezone.utc)
        if hours is not None:
            cutoff = now + dt.timedelta(hours=hours)
        else:
            cutoff = now + dt.timedelta(days=days or 0)

        selected = [
            period
            for period in forecast.periods
            if period.start_time < cutoff and period.end_time > now
        ]
        if selected:
            return selected

        if hours is not None:
            return list(forecast.periods[:hours])
        return list(forecast.periods[: max(1, days or 1)])

    @staticmethod
    def _period_duration_hours(period: Any) -> float:
        return max(0.0, (period.end_time - period.start_time).total_seconds() / 3600.0)

    def _window_metadata(
        self,
        forecast: Forecast,
        *,
        hours: int | None,
        days: int | None,
    ) -> Dict[str, Any]:
        if hours is None and days is None:
            hours = 12
        return {
            "unit": "days" if days is not None else "hours",
            "value": days if days is not None else hours,
            "forecast_type": forecast.forecast_type,
        }


class HistoricalPrecipitationPerceptor:
    """Fetch and condense precipitation historicals into a planner-facing event."""

    async def probe_raw(self, window: str) -> Dict[str, Any]:
        """Return raw precipitation records and the derived summary."""
        async with PrecipitationClient.from_env() as client:
            raw = await client.fetch_raw(window)
            summary = await client.summarize(window)
        return {
            "raw": raw,
            "summary": {
                "window": summary.window,
                "total_inches": summary.total_inches,
                "sample_count": summary.sample_count,
                "started_at": summary.started_at.isoformat() if summary.started_at else None,
                "ended_at": summary.ended_at.isoformat() if summary.ended_at else None,
            },
        }

    async def perceive(self, window: str) -> Event:
        """Return one normalized precipitation event."""
        async with PrecipitationClient.from_env() as client:
            summary = await client.summarize(window)
        return self._summary_event(summary)

    def _summary_event(self, summary: PrecipitationSummary) -> Event:
        payload = {
            "window": summary.window,
            "total_inches": summary.total_inches,
            "sample_count": summary.sample_count,
            "started_at": summary.started_at.isoformat() if summary.started_at else None,
            "ended_at": summary.ended_at.isoformat() if summary.ended_at else None,
            "recent_rain": summary.total_inches > 0.05,
        }
        return Event(
            timestamp=_now(),
            source="precipitation",
            type="precipitation_summary",
            payload=payload,
        )


class IrrigationPerceptor:
    """Fetch and condense irrigation controller state into a planner-facing event."""

    async def probe_raw(self) -> Dict[str, Any]:
        """Return the full controller payloads used during troubleshooting."""
        async with controller_from_env() as controller:
            devices = controller.get_devices()
            sprinkler = controller.list_sprinkler_devices()[0]
            programs = controller.get_programs(str(sprinkler["id"]))
            history = await controller.get_history(str(sprinkler["id"]))
        return {
            "devices": devices,
            "sprinkler": sprinkler,
            "programs": programs,
            "history": history,
        }

    async def perceive(self) -> Event:
        """Return one normalized irrigation-status event."""
        async with controller_from_env() as controller:
            sprinkler = controller.list_sprinkler_devices()[0]
            programs = controller.get_programs(str(sprinkler["id"]))
            history = await controller.get_history(str(sprinkler["id"]))
        return self._irrigation_event(sprinkler, programs, history)

    def _irrigation_event(
        self,
        sprinkler: Dict[str, Any],
        programs: List[Dict[str, Any]],
        history: Any,
    ) -> Event:
        now_local = dt.datetime.now().astimezone()
        # The API status is not reliable on its own for the hose timer, so we
        # also derive whether the device should be running from local schedules.
        expected_on, matching_programs = self._expected_programs_now(programs, now_local)
        status = sprinkler.get("status") or {}
        payload = {
            "device_id": sprinkler.get("id"),
            "name": sprinkler.get("name"),
            "connected": sprinkler.get("is_connected"),
            "run_mode": status.get("run_mode"),
            "api_reports_on": self._api_reports_on(status),
            "expected_on": expected_on,
            "watered_within_24h": self._watered_within_window(history, hours=24),
            "matching_programs": matching_programs,
            "next_start_time": status.get("next_start_time"),
            "next_start_programs": status.get("next_start_programs"),
        }
        return Event(timestamp=_now(), source="irrigation", type="irrigation_status", payload=payload)

    @staticmethod
    def _api_reports_on(status: Dict[str, Any]) -> bool:
        if status.get("run_mode") == "manual":
            return True
        watering_status = status.get("watering_status")
        if not isinstance(watering_status, dict):
            return False
        return any(
            watering_status.get(key) not in (None, [], {}, "")
            for key in ("current_station", "run_time", "started_watering_station_at", "stations", "program")
        )

    @staticmethod
    def _expected_programs_now(programs: List[Dict[str, Any]], now_local: dt.datetime) -> tuple[bool, List[str]]:
        matching: List[str] = []
        day = _bhyve_day(now_local)
        for program in programs:
            if not program.get("enabled", True):
                continue
            frequency = program.get("frequency") or {}
            if frequency.get("type") == "days" and day not in frequency.get("days", []):
                continue
            total_runtime = sum(int(run.get("run_time", 0)) for run in program.get("run_times", []))
            for start_time in program.get("start_times", []):
                try:
                    hour, minute = [int(part) for part in start_time.split(":", 1)]
                except Exception:
                    continue
                started = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
                ended = started + dt.timedelta(minutes=total_runtime)
                if started <= now_local <= ended:
                    matching.append(program.get("program") or program.get("name") or program.get("id"))
        return (len(matching) > 0, matching)

    @classmethod
    def _watered_within_window(cls, history: Any, *, hours: int) -> bool:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
        for started_at, duration_minutes in cls._history_runs(history):
            if started_at is None:
                continue
            ended_at = started_at + dt.timedelta(minutes=max(duration_minutes, 0.0))
            if ended_at >= cutoff:
                return True
        return False

    @classmethod
    def _history_runs(cls, history: Any) -> List[tuple[dt.datetime | None, float]]:
        runs: List[tuple[dt.datetime | None, float]] = []

        def visit(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item)
                return
            if isinstance(value, dict):
                started_at = cls._parse_history_datetime(
                    value.get("started_at")
                    or value.get("start_time")
                    or value.get("watering_started_at")
                    or value.get("created_at")
                )
                duration_minutes = cls._extract_duration_minutes(value)
                if started_at is not None and duration_minutes is not None:
                    runs.append((started_at, duration_minutes))
                for nested in value.values():
                    if isinstance(nested, (list, dict)):
                        visit(nested)

        visit(history)
        return runs

    @staticmethod
    def _parse_history_datetime(value: Any) -> dt.datetime | None:
        if not isinstance(value, str) or not value:
            return None
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = dt.datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)

    @staticmethod
    def _extract_duration_minutes(payload: Dict[str, Any]) -> float | None:
        candidates = (
            ("duration", 1.0),
            ("duration_minutes", 1.0),
            ("run_time", 1.0),
            ("runtime", 1.0),
            ("watering_duration", 1.0),
            ("duration_seconds", 1.0 / 60.0),
            ("run_time_seconds", 1.0 / 60.0),
            ("runtime_seconds", 1.0 / 60.0),
        )
        for key, scale in candidates:
            value = payload.get(key)
            if value in (None, ""):
                continue
            try:
                return float(value) * scale
            except (TypeError, ValueError):
                continue
        return 0.0


class SecurityCameraPerceptor:
    """Sample one or more frames and emit a constrained scene event."""

    def probe_raw(
        self,
        url: str | None = None,
        *,
        sample_frames: int = 1,
        save_frame_path: str | None = None,
        baseline_image_dir: str | None = None,
        baseline_output_path: str | None = None,
        score_image_path: str | None = None,
        score_image_dir: str | None = None,
    ) -> Dict[str, Any]:
        """Return inexpensive stream metadata and sampled frame statistics."""
        if baseline_image_dir:
            summary = self.build_baseline_from_directory(
                baseline_image_dir,
                save_baseline_path=baseline_output_path,
            )
            return {
                "mode": "baseline_directory",
                "baseline": summary,
            }
        if score_image_path:
            if not baseline_output_path:
                raise RuntimeError("baseline output path required when scoring a still image")
            return {
                "mode": "score_image",
                "score": self.score_image_against_baseline(
                    score_image_path,
                    baseline_output_path,
                ),
            }
        if score_image_dir:
            if not baseline_output_path:
                raise RuntimeError("baseline output path required when scoring a still-image directory")
            return {
                "mode": "score_image_directory",
                "evaluation": self.evaluate_labeled_image_directory(
                    score_image_dir,
                    baseline_output_path,
                ),
            }
        if not url:
            raise RuntimeError("camera URL required")
        stream_info = ffprobe_stream_info(url)
        stats, capture_info = self._sample_frame_stats(
            url,
            width=int(stream_info["width"]),
            height=int(stream_info["height"]),
            sample_frames=sample_frames,
            save_frame_path=save_frame_path,
        )
        return {
            "stream": stream_info,
            "sample_frames": sample_frames,
            "frame_stats": stats,
            "capture": capture_info,
        }

    def perceive(
        self,
        url: str | None = None,
        *,
        sample_frames: int = 3,
        save_frame_path: str | None = None,
        baseline_image_dir: str | None = None,
        baseline_output_path: str | None = None,
        score_image_path: str | None = None,
        score_image_dir: str | None = None,
    ) -> Event:
        """Return one normalized camera scene event."""
        if baseline_image_dir:
            summary = self.build_baseline_from_directory(
                baseline_image_dir,
                save_baseline_path=baseline_output_path,
            )
            payload = {
                "mode": "baseline_directory",
                "image_dir": summary["image_dir"],
                "image_count": summary["image_count"],
                "width": summary["width"],
                "height": summary["height"],
                "patch_size": summary["patch_size"],
                "mean_brightness_avg": summary["mean_brightness_avg"],
                "mean_brightness_std": summary["mean_brightness_std"],
                "normalization": summary["normalization"],
                "patch_rows": summary["patch_rows"],
                "patch_cols": summary["patch_cols"],
                "patch_variability_mean": summary["patch_variability_mean"],
                "patch_variability_p95": summary["patch_variability_p95"],
                "patch_variability_max": summary["patch_variability_max"],
                "feature_medians": summary["feature_medians"],
                "top_variable_patches": summary["top_variable_patches"],
                "saved_baseline_path": summary.get("saved_baseline_path"),
                "interpretation": "scene_baseline_established",
            }
            return Event(timestamp=_now(), source="camera", type="scene_baseline", payload=payload)
        if score_image_path:
            if not baseline_output_path:
                raise RuntimeError("baseline output path required when scoring a still image")
            result = self.score_image_against_baseline(score_image_path, baseline_output_path)
            payload = {
                "mode": "score_image",
                **result,
                "interpretation": "scene_anomaly_scored",
            }
            return Event(timestamp=_now(), source="camera", type="scene_anomaly_score", payload=payload)
        if score_image_dir:
            if not baseline_output_path:
                raise RuntimeError("baseline output path required when scoring a still-image directory")
            result = self.evaluate_labeled_image_directory(score_image_dir, baseline_output_path)
            payload = {
                "mode": "score_image_directory",
                **result,
                "interpretation": "scene_anomaly_batch_evaluated",
            }
            return Event(timestamp=_now(), source="camera", type="scene_anomaly_batch", payload=payload)
        if not url:
            raise RuntimeError("camera URL required")
        stream_info = ffprobe_stream_info(url)
        stats, capture_info = self._sample_frame_stats(
            url,
            width=int(stream_info["width"]),
            height=int(stream_info["height"]),
            sample_frames=sample_frames,
            save_frame_path=save_frame_path,
        )
        mean_brightness = round(float(np.mean([frame["mean_brightness"] for frame in stats])), 2) if stats else None
        motion_score = max((frame["motion_score"] for frame in stats), default=0.0)
        payload = {
            "width": stream_info["width"],
            "height": stream_info["height"],
            "sample_frames": len(stats),
            "saved_frame_path": capture_info.get("saved_frame_path"),
            "mean_brightness": mean_brightness,
            "motion_score": round(float(motion_score), 4),
            "interpretation": "unclassified_scene",
            "person_detected": None,
            "animal_detected": None,
            "lawn_mower_active": None,
        }
        return Event(timestamp=_now(), source="camera", type="scene_activity", payload=payload)

    def build_baseline_from_directory(
        self,
        image_dir: str,
        *,
        save_baseline_path: str | None = None,
        patch_size: int = 32,
    ) -> Dict[str, Any]:
        """Build a texture-series baseline from a directory of still images."""
        image_paths = self._list_image_paths(image_dir)
        if len(image_paths) < 2:
            raise RuntimeError("baseline image directory must contain at least two readable images")

        brightness_values: List[float] = []
        grad_mean_series: List[np.ndarray] = []
        grad_std_series: List[np.ndarray] = []
        edge_density_series: List[np.ndarray] = []
        lbp_hist_series: List[np.ndarray] = []
        reference_shape: Tuple[int, int, int] | None = None
        descriptor_shape: Tuple[int, int] | None = None
        for path in image_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if reference_shape is None:
                reference_shape = frame.shape
            if frame.shape != reference_shape:
                raise RuntimeError(
                    f"baseline image {path} has shape {frame.shape}, expected {reference_shape}"
                )
            descriptors = self._compute_patch_texture_descriptors(frame, patch_size=patch_size)
            if descriptor_shape is None:
                descriptor_shape = descriptors["grad_mean"].shape
            brightness_values.append(float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
            grad_mean_series.append(descriptors["grad_mean"])
            grad_std_series.append(descriptors["grad_std"])
            edge_density_series.append(descriptors["edge_density"])
            lbp_hist_series.append(descriptors["lbp_hist"])

        if len(grad_mean_series) < 2 or reference_shape is None or descriptor_shape is None:
            raise RuntimeError("baseline image directory must contain at least two readable images")

        height, width = reference_shape[:2]
        patch_rows, patch_cols = descriptor_shape
        grad_mean_stack = np.stack(grad_mean_series, axis=0)
        grad_std_stack = np.stack(grad_std_series, axis=0)
        edge_density_stack = np.stack(edge_density_series, axis=0)
        lbp_hist_stack = np.stack(lbp_hist_series, axis=0)

        grad_mean_median = np.median(grad_mean_stack, axis=0)
        grad_std_median = np.median(grad_std_stack, axis=0)
        edge_density_median = np.median(edge_density_stack, axis=0)
        lbp_hist_mean = np.mean(lbp_hist_stack, axis=0)
        lbp_hist_std = np.std(lbp_hist_stack, axis=0)

        grad_mean_mad = np.median(np.abs(grad_mean_stack - grad_mean_median), axis=0)
        grad_std_mad = np.median(np.abs(grad_std_stack - grad_std_median), axis=0)
        edge_density_mad = np.median(np.abs(edge_density_stack - edge_density_median), axis=0)

        patch_variability = (
            grad_mean_mad
            + grad_std_mad
            + (edge_density_mad * 255.0)
            + np.mean(lbp_hist_std, axis=2)
        )

        top_variable_patches = self._top_scored_patches(
            patch_variability,
            patch_size=patch_size,
            image_width=width,
            image_height=height,
            score_label="variability_score",
        )

        summary = {
            "image_dir": str(Path(image_dir).expanduser().resolve()),
            "image_count": len(grad_mean_series),
            "sample_image_names": [path.name for path in image_paths[: min(10, len(image_paths))]],
            "width": width,
            "height": height,
            "patch_size": patch_size,
            "patch_rows": int(patch_rows),
            "patch_cols": int(patch_cols),
            "normalization": "local_contrast_gradient_lbp",
            "mean_brightness_avg": round(float(np.mean(brightness_values)), 4),
            "mean_brightness_std": round(float(np.std(brightness_values)), 4),
            "feature_medians": {
                "gradient_mean": round(float(np.median(grad_mean_median)), 4),
                "gradient_std": round(float(np.median(grad_std_median)), 4),
                "edge_density": round(float(np.median(edge_density_median)), 6),
            },
            "patch_variability_mean": round(float(np.mean(patch_variability)), 4),
            "patch_variability_p95": round(float(np.percentile(patch_variability, 95)), 4),
            "patch_variability_max": round(float(np.max(patch_variability)), 4),
            "top_variable_patches": top_variable_patches,
        }

        if save_baseline_path:
            saved_path = self._write_baseline_artifact(
                save_baseline_path,
                summary=summary,
                patch_model={
                    "grad_mean_median": grad_mean_median,
                    "grad_mean_mad": grad_mean_mad,
                    "grad_std_median": grad_std_median,
                    "grad_std_mad": grad_std_mad,
                    "edge_density_median": edge_density_median,
                    "edge_density_mad": edge_density_mad,
                    "lbp_hist_mean": lbp_hist_mean,
                    "lbp_hist_std": lbp_hist_std,
                    "patch_variability": patch_variability,
                },
            )
            summary["saved_baseline_path"] = saved_path

        return summary

    def score_image_against_baseline(
        self,
        image_path: str,
        baseline_path: str,
    ) -> Dict[str, Any]:
        """Score one still image against a saved texture baseline."""
        baseline = self._load_baseline_artifact(baseline_path)
        model = baseline["model"]
        summary = baseline["summary"]
        patch_size = int(summary["patch_size"])

        frame = cv2.imread(str(Path(image_path).expanduser()), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"unable to read score image: {image_path}")
        expected_shape = (int(summary["height"]), int(summary["width"]), 3)
        if frame.shape != expected_shape:
            raise RuntimeError(f"score image shape {frame.shape} did not match baseline shape {expected_shape}")

        descriptors = self._compute_patch_texture_descriptors(frame, patch_size=patch_size)
        grad_mean_score = np.abs(descriptors["grad_mean"] - model["grad_mean_median"]) / (
            model["grad_mean_mad"] + 1.0
        )
        grad_std_score = np.abs(descriptors["grad_std"] - model["grad_std_median"]) / (
            model["grad_std_mad"] + 1.0
        )
        edge_density_score = np.abs(descriptors["edge_density"] - model["edge_density_median"]) / (
            model["edge_density_mad"] + 0.01
        )
        lbp_distance = 0.5 * np.sum(
            np.abs(descriptors["lbp_hist"] - model["lbp_hist_mean"]),
            axis=2,
        )
        score_grid = (
            grad_mean_score
            + grad_std_score
            + edge_density_score
            + (lbp_distance * 8.0)
        ) / 4.0
        threshold = max(3.5, float(np.percentile(model["patch_variability"], 95) / 4.0))
        anomalous = score_grid >= threshold
        candidate_boxes = self._merge_anomalous_patches(
            anomalous,
            score_grid,
            patch_size=patch_size,
            image_width=int(summary["width"]),
            image_height=int(summary["height"]),
        )
        scene_score = self._scene_score_from_grid(score_grid)

        return {
            "image_path": str(Path(image_path).expanduser().resolve()),
            "baseline_path": str(Path(baseline_path).expanduser().resolve()),
            "width": int(summary["width"]),
            "height": int(summary["height"]),
            "patch_size": patch_size,
            "patch_rows": int(summary["patch_rows"]),
            "patch_cols": int(summary["patch_cols"]),
            "anomaly_threshold": round(float(threshold), 4),
            "scene_score": round(float(scene_score), 4),
            "anomaly_score_mean": round(float(np.mean(score_grid)), 4),
            "anomaly_score_p95": round(float(np.percentile(score_grid, 95)), 4),
            "anomaly_score_max": round(float(np.max(score_grid)), 4),
            "anomalous_patch_count": int(np.sum(anomalous)),
            "top_anomalous_patches": self._top_scored_patches(
                score_grid,
                patch_size=patch_size,
                image_width=int(summary["width"]),
                image_height=int(summary["height"]),
                score_label="anomaly_score",
                mask=anomalous,
            ),
            "candidate_boxes": candidate_boxes,
        }

    def evaluate_labeled_image_directory(
        self,
        image_dir: str,
        baseline_path: str,
    ) -> Dict[str, Any]:
        """Score a labeled directory and aggregate anomaly separation by label."""
        image_paths = self._list_image_paths(image_dir)
        records: List[Dict[str, Any]] = []
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for path in image_paths:
            score = self.score_image_against_baseline(str(path), baseline_path)
            label = self._label_from_filename(path)
            record = {
                "image_name": path.name,
                "label": label,
                "scene_score": score["scene_score"],
                "anomaly_score_max": score["anomaly_score_max"],
                "anomaly_score_p95": score["anomaly_score_p95"],
                "anomalous_patch_count": score["anomalous_patch_count"],
                "candidate_box_count": len(score["candidate_boxes"]),
                "candidate_boxes": score["candidate_boxes"][:3],
            }
            records.append(record)
            by_label.setdefault(label, []).append(record)

        label_summary: Dict[str, Dict[str, Any]] = {}
        for label, items in sorted(by_label.items()):
            scene_scores = [item["scene_score"] for item in items]
            max_scores = [item["anomaly_score_max"] for item in items]
            patch_counts = [item["anomalous_patch_count"] for item in items]
            box_counts = [item["candidate_box_count"] for item in items]
            label_summary[label] = {
                "count": len(items),
                "scene_score_avg": round(float(np.mean(scene_scores)), 4),
                "scene_score_p95": round(float(np.percentile(scene_scores, 95)), 4),
                "scene_score_max": round(float(np.max(scene_scores)), 4),
                "anomaly_score_max_avg": round(float(np.mean(max_scores)), 4),
                "anomaly_score_max_p95": round(float(np.percentile(max_scores, 95)), 4),
                "anomaly_score_max_max": round(float(np.max(max_scores)), 4),
                "anomalous_patch_count_avg": round(float(np.mean(patch_counts)), 4),
                "candidate_box_count_avg": round(float(np.mean(box_counts)), 4),
                "top_examples": sorted(items, key=lambda item: item["anomaly_score_max"], reverse=True)[:3],
            }

        threshold_summary = self._threshold_summary(records)
        return {
            "image_dir": str(Path(image_dir).expanduser().resolve()),
            "baseline_path": str(Path(baseline_path).expanduser().resolve()),
            "image_count": len(records),
            "label_summary": label_summary,
            "threshold_summary": threshold_summary,
            "records": sorted(records, key=lambda item: item["anomaly_score_max"], reverse=True),
        }

    def _sample_frame_stats(
        self,
        url: str,
        *,
        width: int,
        height: int,
        sample_frames: int,
        save_frame_path: str | None = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Collect lightweight frame metrics without downstream inference."""
        stats: List[Dict[str, Any]] = []
        previous_gray = None
        saved_frame_path: str | None = None
        first_frame_shape: List[int] | None = None
        first_frame_dtype: str | None = None
        started_at = time.monotonic()
        with FFmpegVideoStream(url, width, height).start() as stream:
            for index in range(sample_frames):
                frame, error = stream.read()
                if frame is None:
                    raise RuntimeError(f"Unable to read RTSP frame: {error or 'unknown error'}")
                if index == 0:
                    first_frame_shape = list(frame.shape)
                    first_frame_dtype = str(frame.dtype)
                if index == 0 and save_frame_path:
                    saved_frame_path = self._write_debug_frame(frame, save_frame_path)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = float(np.mean(gray))
                brightness_stddev = float(np.std(gray))
                contrast = brightness_stddev
                blur_laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                motion_score = 0.0
                if previous_gray is not None:
                    motion_score = float(np.mean(cv2.absdiff(gray, previous_gray)))
                previous_gray = gray
                stats.append(
                    {
                        "index": index,
                        "mean_brightness": mean_brightness,
                        "brightness_stddev": brightness_stddev,
                        "contrast": contrast,
                        "blur_laplacian_variance": blur_laplacian_variance,
                        "motion_score": motion_score,
                    }
                )
        elapsed_seconds = max(time.monotonic() - started_at, 1e-6)
        capture_info = {
            "saved_frame_path": saved_frame_path,
            "frame_shape": first_frame_shape,
            "dtype": first_frame_dtype,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "approx_fps": round(len(stats) / elapsed_seconds, 3),
        }
        return stats, capture_info

    @staticmethod
    def _list_image_paths(image_dir: str) -> List[Path]:
        path = Path(image_dir).expanduser()
        if not path.is_dir():
            raise RuntimeError(f"baseline image directory not found: {path}")
        image_paths = sorted(
            file_path
            for file_path in path.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )
        if not image_paths:
            raise RuntimeError(f"no supported image files found in baseline directory: {path}")
        return image_paths

    @staticmethod
    def _top_scored_patches(
        score_grid: np.ndarray,
        *,
        patch_size: int,
        image_width: int,
        image_height: int,
        score_label: str,
        mask: np.ndarray | None = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        rows, cols = score_grid.shape
        for row in range(rows):
            for col in range(cols):
                if mask is not None and not bool(mask[row, col]):
                    continue
                x0 = col * patch_size
                y0 = row * patch_size
                x1 = min(image_width, x0 + patch_size)
                y1 = min(image_height, y0 + patch_size)
                ranked.append(
                    {
                        "row": int(row),
                        "col": int(col),
                        "x0": int(x0),
                        "y0": int(y0),
                        "x1": int(x1),
                        "y1": int(y1),
                        score_label: round(float(score_grid[row, col]), 4),
                    }
                )
        ranked.sort(key=lambda item: item[score_label], reverse=True)
        return ranked[:limit]

    @staticmethod
    def _write_baseline_artifact(
        save_baseline_path: str,
        *,
        summary: Dict[str, Any],
        patch_model: Dict[str, np.ndarray],
    ) -> str:
        path = Path(save_baseline_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "summary": summary,
            "model": {key: np.round(value, 6).tolist() for key, value in patch_model.items()},
        }
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        return str(path.resolve())

    @staticmethod
    def _load_baseline_artifact(path: str) -> Dict[str, Any]:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        model = {key: np.array(value, dtype=np.float32) for key, value in payload["model"].items()}
        return {
            "summary": payload["summary"],
            "model": model,
        }

    @staticmethod
    def _label_from_filename(path: Path) -> str:
        stem = path.stem
        match = re.search(r" - ([A-Z][A-Z _]+)$", stem)
        if not match:
            return "CLEAR"
        suffix = match.group(1).strip()
        if not suffix:
            return "CLEAR"
        return suffix.upper().replace(" ", "_")

    @staticmethod
    def _compute_patch_texture_descriptors(
        frame: np.ndarray,
        *,
        patch_size: int,
    ) -> Dict[str, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        normalized = SecurityCameraPerceptor._normalize_local_contrast(gray)
        grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        normalized_u8 = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edges = cv2.Canny(normalized_u8, 50, 150).astype(np.float32) / 255.0
        lbp = SecurityCameraPerceptor._lbp_image(normalized_u8)

        height, width = gray.shape
        patch_rows = max(1, height // patch_size)
        patch_cols = max(1, width // patch_size)
        effective_height = patch_rows * patch_size
        effective_width = patch_cols * patch_size

        grad_mag = grad_mag[:effective_height, :effective_width]
        edges = edges[:effective_height, :effective_width]
        lbp = lbp[:effective_height, :effective_width]

        grad_mean = grad_mag.reshape(patch_rows, patch_size, patch_cols, patch_size).mean(axis=(1, 3))
        grad_std = grad_mag.reshape(patch_rows, patch_size, patch_cols, patch_size).std(axis=(1, 3))
        edge_density = edges.reshape(patch_rows, patch_size, patch_cols, patch_size).mean(axis=(1, 3))
        lbp_hist = SecurityCameraPerceptor._patch_lbp_histograms(
            lbp,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            patch_size=patch_size,
        )
        return {
            "grad_mean": grad_mean.astype(np.float32),
            "grad_std": grad_std.astype(np.float32),
            "edge_density": edge_density.astype(np.float32),
            "lbp_hist": lbp_hist.astype(np.float32),
        }

    @staticmethod
    def _normalize_local_contrast(gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (0, 0), 15)
        high_pass = gray - blurred
        local_std = cv2.GaussianBlur(high_pass * high_pass, (0, 0), 15)
        local_std = np.sqrt(np.maximum(local_std, 1.0))
        return high_pass / local_std

    @staticmethod
    def _lbp_image(gray_u8: np.ndarray) -> np.ndarray:
        center = gray_u8[1:-1, 1:-1]
        lbp = np.zeros_like(center, dtype=np.uint8)
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ]
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = gray_u8[1 + dy : gray_u8.shape[0] - 1 + dy, 1 + dx : gray_u8.shape[1] - 1 + dx]
            lbp |= ((neighbor >= center).astype(np.uint8) << bit)
        padded = np.zeros_like(gray_u8, dtype=np.uint8)
        padded[1:-1, 1:-1] = lbp
        return padded

    @staticmethod
    def _patch_lbp_histograms(
        lbp: np.ndarray,
        *,
        patch_rows: int,
        patch_cols: int,
        patch_size: int,
        bins: int = 16,
    ) -> np.ndarray:
        hist = np.zeros((patch_rows, patch_cols, bins), dtype=np.float32)
        bin_edges = np.linspace(0, 256, bins + 1, dtype=np.int32)
        for row in range(patch_rows):
            for col in range(patch_cols):
                patch = lbp[
                    row * patch_size : (row + 1) * patch_size,
                    col * patch_size : (col + 1) * patch_size,
                ]
                counts, _ = np.histogram(patch, bins=bin_edges)
                counts = counts.astype(np.float32)
                counts /= max(float(np.sum(counts)), 1.0)
                hist[row, col] = counts
        return hist

    @staticmethod
    def _merge_anomalous_patches(
        anomalous: np.ndarray,
        score_grid: np.ndarray,
        *,
        patch_size: int,
        image_width: int,
        image_height: int,
    ) -> List[Dict[str, Any]]:
        rows, cols = anomalous.shape
        visited = np.zeros_like(anomalous, dtype=bool)
        boxes: List[Dict[str, Any]] = []
        for row in range(rows):
            for col in range(cols):
                if not anomalous[row, col] or visited[row, col]:
                    continue
                stack = [(row, col)]
                visited[row, col] = True
                component: List[Tuple[int, int]] = []
                while stack:
                    current_row, current_col = stack.pop()
                    component.append((current_row, current_col))
                    for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        next_row = current_row + delta_row
                        next_col = current_col + delta_col
                        if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                            continue
                        if visited[next_row, next_col] or not anomalous[next_row, next_col]:
                            continue
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
                box_rows = [item[0] for item in component]
                box_cols = [item[1] for item in component]
                x0 = min(box_cols) * patch_size
                y0 = min(box_rows) * patch_size
                x1 = min(image_width, (max(box_cols) + 1) * patch_size)
                y1 = min(image_height, (max(box_rows) + 1) * patch_size)
                component_scores = [float(score_grid[item[0], item[1]]) for item in component]
                boxes.append(
                    {
                        "x0": int(x0),
                        "y0": int(y0),
                        "x1": int(x1),
                        "y1": int(y1),
                        "patch_count": int(len(component)),
                        "score_max": round(float(max(component_scores)), 4),
                        "score_mean": round(float(np.mean(component_scores)), 4),
                    }
                )
        boxes.sort(key=lambda item: item["score_max"], reverse=True)
        return boxes

    @staticmethod
    def _scene_score_from_grid(score_grid: np.ndarray) -> float:
        flat = np.sort(score_grid.reshape(-1))
        if flat.size == 0:
            return 0.0
        top_k = max(4, min(24, flat.size // 50))
        top_mean = float(np.mean(flat[-top_k:]))
        p99 = float(np.percentile(flat, 99))
        p95 = float(np.percentile(flat, 95))
        return (0.5 * top_mean) + (0.3 * p99) + (0.2 * p95)

    @staticmethod
    def _threshold_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        labeled = [
            {
                "scene_score": float(record["scene_score"]),
                "is_positive": record["label"] != "CLEAR",
            }
            for record in records
        ]
        positives = [item for item in labeled if item["is_positive"]]
        negatives = [item for item in labeled if not item["is_positive"]]
        if not positives or not negatives:
            return {"error": "need both CLEAR and obstacle-labeled images for threshold evaluation"}

        candidate_scores = sorted({item["scene_score"] for item in labeled})
        thresholds = []
        thresholds.extend(candidate_scores)
        for left, right in zip(candidate_scores, candidate_scores[1:]):
            thresholds.append((left + right) / 2.0)
        thresholds = sorted(set(thresholds))

        best_accuracy = None
        best_f1 = None
        for threshold in thresholds:
            tp = sum(1 for item in labeled if item["is_positive"] and item["scene_score"] >= threshold)
            fp = sum(1 for item in labeled if (not item["is_positive"]) and item["scene_score"] >= threshold)
            tn = sum(1 for item in labeled if (not item["is_positive"]) and item["scene_score"] < threshold)
            fn = sum(1 for item in labeled if item["is_positive"] and item["scene_score"] < threshold)

            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            balanced_accuracy = (recall + specificity) / 2.0
            summary = {
                "threshold": round(float(threshold), 4),
                "accuracy": round(float(accuracy), 4),
                "balanced_accuracy": round(float(balanced_accuracy), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
            if best_accuracy is None or (
                summary["accuracy"], summary["balanced_accuracy"], summary["f1"]
            ) > (
                best_accuracy["accuracy"],
                best_accuracy["balanced_accuracy"],
                best_accuracy["f1"],
            ):
                best_accuracy = summary
            if best_f1 is None or (
                summary["f1"], summary["balanced_accuracy"], summary["accuracy"]
            ) > (
                best_f1["f1"],
                best_f1["balanced_accuracy"],
                best_f1["accuracy"],
            ):
                best_f1 = summary

        return {
            "scene_score_clear_avg": round(float(np.mean([item["scene_score"] for item in negatives])), 4),
            "scene_score_positive_avg": round(float(np.mean([item["scene_score"] for item in positives])), 4),
            "best_accuracy_threshold": best_accuracy,
            "best_f1_threshold": best_f1,
        }

    @staticmethod
    def _write_debug_frame(frame: Any, save_frame_path: str) -> str:
        path = Path(save_frame_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Failed to write debug frame to {path}")
        return str(path.resolve())
