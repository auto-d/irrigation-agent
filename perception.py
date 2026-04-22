"""Thin perception adapters that turn service outputs into constrained events."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from typing import Any, Dict, List

from classic_perception import ClassicSecurityCameraPerceptor
from neural_perception import NeuralSecurityCameraPerceptor
from planner import Event
from services.bhyve.controller import controller_from_env
from services.precipitation import PrecipitationClient, PrecipitationSummary
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




def camera_perceptor_for_backend(backend: str) -> Any:
    """Return the configured camera perceptor implementation."""
    normalized = (backend or "classic").strip().lower()
    if normalized == "classic":
        return ClassicSecurityCameraPerceptor()
    if normalized == "neural":
        return NeuralSecurityCameraPerceptor()
    raise ValueError(f"Unknown camera backend: {backend}")
