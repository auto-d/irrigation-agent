"""Helpers for pulling NWS forecasts from api.weather.gov."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from aiohttp import ClientSession
else:
    ClientSession = Any

DEFAULT_NWS_API_BASE_URL = "https://api.weather.gov"
DEFAULT_NWS_ACCEPT = "application/geo+json"
DEFAULT_NWS_USER_AGENT = "agent-bakeoff/weather-client"


@dataclass(frozen=True)
class PointMetadata:
    """Normalized metadata returned by the NWS points lookup."""

    latitude: float
    longitude: float
    city: Optional[str]
    state: Optional[str]
    timezone: Optional[str]
    radar_station: Optional[str]
    office: Optional[str]
    grid_x: Optional[int]
    grid_y: Optional[int]
    forecast_url: str
    hourly_forecast_url: str
    forecast_grid_data_url: Optional[str]
    observation_stations_url: Optional[str]


@dataclass(frozen=True)
class ForecastPeriod:
    """One forecast period returned by the NWS forecast endpoints."""

    number: int
    name: str
    start_time: dt.datetime
    end_time: dt.datetime
    is_daytime: bool
    temperature: Optional[int]
    temperature_unit: Optional[str]
    temperature_trend: Optional[str]
    probability_of_precipitation: Optional[int]
    wind_speed: Optional[str]
    wind_direction: Optional[str]
    icon_url: Optional[str]
    short_forecast: Optional[str]
    detailed_forecast: Optional[str]


@dataclass(frozen=True)
class Forecast:
    """Normalized daily or hourly forecast payload."""

    forecast_type: str
    generated_at: Optional[dt.datetime]
    updated_at: Optional[dt.datetime]
    units: Optional[str]
    point: PointMetadata
    periods: Sequence[ForecastPeriod]


class WeatherClient:
    """Thin async wrapper around the NWS points and forecast endpoints."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_NWS_API_BASE_URL,
        session: Optional[ClientSession] = None,
        user_agent: str = DEFAULT_NWS_USER_AGENT,
        accept: str = DEFAULT_NWS_ACCEPT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = session
        self._owns_session = session is None
        self._user_agent = user_agent
        self._accept = accept

    async def __aenter__(self) -> "WeatherClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @classmethod
    def from_env(cls, **kwargs: Any) -> "WeatherClient":
        """Build a weather client using environment overrides when present."""
        if "base_url" not in kwargs:
            kwargs["base_url"] = os.getenv("NWS_API_BASE_URL", DEFAULT_NWS_API_BASE_URL)
        if "user_agent" not in kwargs:
            kwargs["user_agent"] = os.getenv("NWS_USER_AGENT", DEFAULT_NWS_USER_AGENT)
        return cls(**kwargs)

    async def close(self) -> None:
        """Close an owned aiohttp session."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def fetch_point(self, latitude: float, longitude: float) -> PointMetadata:
        """Resolve a lat/lon to NWS grid metadata and forecast URLs."""
        payload = await self.fetch_json(f"/points/{latitude},{longitude}")
        return self._parse_point(payload)

    async def fetch_forecast(self, latitude: float, longitude: float) -> Forecast:
        """Fetch the standard period forecast for a point."""
        point = await self.fetch_point(latitude, longitude)
        payload = await self.fetch_json(point.forecast_url)
        return self._parse_forecast(payload, point=point, forecast_type="daily")

    async def fetch_hourly_forecast(self, latitude: float, longitude: float) -> Forecast:
        """Fetch the hourly forecast for a point."""
        point = await self.fetch_point(latitude, longitude)
        payload = await self.fetch_json(point.hourly_forecast_url)
        return self._parse_forecast(payload, point=point, forecast_type="hourly")

    async def fetch_json(self, path_or_url: str) -> Mapping[str, Any]:
        """Fetch a JSON object from the NWS API."""
        session = await self._ensure_session()
        url = self._normalize_url(path_or_url)

        async with session.get(url, headers=self._request_headers()) as response:
            response.raise_for_status()
            payload = await response.json()

        if not isinstance(payload, Mapping):
            raise ValueError("NWS response payload was not a JSON object")
        return payload

    async def _ensure_session(self) -> ClientSession:
        if self._session is None:
            try:
                from aiohttp import ClientSession as AiohttpClientSession
            except ModuleNotFoundError as err:
                raise RuntimeError(
                    "aiohttp is required to create a live NWS session; "
                    "install aiohttp or inject an existing session"
                ) from err
            self._session = AiohttpClientSession()
        return self._session

    def _normalize_url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        if not path_or_url.startswith("/"):
            path_or_url = f"/{path_or_url}"
        return f"{self._base_url}{path_or_url}"

    def _request_headers(self) -> Dict[str, str]:
        return {
            "accept": self._accept,
            "user-agent": self._user_agent,
        }

    def _parse_point(self, payload: Mapping[str, Any]) -> PointMetadata:
        properties = payload.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError("NWS point response properties were missing")

        relative_location = properties.get("relativeLocation", {})
        if not isinstance(relative_location, Mapping):
            relative_location = {}
        relative_props = relative_location.get("properties", {})
        if not isinstance(relative_props, Mapping):
            relative_props = {}

        city = _optional_str(relative_props.get("city"))
        state = _optional_str(relative_props.get("state"))

        forecast_url = _required_str(properties.get("forecast"), "forecast URL")
        hourly_forecast_url = _required_str(
            properties.get("forecastHourly"), "hourly forecast URL"
        )

        return PointMetadata(
            latitude=float(payload.get("geometry", {}).get("coordinates", [0.0, 0.0])[1])
            if _has_geometry_coordinates(payload)
            else float(properties.get("latitude", 0.0)),
            longitude=float(payload.get("geometry", {}).get("coordinates", [0.0, 0.0])[0])
            if _has_geometry_coordinates(payload)
            else float(properties.get("longitude", 0.0)),
            city=city,
            state=state,
            timezone=_optional_str(properties.get("timeZone")),
            radar_station=_optional_str(properties.get("radarStation")),
            office=_optional_str(properties.get("cwa")),
            grid_x=_optional_int(properties.get("gridX")),
            grid_y=_optional_int(properties.get("gridY")),
            forecast_url=forecast_url,
            hourly_forecast_url=hourly_forecast_url,
            forecast_grid_data_url=_optional_str(properties.get("forecastGridData")),
            observation_stations_url=_optional_str(properties.get("observationStations")),
        )

    def _parse_forecast(
        self,
        payload: Mapping[str, Any],
        *,
        point: PointMetadata,
        forecast_type: str,
    ) -> Forecast:
        properties = payload.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError("NWS forecast response properties were missing")

        raw_periods = properties.get("periods")
        if not isinstance(raw_periods, list):
            raise ValueError("NWS forecast response periods were missing")

        periods = [self._parse_period(period) for period in raw_periods]
        return Forecast(
            forecast_type=forecast_type,
            generated_at=_parse_datetime(properties.get("generatedAt")),
            updated_at=_parse_datetime(properties.get("updateTime")),
            units=_optional_str(properties.get("units")),
            point=point,
            periods=periods,
        )

    def _parse_period(self, period: Any) -> ForecastPeriod:
        if not isinstance(period, Mapping):
            raise ValueError("forecast period item was not an object")

        precipitation = period.get("probabilityOfPrecipitation")
        pop_value: Optional[int] = None
        if isinstance(precipitation, Mapping):
            pop_value = _optional_int(precipitation.get("value"))

        return ForecastPeriod(
            number=int(period["number"]),
            name=str(period["name"]),
            start_time=dt.datetime.fromisoformat(str(period["startTime"])),
            end_time=dt.datetime.fromisoformat(str(period["endTime"])),
            is_daytime=bool(period.get("isDaytime")),
            temperature=_optional_int(period.get("temperature")),
            temperature_unit=_optional_str(period.get("temperatureUnit")),
            temperature_trend=_optional_str(period.get("temperatureTrend")),
            probability_of_precipitation=pop_value,
            wind_speed=_optional_str(period.get("windSpeed")),
            wind_direction=_optional_str(period.get("windDirection")),
            icon_url=_optional_str(period.get("icon")),
            short_forecast=_optional_str(period.get("shortForecast")),
            detailed_forecast=_optional_str(period.get("detailedForecast")),
        )


def _has_geometry_coordinates(payload: Mapping[str, Any]) -> bool:
    geometry = payload.get("geometry")
    if not isinstance(geometry, Mapping):
        return False
    coordinates = geometry.get("coordinates")
    return isinstance(coordinates, list) and len(coordinates) >= 2


def _parse_datetime(value: Any) -> Optional[dt.datetime]:
    if value in (None, ""):
        return None
    return dt.datetime.fromisoformat(str(value))


def _optional_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _required_str(value: Any, label: str) -> str:
    normalized = _optional_str(value)
    if normalized is None:
        raise ValueError(f"NWS response did not include {label}")
    return normalized


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)
