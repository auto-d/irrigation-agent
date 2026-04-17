"""Helpers for pulling historical precipitation from the USGS Water Data API."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from aiohttp import ClientSession
else:
    ClientSession = Any

DEFAULT_USGS_BASE_URL = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"
)
DEFAULT_USGS_PRECIP_PROPERTIES: Sequence[str] = (
    "time",
    "value",
    "unit_of_measure",
    "approval_status",
    "qualifier",
)
DEFAULT_USGS_TIME_SERIES_ID = "30259de5ae144951809547f26f5df5d5"
DEFAULT_USGS_LIMIT = 50_000
DEFAULT_USGS_ORIGIN = "https://waterdata.usgs.gov"
DEFAULT_USGS_REFERER = "https://waterdata.usgs.gov/"

WindowLike = Union[str, dt.timedelta]


@dataclass(frozen=True)
class PrecipitationSample:
    """One precipitation observation returned by the USGS continuous collection."""

    observed_at: dt.datetime
    value_inches: float
    unit_of_measure: str
    approval_status: Optional[str]
    qualifier: Optional[str]
    feature_id: Optional[str] = None


@dataclass(frozen=True)
class PrecipitationSummary:
    """Aggregated precipitation over a requested window."""

    window: str
    total_inches: float
    sample_count: int
    started_at: Optional[dt.datetime]
    ended_at: Optional[dt.datetime]
    samples: Sequence[PrecipitationSample]


@dataclass(frozen=True)
class PrecipitationAPIError(RuntimeError):
    """Structured upstream API failure."""

    status: int
    code: Optional[str]
    message: str

    def __str__(self) -> str:
        if self.code:
            return f"USGS API error {self.status} ({self.code}): {self.message}"
        return f"USGS API error {self.status}: {self.message}"


def window_to_api_time(window: WindowLike) -> str:
    """Normalize supported window values into the USGS `time` parameter format."""
    if isinstance(window, dt.timedelta):
        return _timedelta_to_duration(window)

    normalized = window.strip().upper()
    if not normalized:
        raise ValueError("window cannot be empty")

    if normalized.startswith("P"):
        return normalized

    suffix = normalized[-1]
    amount_text = normalized[:-1]
    if suffix not in {"M", "H", "D", "W"} or not amount_text:
        raise ValueError(
            "window must be a timedelta, an ISO-8601 duration such as 'P30D', "
            "or a shorthand like '30D', '12H', '90M', or '2W'"
        )

    try:
        amount = int(amount_text)
    except ValueError as err:
        raise ValueError(f"invalid window amount: {window}") from err

    if amount <= 0:
        raise ValueError("window must be greater than zero")

    if suffix in {"M", "H"}:
        return f"PT{amount}{suffix}"
    return f"P{amount}{suffix}"


def _timedelta_to_duration(window: dt.timedelta) -> str:
    total_seconds = int(window.total_seconds())
    if total_seconds <= 0:
        raise ValueError("window must be greater than zero")

    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)

    parts: List[str] = ["P"]
    if days:
        parts.append(f"{days}D")

    time_parts: List[str] = []
    if hours:
        time_parts.append(f"{hours}H")
    if minutes:
        time_parts.append(f"{minutes}M")
    if seconds:
        time_parts.append(f"{seconds}S")

    if time_parts:
        parts.append("T")
        parts.extend(time_parts)

    if parts == ["P"]:
        return "PT0S"
    return "".join(parts)


class PrecipitationClient:
    """Thin async wrapper around the USGS precipitation historicals endpoint."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        time_series_id: str = DEFAULT_USGS_TIME_SERIES_ID,
        base_url: str = DEFAULT_USGS_BASE_URL,
        properties: Sequence[str] = DEFAULT_USGS_PRECIP_PROPERTIES,
        limit: int = DEFAULT_USGS_LIMIT,
        session: Optional[ClientSession] = None,
        user_agent: str = "agent-bakeoff/precipitation-client",
    ) -> None:
        self._api_key = api_key or os.getenv("USGS_WATERDATA_API_KEY")
        self._time_series_id = time_series_id
        self._base_url = base_url
        self._properties = tuple(properties)
        self._limit = limit
        self._session = session
        self._owns_session = session is None
        self._user_agent = user_agent

    async def __aenter__(self) -> "PrecipitationClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @classmethod
    def from_env(cls, **kwargs: Any) -> "PrecipitationClient":
        """Build a client from environment variables when available."""
        if "api_key" not in kwargs:
            kwargs["api_key"] = os.getenv("USGS_WATERDATA_API_KEY")
        if "time_series_id" not in kwargs:
            kwargs["time_series_id"] = os.getenv(
                "USGS_PRECIP_TIME_SERIES_ID", DEFAULT_USGS_TIME_SERIES_ID
            )
        return cls(**kwargs)

    async def close(self) -> None:
        """Close an owned aiohttp session."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def fetch_raw(self, window: WindowLike) -> Mapping[str, Any]:
        """Return the raw JSON payload for a requested historical window."""
        session = await self._ensure_session()
        params = {
            "f": "json",
            "limit": str(self._limit),
            "properties": ",".join(self._properties),
            "time_series_id": self._time_series_id,
            "time": window_to_api_time(window),
        }

        async with session.get(
            self._base_url,
            params=params,
            headers=self._request_headers(),
        ) as response:
            payload = await response.json()
            if response.status >= 400:
                raise self._build_api_error(response.status, payload)

        if not isinstance(payload, Mapping):
            raise ValueError("USGS response payload was not a JSON object")
        return payload

    async def fetch_samples(self, window: WindowLike) -> List[PrecipitationSample]:
        """Return normalized precipitation samples for a requested window."""
        payload = await self.fetch_raw(window)
        features = payload.get("features", [])
        if not isinstance(features, Iterable):
            raise ValueError("USGS response did not contain a features list")

        return [self._parse_feature(feature) for feature in features]

    async def total_precipitation(self, window: WindowLike) -> float:
        """Return the summed rainfall depth in inches for a window."""
        samples = await self.fetch_samples(window)
        return sum(sample.value_inches for sample in samples)

    async def summarize(self, window: WindowLike) -> PrecipitationSummary:
        """Return normalized observations and a simple aggregate summary."""
        normalized_window = window_to_api_time(window)
        samples = await self.fetch_samples(normalized_window)
        started_at = samples[0].observed_at if samples else None
        ended_at = samples[-1].observed_at if samples else None
        return PrecipitationSummary(
            window=normalized_window,
            total_inches=sum(sample.value_inches for sample in samples),
            sample_count=len(samples),
            started_at=started_at,
            ended_at=ended_at,
            samples=samples,
        )

    async def _ensure_session(self) -> ClientSession:
        if self._session is None:
            try:
                from aiohttp import ClientSession as AiohttpClientSession
            except ModuleNotFoundError as err:
                raise RuntimeError(
                    "aiohttp is required to create a live USGS session; "
                    "install aiohttp or inject an existing session"
                ) from err
            self._session = AiohttpClientSession()
        return self._session

    def _request_headers(self) -> Dict[str, str]:
        headers = {
            "accept": "*/*",
            "origin": DEFAULT_USGS_ORIGIN,
            "referer": DEFAULT_USGS_REFERER,
            "user-agent": self._user_agent,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    def _parse_feature(self, feature: Any) -> PrecipitationSample:
        if not isinstance(feature, Mapping):
            raise ValueError("feature item was not an object")

        properties = feature.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError("feature properties were missing")

        observed_at = dt.datetime.fromisoformat(str(properties["time"]))
        raw_value = properties.get("value", "0")
        unit = str(properties.get("unit_of_measure", "in"))

        value_inches = self._to_inches(raw_value, unit)
        qualifier = properties.get("qualifier")
        approval_status = properties.get("approval_status")

        return PrecipitationSample(
            observed_at=observed_at,
            value_inches=value_inches,
            unit_of_measure=unit,
            approval_status=str(approval_status) if approval_status is not None else None,
            qualifier=str(qualifier) if qualifier is not None else None,
            feature_id=str(feature.get("id")) if feature.get("id") is not None else None,
        )

    def _to_inches(self, raw_value: Any, unit: str) -> float:
        value = float(raw_value)
        normalized_unit = unit.strip().lower()
        if normalized_unit in {"in", "inch", "inches"}:
            return value
        if normalized_unit in {"mm", "millimeter", "millimeters"}:
            return value / 25.4
        raise ValueError(f"unsupported precipitation unit: {unit}")

    def _build_api_error(self, status: int, payload: Any) -> PrecipitationAPIError:
        if isinstance(payload, Mapping):
            error = payload.get("error")
            if isinstance(error, Mapping):
                code = error.get("code")
                message = error.get("message")
                if message is not None:
                    return PrecipitationAPIError(
                        status=status,
                        code=str(code) if code is not None else None,
                        message=str(message),
                    )
        return PrecipitationAPIError(
            status=status,
            code=None,
            message="request failed",
        )
