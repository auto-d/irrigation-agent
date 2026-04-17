"""Tests for the NWS weather client."""

import unittest
from typing import Any, Dict, Optional

from services.weather import WeatherClient


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> Dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, payloads_by_url: Dict[str, Dict[str, Any]]) -> None:
        self.payloads_by_url = payloads_by_url
        self.calls = []

    def get(self, url: str, *, headers: Dict[str, str]) -> _FakeResponse:
        self.calls.append({"url": url, "headers": headers})
        return _FakeResponse(self.payloads_by_url[url])


class WeatherClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_forecast_resolves_points_then_forecast(self) -> None:
        session = _FakeSession(
            {
                "https://api.weather.gov/points/42.8864,-78.8784": {
                    "geometry": {"type": "Point", "coordinates": [-78.8784, 42.8864]},
                    "properties": {
                        "cwa": "BUF",
                        "gridX": 65,
                        "gridY": 42,
                        "timeZone": "America/New_York",
                        "radarStation": "KBUF",
                        "forecast": "https://api.weather.gov/gridpoints/BUF/65,42/forecast",
                        "forecastHourly": "https://api.weather.gov/gridpoints/BUF/65,42/forecast/hourly",
                        "forecastGridData": "https://api.weather.gov/gridpoints/BUF/65,42",
                        "observationStations": "https://api.weather.gov/gridpoints/BUF/65,42/stations",
                        "relativeLocation": {
                            "properties": {
                                "city": "Buffalo",
                                "state": "NY",
                            }
                        },
                    },
                },
                "https://api.weather.gov/gridpoints/BUF/65,42/forecast": {
                    "properties": {
                        "units": "us",
                        "generatedAt": "2026-04-17T18:00:00+00:00",
                        "updateTime": "2026-04-17T17:30:00+00:00",
                        "periods": [
                            {
                                "number": 1,
                                "name": "Tonight",
                                "startTime": "2026-04-17T18:00:00-04:00",
                                "endTime": "2026-04-18T06:00:00-04:00",
                                "isDaytime": False,
                                "temperature": 44,
                                "temperatureUnit": "F",
                                "temperatureTrend": None,
                                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 35},
                                "windSpeed": "5 mph",
                                "windDirection": "NW",
                                "icon": "https://api.weather.gov/icons/land/night/rain",
                                "shortForecast": "Chance Rain Showers",
                                "detailedForecast": "A chance of rain showers before midnight.",
                            }
                        ],
                    }
                },
            }
        )
        client = WeatherClient(session=session, user_agent="agent-bakeoff-tests")

        forecast = await client.fetch_forecast(42.8864, -78.8784)

        self.assertEqual(forecast.forecast_type, "daily")
        self.assertEqual(forecast.point.office, "BUF")
        self.assertEqual(forecast.point.city, "Buffalo")
        self.assertEqual(forecast.point.state, "NY")
        self.assertEqual(len(forecast.periods), 1)
        self.assertEqual(forecast.periods[0].probability_of_precipitation, 35)
        self.assertEqual(session.calls[0]["url"], "https://api.weather.gov/points/42.8864,-78.8784")
        self.assertEqual(
            session.calls[1]["url"],
            "https://api.weather.gov/gridpoints/BUF/65,42/forecast",
        )
        self.assertEqual(session.calls[0]["headers"]["user-agent"], "agent-bakeoff-tests")

    async def test_fetch_hourly_forecast_uses_hourly_endpoint(self) -> None:
        session = _FakeSession(
            {
                "https://api.weather.gov/points/40.7128,-74.006": {
                    "geometry": {"type": "Point", "coordinates": [-74.0060, 40.7128]},
                    "properties": {
                        "cwa": "OKX",
                        "gridX": 33,
                        "gridY": 37,
                        "timeZone": "America/New_York",
                        "radarStation": "KOKX",
                        "forecast": "https://api.weather.gov/gridpoints/OKX/33,37/forecast",
                        "forecastHourly": "https://api.weather.gov/gridpoints/OKX/33,37/forecast/hourly",
                        "relativeLocation": {
                            "properties": {
                                "city": "New York",
                                "state": "NY",
                            }
                        },
                    },
                },
                "https://api.weather.gov/gridpoints/OKX/33,37/forecast/hourly": {
                    "properties": {
                        "units": "us",
                        "generatedAt": "2026-04-17T18:00:00+00:00",
                        "updateTime": "2026-04-17T17:30:00+00:00",
                        "periods": [
                            {
                                "number": 1,
                                "name": "",
                                "startTime": "2026-04-17T15:00:00-04:00",
                                "endTime": "2026-04-17T16:00:00-04:00",
                                "isDaytime": True,
                                "temperature": 59,
                                "temperatureUnit": "F",
                                "temperatureTrend": None,
                                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": None},
                                "windSpeed": "8 mph",
                                "windDirection": "W",
                                "icon": None,
                                "shortForecast": "Partly Sunny",
                                "detailedForecast": "",
                            }
                        ],
                    }
                },
            }
        )
        client = WeatherClient(session=session)

        forecast = await client.fetch_hourly_forecast(40.7128, -74.0060)

        self.assertEqual(forecast.forecast_type, "hourly")
        self.assertEqual(forecast.periods[0].temperature, 59)
        self.assertIsNone(forecast.periods[0].probability_of_precipitation)
        self.assertIsNone(forecast.periods[0].detailed_forecast)
        self.assertEqual(
            session.calls[1]["url"],
            "https://api.weather.gov/gridpoints/OKX/33,37/forecast/hourly",
        )

    async def test_fetch_point_requires_forecast_links(self) -> None:
        session = _FakeSession(
            {
                "https://api.weather.gov/points/1.0,2.0": {
                    "properties": {}
                }
            }
        )
        client = WeatherClient(session=session)

        with self.assertRaises(ValueError):
            await client.fetch_point(1.0, 2.0)


if __name__ == "__main__":
    unittest.main()
