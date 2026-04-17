"""Tests for the precipitation historicals client."""

import datetime as dt
import unittest
from typing import Any, Dict, Optional

from services.precipitation import (
    PrecipitationAPIError,
    PrecipitationClient,
    window_to_api_time,
)


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], *, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> Dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, payload: Dict[str, Any], *, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.last_url: Optional[str] = None
        self.last_params: Optional[Dict[str, str]] = None
        self.last_headers: Optional[Dict[str, str]] = None

    def get(self, url: str, *, params: Dict[str, str], headers: Dict[str, str]) -> _FakeResponse:
        self.last_url = url
        self.last_params = params
        self.last_headers = headers
        return _FakeResponse(self.payload, status=self.status)


class WindowToApiTimeTests(unittest.TestCase):
    def test_accepts_iso_duration(self) -> None:
        self.assertEqual(window_to_api_time("P30D"), "P30D")

    def test_converts_shorthand_windows(self) -> None:
        self.assertEqual(window_to_api_time("12h"), "PT12H")
        self.assertEqual(window_to_api_time("90m"), "PT90M")
        self.assertEqual(window_to_api_time("7d"), "P7D")
        self.assertEqual(window_to_api_time("2w"), "P2W")

    def test_converts_timedelta(self) -> None:
        self.assertEqual(
            window_to_api_time(dt.timedelta(days=1, hours=2, minutes=5)),
            "P1DT2H5M",
        )

    def test_rejects_invalid_window(self) -> None:
        with self.assertRaises(ValueError):
            window_to_api_time("tomorrow")


class PrecipitationClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_samples_normalizes_response(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "id": "sample-1",
                    "properties": {
                        "time": "2026-04-17T17:25:00+00:00",
                        "value": "0.10",
                        "unit_of_measure": "in",
                        "approval_status": "Provisional",
                        "qualifier": None,
                    },
                },
                {
                    "id": "sample-2",
                    "properties": {
                        "time": "2026-04-17T17:30:00+00:00",
                        "value": "2.54",
                        "unit_of_measure": "mm",
                        "approval_status": "Approved",
                        "qualifier": "A",
                    },
                },
            ],
        }
        session = _FakeSession(payload)
        client = PrecipitationClient(api_key="test-key", session=session)

        samples = await client.fetch_samples("30D")

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].value_inches, 0.10)
        self.assertAlmostEqual(samples[1].value_inches, 0.1, places=4)
        self.assertEqual(samples[1].qualifier, "A")
        self.assertEqual(session.last_params["time"], "P30D")
        self.assertEqual(session.last_headers["x-api-key"], "test-key")

    async def test_summarize_returns_total_and_window_bounds(self) -> None:
        payload = {
            "features": [
                {
                    "id": "sample-1",
                    "properties": {
                        "time": "2026-04-17T17:25:00+00:00",
                        "value": "0.01",
                        "unit_of_measure": "in",
                        "approval_status": "Provisional",
                        "qualifier": None,
                    },
                },
                {
                    "id": "sample-2",
                    "properties": {
                        "time": "2026-04-17T17:30:00+00:00",
                        "value": "0.02",
                        "unit_of_measure": "in",
                        "approval_status": "Provisional",
                        "qualifier": None,
                    },
                },
            ]
        }
        client = PrecipitationClient(session=_FakeSession(payload))

        summary = await client.summarize(dt.timedelta(hours=6))

        self.assertEqual(summary.window, "PT6H")
        self.assertEqual(summary.sample_count, 2)
        self.assertAlmostEqual(summary.total_inches, 0.03, places=4)
        self.assertEqual(summary.started_at.isoformat(), "2026-04-17T17:25:00+00:00")
        self.assertEqual(summary.ended_at.isoformat(), "2026-04-17T17:30:00+00:00")

    async def test_fetch_raw_raises_structured_api_error(self) -> None:
        client = PrecipitationClient(
            session=_FakeSession(
                {
                    "error": {
                        "code": "API_KEY_INVALID",
                        "message": "An invalid api_key was supplied.",
                    }
                },
                status=403,
            )
        )

        with self.assertRaises(PrecipitationAPIError) as context:
            await client.fetch_raw("P1D")

        self.assertEqual(context.exception.status, 403)
        self.assertEqual(context.exception.code, "API_KEY_INVALID")
        self.assertIn("invalid api_key", context.exception.message)


if __name__ == "__main__":
    unittest.main()
