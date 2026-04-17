#!/usr/bin/env python3
"""Explore the live Orbit B-hyve API shape for planning-agent integration."""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from aiohttp import ClientResponseError, ClientSession

from services.bhyve.vendor.pybhyve import Client
from services.bhyve.vendor.pybhyve.const import (
    API_HOST,
    DEVICE_HISTORY_PATH,
    DEVICES_PATH,
    TIMER_PROGRAMS_PATH,
)


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


async def _noop_websocket_handler(data) -> None:
    logging.getLogger(__name__).debug("websocket event: %s", data)


def _compact_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _summarize_device(device: Dict[str, Any]) -> Dict[str, Any]:
    status = device.get("status") or {}
    return {
        "id": device.get("id"),
        "name": device.get("name"),
        "type": device.get("type"),
        "hardware_version": device.get("hardware_version"),
        "is_connected": device.get("is_connected"),
        "run_mode": status.get("run_mode"),
        "watering_status": status.get("watering_status") or status.get("watering-status"),
        "watering_statuses": status.get("watering_statuses"),
        "next_start_time": status.get("next_start_time"),
        "next_start_programs": status.get("next_start_programs"),
        "rain_delay": status.get("rain_delay"),
        "smart_watering_enabled": device.get("smart_watering_enabled"),
        "num_stations": device.get("num_stations"),
        "reference": device.get("reference"),
    }


async def _fetch_endpoint(
    client: Client, label: str, endpoint: str, *, params: Optional[Dict[str, str]] = None
) -> Any:
    url = f"{API_HOST}{endpoint}"
    logging.info("%s -> GET %s params=%s", label, url, params or {})
    try:
        payload = await client._request("get", endpoint, params=params)
    except ClientResponseError as err:
        logging.error("%s failed: %s", label, err)
        return None
    except Exception as err:
        logging.error("%s failed: %s", label, err)
        return None

    if isinstance(payload, list):
        logging.info("%s returned list[%d]", label, len(payload))
    elif isinstance(payload, dict):
        logging.info("%s returned dict keys=%s", label, sorted(payload.keys()))
    else:
        logging.info("%s returned %s", label, type(payload).__name__)
    return payload


async def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

    username = _get_required_env("BHYVE_EMAIL")
    password = _get_required_env("BHYVE_PASSWORD")

    async with ClientSession() as session:
        client = Client(
            username,
            password,
            asyncio.get_running_loop(),
            session,
            _noop_websocket_handler,
        )

        try:
            await client.login()

            devices = await _fetch_endpoint(client, "devices", DEVICES_PATH, params={"t": "explore"})
            print("=== DEVICE SUMMARIES ===")
            print(_compact_json([_summarize_device(device) for device in devices or []]))

            print("\n=== TIMER PROGRAMS ===")
            timer_programs = await _fetch_endpoint(
                client, "timer_programs", TIMER_PROGRAMS_PATH, params={"t": "explore"}
            )
            print(_compact_json(timer_programs))

            print("\n=== DEVICE HISTORIES ===")
            histories: Dict[str, List[Any] | Dict[str, Any] | None] = {}
            for device in devices or []:
                device_id = device.get("id")
                histories[device_id] = await _fetch_endpoint(
                    client,
                    f"history[{device_id}]",
                    DEVICE_HISTORY_PATH.format(device_id),
                    params={"page": "1", "per-page": "10", "t": "explore"},
                )
            print(_compact_json(histories))

            print("\n=== LANDSCAPES ===")
            landscapes: Dict[str, Any] = {}
            for device in devices or []:
                if device.get("type") != "sprinkler_timer":
                    continue
                device_id = device.get("id")
                endpoint = f"/v1/landscape_descriptions/{device_id}"
                landscapes[device_id] = await _fetch_endpoint(
                    client, f"landscapes[{device_id}]", endpoint, params={"t": "explore"}
                )
            print(_compact_json(landscapes))
            return 0
        finally:
            await client.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
