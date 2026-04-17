#!/usr/bin/env python3
"""Thin B-hyve wrapper for planning-agent state reads and valve control.

Preferred direct-use surface for planner code:

- ``BHyveController`` for async context-managed use
- ``controller_from_env()`` to build from ``BHYVE_EMAIL`` and ``BHYVE_PASSWORD``
- ``turn_on()``, ``turn_off()``, and ``cycle()`` for actuation

The CLI remains available for debugging and manual verification.
"""

import argparse
import asyncio
import datetime as dt
import json
import logging
import math
import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from aiohttp import ClientSession

from services.bhyve.vendor.pybhyve import Client
from services.bhyve.vendor.pybhyve.const import (
    API_HOST,
    DEVICE_HISTORY_PATH,
    DEVICES_PATH,
    TIMER_PROGRAMS_PATH,
)

LANDSCAPE_DESCRIPTIONS_PATH = "/v1/landscape_descriptions"

EVENT_CHANGE_MODE = "change_mode"
EVENT_DEVICE_IDLE = "device_idle"
EVENT_PROGRAM_CHANGED = "program_changed"
EVENT_RAIN_DELAY = "rain_delay"
EVENT_SET_MANUAL_PRESET_RUNTIME = "set_manual_preset_runtime"
EVENT_WATERING_COMPLETE = "watering_complete"
EVENT_WATERING_IN_PROGRESS = "watering_in_progress_notification"

__all__ = [
    "BHyveController",
    "controller_from_env",
]


def _is_active_watering_status(status: Dict[str, Any]) -> bool:
    """Return true when status looks like an active watering run."""
    if status.get("run_mode") == "manual":
        return True

    watering_status = status.get("watering_status")
    if not isinstance(watering_status, dict):
        return False

    # HT25 often reports {"clear_on_idle": true} even when idle. Treat only
    # substantive runtime fields as evidence of active watering.
    for key in ("current_station", "run_time", "started_watering_station_at", "stations", "program"):
        value = watering_status.get(key)
        if value not in (None, [], {}, ""):
            return True
    return False


def _seconds_to_run_minutes(seconds: float) -> float:
    """Convert a desired on-window in seconds to a B-hyve run_time in minutes."""
    if seconds <= 0:
        raise ValueError("seconds must be greater than zero")
    return float(max(1, math.ceil(seconds / 60.0)))


class BHyveController:
    """Async wrapper around the vendored B-hyve client.

    Example:

    ```python
    import asyncio
    from services.bhyve.controller import controller_from_env

    async def main():
        async with controller_from_env() as controller:
            devices = controller.list_sprinkler_devices()
            device_id = devices[0]["id"]
            await controller.cycle(device_id, seconds=5)

    asyncio.run(main())
    ```
    """

    def __init__(
        self,
        username: str,
        password: str,
        *,
        session: Optional[ClientSession] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._username = username
        self._password = password
        self._session = session
        self._owns_session = session is None
        self._logger = logger or logging.getLogger(__name__)
        self._client: Optional[Client] = None
        self._devices_by_id: Dict[str, Dict[str, Any]] = {}
        self._programs_by_id: Dict[str, Dict[str, Any]] = {}
        self._last_event: Optional[Dict[str, Any]] = None
        self._event_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

    async def __aenter__(self) -> "BHyveController":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        """Log in and start the websocket listener."""
        if self._session is None:
            self._session = ClientSession()
        if self._client is not None:
            return

        self._client = Client(
            self._username,
            self._password,
            asyncio.get_running_loop(),
            self._session,
            self._handle_event,
        )
        await self._client.login()
        await self._wait_for_websocket_ready()
        await self.refresh()

    async def close(self) -> None:
        """Close websocket and owned aiohttp session."""
        if self._client is not None:
            try:
                await self._client.stop()
            except AttributeError:
                self._logger.debug("Underlying websocket was not fully initialized during stop")
            self._client = None
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def refresh(self) -> List[Dict[str, Any]]:
        """Refresh and cache devices and programs."""
        client = self._require_client()
        devices = await client.devices
        programs = await client.timer_programs
        self._devices_by_id = {str(device["id"]): deepcopy(device) for device in devices}
        self._programs_by_id = {str(program["id"]): deepcopy(program) for program in programs}
        return self.get_devices()

    async def refresh_device(self, device_id: str) -> Dict[str, Any]:
        """Refresh one device directly from the live devices endpoint."""
        devices = await self._request(
            "get",
            DEVICES_PATH,
            params={"t": str(asyncio.get_running_loop().time())},
        )
        self._devices_by_id = {str(device["id"]): deepcopy(device) for device in devices}
        return self.get_device(device_id)

    def get_devices(self) -> List[Dict[str, Any]]:
        """Return cached devices."""
        return deepcopy(list(self._devices_by_id.values()))

    def get_device(self, device_id: str) -> Dict[str, Any]:
        """Return one cached device."""
        try:
            return deepcopy(self._devices_by_id[device_id])
        except KeyError as err:
            raise KeyError(f"Unknown device_id: {device_id}") from err

    def get_device_status(self, device_id: str) -> Dict[str, Any]:
        """Return cached status for a device."""
        return deepcopy(self.get_device(device_id).get("status", {}))

    def list_sprinkler_devices(self) -> List[Dict[str, Any]]:
        """Return cached sprinkler timer devices."""
        return [
            device for device in self.get_devices() if device.get("type") == "sprinkler_timer"
        ]

    def get_programs(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return cached programs, optionally filtered by device."""
        programs = list(self._programs_by_id.values())
        if device_id is not None:
            programs = [program for program in programs if program.get("device_id") == device_id]
        return deepcopy(programs)

    def get_program(self, program_id: str) -> Dict[str, Any]:
        """Return one cached program."""
        try:
            return deepcopy(self._programs_by_id[program_id])
        except KeyError as err:
            raise KeyError(f"Unknown program_id: {program_id}") from err

    async def get_history(self, device_id: str) -> Any:
        """Fetch device watering history from REST."""
        return await self._request(
            "get",
            DEVICE_HISTORY_PATH.format(device_id),
            params={"page": "1", "per-page": "10", "t": str(asyncio.get_running_loop().time())},
        )

    async def get_landscapes(self, device_id: str) -> Any:
        """Fetch landscape descriptions for a sprinkler device."""
        return await self._request(
            "get",
            f"{LANDSCAPE_DESCRIPTIONS_PATH}/{device_id}",
            params={"t": str(asyncio.get_running_loop().time())},
        )

    async def update_program(
        self,
        program_id: str,
        *,
        enabled: Optional[bool] = None,
        start_times: Optional[List[str]] = None,
        frequency: Optional[Dict[str, Any]] = None,
        budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update a non-smart program via REST."""
        program = self.get_program(program_id)
        if program.get("is_smart_program"):
            raise ValueError("Cannot update a smart watering program")

        allowed_keys = {
            "budget",
            "device_id",
            "enabled",
            "frequency",
            "id",
            "name",
            "program",
            "program_start_date",
            "run_times",
            "start_times",
        }
        payload = {key: value for key, value in program.items() if key in allowed_keys}
        if enabled is not None:
            payload["enabled"] = enabled
        if start_times is not None:
            payload["start_times"] = start_times
        if frequency is not None:
            payload["frequency"] = frequency
        if budget is not None:
            payload["budget"] = budget

        await self._request(
            "put",
            f"{TIMER_PROGRAMS_PATH}/{program_id}",
            json={"sprinkler_timer_program": payload},
        )
        await self.refresh()
        return self.get_program(program_id)

    async def set_rain_delay(self, device_id: str, hours: int) -> Dict[str, Any]:
        """Set or clear a rain delay via websocket."""
        await self._send_message(
            {"event": EVENT_RAIN_DELAY, "device_id": device_id, "delay": hours}
        )
        return self.get_device_status(device_id)

    async def set_mode(self, device_id: str, mode: str) -> Dict[str, Any]:
        """Set device mode via websocket."""
        if mode not in {"auto", "manual", "off"}:
            raise ValueError(f"Unsupported mode: {mode}")
        await self._send_message(
            {"event": EVENT_CHANGE_MODE, "device_id": device_id, "mode": mode}
        )
        return self.get_device_status(device_id)

    async def start_program(
        self, device_id: str, program_id: str, *, wait_for_update: bool = False, timeout: float = 15.0
    ) -> Dict[str, Any]:
        """Start a saved program via websocket."""
        program = self.get_program(program_id)
        if program.get("device_id") != device_id:
            raise ValueError(f"Program {program_id} does not belong to device {device_id}")

        payload = {
            "event": EVENT_CHANGE_MODE,
            "mode": "manual",
            "device_id": device_id,
            "timestamp": self._utc_timestamp(),
            "program": program.get("program"),
        }
        if payload["program"] is None:
            raise ValueError(f"Program {program_id} has no runnable program payload")

        await self._send_message(payload)
        if wait_for_update:
            event = await self.wait_for_event(
                lambda evt: evt.get("event") in {EVENT_WATERING_IN_PROGRESS, EVENT_CHANGE_MODE}
                and evt.get("device_id") == device_id,
                timeout=timeout,
            )
            return event
        return self.get_device_status(device_id)

    async def start_watering(
        self,
        device_id: str,
        *,
        station: int = 1,
        minutes: float = 1.0,
        wait_for_update: bool = False,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """Open the valve and start manual watering."""
        self.get_device(device_id)
        payload = {
            "event": EVENT_CHANGE_MODE,
            "mode": "manual",
            "device_id": device_id,
            "timestamp": self._utc_timestamp(),
            "stations": [{"station": station, "run_time": minutes}],
        }
        await self._send_message(payload)
        if wait_for_update:
            return await self.wait_for_event(
                lambda evt: evt.get("event") == EVENT_WATERING_IN_PROGRESS
                and evt.get("device_id") == device_id,
                timeout=timeout,
            )
        return self.get_device_status(device_id)

    async def turn_on(
        self,
        device_id: str,
        *,
        seconds: float,
        station: int = 1,
        wait_for_update: bool = False,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """Turn the valve on, mapping requested seconds to B-hyve runtime minutes."""
        return await self.start_watering(
            device_id,
            station=station,
            minutes=_seconds_to_run_minutes(seconds),
            wait_for_update=wait_for_update,
            timeout=timeout,
        )

    async def stop_watering(
        self,
        device_id: str,
        *,
        wait_for_update: bool = False,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """Close the valve by sending an empty stations payload."""
        self.get_device(device_id)
        payload = {
            "event": EVENT_CHANGE_MODE,
            "mode": "manual",
            "device_id": device_id,
            "timestamp": self._utc_timestamp(),
            "stations": [],
        }
        await self._send_message(payload)
        if wait_for_update:
            return await self.wait_for_event(
                lambda evt: evt.get("event") in {EVENT_WATERING_COMPLETE, EVENT_DEVICE_IDLE}
                and evt.get("device_id") == device_id,
                timeout=timeout,
            )
        return self.get_device_status(device_id)

    async def set_manual_preset_runtime(self, device_id: str, minutes: int) -> Dict[str, Any]:
        """Set device manual preset runtime in minutes."""
        await self._send_message(
            {
                "event": EVENT_SET_MANUAL_PRESET_RUNTIME,
                "device_id": device_id,
                "seconds": minutes * 60,
            }
        )
        return self.get_device_status(device_id)

    async def turn_off(
        self,
        device_id: str,
        *,
        wait_for_update: bool = False,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """Turn the valve off immediately."""
        return await self.stop_watering(
            device_id,
            wait_for_update=wait_for_update,
            timeout=timeout,
        )

    async def cycle(
        self,
        device_id: str,
        *,
        seconds: float,
        station: int = 1,
        wait_for_update: bool = False,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """Turn the valve on for a fixed number of seconds, then turn it off."""
        result: Dict[str, Any] = {
            "device_id": device_id,
            "requested_on_seconds": seconds,
            "requested_run_minutes": _seconds_to_run_minutes(seconds),
            "wait_for_update": wait_for_update,
            "start": {"sent": True},
            "stop": {"sent": False},
        }

        try:
            result["start"]["result"] = await self.turn_on(
                device_id,
                seconds=seconds,
                station=station,
                wait_for_update=wait_for_update,
                timeout=timeout,
            )
        except TimeoutError as err:
            result["start"]["result"] = {
                "observation": "unconfirmed",
                "error": str(err),
            }

        try:
            await asyncio.sleep(seconds)
        finally:
            try:
                result["stop"] = {
                    "sent": True,
                    "result": await self.turn_off(
                        device_id,
                        wait_for_update=wait_for_update,
                        timeout=timeout,
                    ),
                }
            except TimeoutError as err:
                result["stop"] = {
                    "sent": True,
                    "result": {
                        "observation": "unconfirmed",
                        "error": str(err),
                    },
                }
        return result

    async def poll_device_status(
        self,
        device_id: str,
        predicate: Callable[[Dict[str, Any]], bool],
        *,
        timeout: float = 15.0,
        interval: float = 1.0,
    ) -> Dict[str, Any]:
        """Poll live device state until a predicate matches."""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            device = await self.refresh_device(device_id)
            if predicate(device):
                return deepcopy(device)
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Timed out waiting for matching polled device state")
            await asyncio.sleep(interval)

    async def wait_for_event(
        self, predicate: Callable[[Dict[str, Any]], bool], *, timeout: float = 15.0
    ) -> Dict[str, Any]:
        """Wait for the next websocket event matching a predicate."""
        if self._last_event is not None and predicate(self._last_event):
            return deepcopy(self._last_event)

        while True:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=timeout)
            except asyncio.TimeoutError as err:
                raise TimeoutError("Timed out waiting for matching websocket event") from err
            if predicate(event):
                return deepcopy(event)

    async def _request(
        self, method: str, endpoint: str, *, params: Optional[Dict[str, str]] = None, json: Any = None
    ) -> Any:
        client = self._require_client()
        return await client._request(method, endpoint, params=params, json=json)

    async def _send_message(self, payload: Dict[str, Any]) -> None:
        client = self._require_client()
        await self._wait_for_websocket_ready()
        self._logger.info("Sending websocket payload: %s", json.dumps(payload, sort_keys=True))
        await client.send_message(payload)

    async def _handle_event(self, data: Dict[str, Any]) -> None:
        self._last_event = deepcopy(data)
        await self._event_queue.put(deepcopy(data))
        event = data.get("event")
        device_id = data.get("device_id")

        self._logger.info("Received websocket event: %s", json.dumps(data, sort_keys=True))

        if event == EVENT_PROGRAM_CHANGED:
            program = data.get("program")
            if isinstance(program, dict) and "id" in program:
                self._programs_by_id[str(program["id"])] = deepcopy(program)
            return

        if not device_id or device_id not in self._devices_by_id:
            return

        device = self._devices_by_id[device_id]
        status = device.setdefault("status", {})

        if event == EVENT_CHANGE_MODE:
            status.update(data)
        elif event == EVENT_DEVICE_IDLE:
            status["run_mode"] = "off"
            status.pop("watering_status", None)
        elif event == EVENT_WATERING_IN_PROGRESS:
            status["run_mode"] = data.get("mode", "manual")
            status["watering_status"] = {
                "current_station": data.get("current_station"),
                "program": data.get("program"),
                "run_time": data.get("run_time"),
                "started_watering_station_at": data.get("started_watering_station_at"),
                "stations": data.get("stations", []),
            }
        elif event == EVENT_WATERING_COMPLETE:
            status["run_mode"] = "off"
            status.pop("watering_status", None)
        elif event == EVENT_RAIN_DELAY:
            status["rain_delay"] = data.get("delay", 0)
        elif event == EVENT_SET_MANUAL_PRESET_RUNTIME:
            status["manual_preset_runtime"] = data.get("runtime")

    def _require_client(self) -> Client:
        if self._client is None:
            raise RuntimeError("Controller is not connected")
        return self._client

    async def _wait_for_websocket_ready(self, *, timeout: float = 15.0) -> None:
        client = self._require_client()
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            websocket = getattr(client, "_websocket", None)
            ws = getattr(websocket, "_ws", None)
            if ws is not None and not ws.closed:
                return
            await asyncio.sleep(0.2)
        raise TimeoutError("Timed out waiting for B-hyve websocket connection")

    @staticmethod
    def _utc_timestamp() -> str:
        return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def controller_from_env(*, logger: Optional[logging.Logger] = None) -> BHyveController:
    """Build a controller from ``BHYVE_EMAIL`` and ``BHYVE_PASSWORD``.

    This is the intended constructor for planner code running inside the repo.
    """
    return BHyveController(
        _get_required_env("BHYVE_EMAIL"),
        _get_required_env("BHYVE_PASSWORD"),
        logger=logger,
    )


def _find_target_device_id(
    controller: BHyveController, explicit_device_id: Optional[str]
) -> str:
    if explicit_device_id:
        return explicit_device_id
    sprinklers = controller.list_sprinkler_devices()
    if len(sprinklers) != 1:
        raise SystemExit(
            f"Expected exactly one sprinkler_timer device, found {len(sprinklers)}. "
            "Pass --device-id explicitly."
        )
    return str(sprinklers[0]["id"])


async def _run_cli(args: argparse.Namespace) -> int:
    logging.basicConfig(level=args.log_level.upper())

    async with controller_from_env() as controller:
        if args.command == "devices":
            print(json.dumps(controller.get_devices(), indent=2, sort_keys=True))
            return 0

        if args.command == "status":
            device_id = _find_target_device_id(controller, args.device_id)
            print(json.dumps(controller.get_device(device_id), indent=2, sort_keys=True))
            return 0

        if args.command == "programs":
            device_id = _find_target_device_id(controller, args.device_id)
            print(json.dumps(controller.get_programs(device_id), indent=2, sort_keys=True))
            return 0

        if args.command == "history":
            device_id = _find_target_device_id(controller, args.device_id)
            print(json.dumps(await controller.get_history(device_id), indent=2, sort_keys=True))
            return 0

        if args.command == "landscapes":
            device_id = _find_target_device_id(controller, args.device_id)
            print(json.dumps(await controller.get_landscapes(device_id), indent=2, sort_keys=True))
            return 0

        if args.command == "on":
            device_id = _find_target_device_id(controller, args.device_id)
            result: Dict[str, Any] = {
                "sent": True,
                "device_id": device_id,
                "requested_on_seconds": args.seconds,
                "requested_run_minutes": _seconds_to_run_minutes(args.seconds),
            }
            command_result = await controller.turn_on(
                device_id,
                seconds=args.seconds,
                station=args.station,
                wait_for_update=args.wait,
                timeout=args.timeout,
            )
            result["command_result"] = command_result
            if args.poll:
                try:
                    polled = await controller.poll_device_status(
                        device_id,
                        lambda device: _is_active_watering_status(device.get("status") or {}),
                        timeout=args.timeout,
                        interval=args.poll_interval,
                    )
                    result["polled_status"] = {
                        "observation": "confirmed",
                        "device": polled,
                    }
                except TimeoutError as err:
                    result["polled_status"] = {
                        "observation": "unconfirmed",
                        "error": str(err),
                    }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        if args.command == "off":
            device_id = _find_target_device_id(controller, args.device_id)
            result = {
                "sent": True,
                "device_id": device_id,
            }
            command_result = await controller.turn_off(
                device_id,
                wait_for_update=args.wait,
                timeout=args.timeout,
            )
            result["command_result"] = command_result
            if args.poll:
                try:
                    polled = await controller.poll_device_status(
                        device_id,
                        lambda device: (
                            (device.get("status") or {}).get("run_mode") in {"auto", "off"}
                            and not _is_active_watering_status(device.get("status") or {})
                        ),
                        timeout=args.timeout,
                        interval=args.poll_interval,
                    )
                    result["polled_status"] = {
                        "observation": "confirmed",
                        "device": polled,
                    }
                except TimeoutError as err:
                    result["polled_status"] = {
                        "observation": "unconfirmed",
                        "error": str(err),
                    }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        if args.command == "cycle":
            device_id = _find_target_device_id(controller, args.device_id)
            cycle_result = await controller.cycle(
                device_id,
                seconds=args.seconds,
                station=args.station,
                wait_for_update=args.wait,
                timeout=args.timeout,
            )
            cycle_result["poll_for_status"] = args.poll
            if args.poll:
                try:
                    polled = await controller.poll_device_status(
                        device_id,
                        lambda device: _is_active_watering_status(device.get("status") or {}),
                        timeout=args.timeout,
                        interval=args.poll_interval,
                    )
                    cycle_result["start"]["polled_status"] = {
                        "observation": "confirmed",
                        "device": polled,
                    }
                except TimeoutError as err:
                    cycle_result["start"]["polled_status"] = {
                        "observation": "unconfirmed",
                        "error": str(err),
                    }
            if args.poll:
                try:
                    polled = await controller.poll_device_status(
                        device_id,
                        lambda device: (
                            (device.get("status") or {}).get("run_mode") in {"auto", "off"}
                            and not _is_active_watering_status(device.get("status") or {})
                        ),
                        timeout=args.timeout,
                        interval=args.poll_interval,
                    )
                    cycle_result["stop"]["polled_status"] = {
                        "observation": "confirmed",
                        "device": polled,
                    }
                except TimeoutError as err:
                    cycle_result["stop"]["polled_status"] = {
                        "observation": "unconfirmed",
                        "error": str(err),
                    }

            print(json.dumps(cycle_result, indent=2, sort_keys=True))
            return 0

        raise SystemExit(f"Unsupported command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("devices",):
        subparsers.add_parser(command)

    for command in ("status", "programs", "history", "landscapes"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--device-id")

    on_parser = subparsers.add_parser("on")
    on_parser.add_argument("--device-id")
    on_parser.add_argument("--station", type=int, default=1)
    on_parser.add_argument("--seconds", type=float, required=True)
    on_parser.add_argument("--wait", action="store_true")
    on_parser.add_argument("--poll", action="store_true")
    on_parser.add_argument("--poll-interval", type=float, default=1.0)
    on_parser.add_argument("--timeout", type=float, default=20.0)

    off_parser = subparsers.add_parser("off")
    off_parser.add_argument("--device-id")
    off_parser.add_argument("--wait", action="store_true")
    off_parser.add_argument("--poll", action="store_true")
    off_parser.add_argument("--poll-interval", type=float, default=1.0)
    off_parser.add_argument("--timeout", type=float, default=20.0)

    cycle_parser = subparsers.add_parser("cycle")
    cycle_parser.add_argument("--device-id")
    cycle_parser.add_argument("--station", type=int, default=1)
    cycle_parser.add_argument("--seconds", type=float, required=True)
    cycle_parser.add_argument("--wait", action="store_true")
    cycle_parser.add_argument("--poll", action="store_true")
    cycle_parser.add_argument("--poll-interval", type=float, default=1.0)
    cycle_parser.add_argument("--timeout", type=float, default=20.0)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    raise SystemExit(main())
