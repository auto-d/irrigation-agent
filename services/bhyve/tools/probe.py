#!/usr/bin/env python3
"""Probe the cloned pybhyve client against the Orbit B-hyve API."""

import asyncio
import json
import logging
import os

from aiohttp import ClientSession

from services.bhyve.vendor.pybhyve import Client
from services.bhyve.vendor.pybhyve.const import API_HOST, DEVICES_PATH, LOGIN_PATH
from services.bhyve.vendor.pybhyve.errors import BHyveError


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


async def _noop_websocket_handler(data) -> None:
    logging.getLogger(__name__).debug("websocket event: %s", data)


def _mask_secret(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def _redact_text(value: str, secret: str) -> str:
    if not secret:
        return value
    return value.replace(secret, _mask_secret(secret))


async def _log_login_attempt(session: ClientSession, username: str, password: str) -> str:
    login_url = f"{API_HOST}{LOGIN_PATH}"
    payload = {"session": {"email": username, "password": password}}

    logging.info("BHYVE_EMAIL=%r", username)
    logging.info("BHYVE_PASSWORD=%s", _mask_secret(password))
    logging.info("POST %s", login_url)
    logging.info(
        "Request JSON: %s",
        json.dumps({"session": {"email": username, "password": _mask_secret(password)}}),
    )

    async with session.post(login_url, json=payload) as response:
        body = await response.text()
        logging.info("Login response status: %s", response.status)
        logging.info("Login response headers: %s", dict(response.headers))
        data = json.loads(body)
        token = data.get("orbit_session_token")
        logging.info("Login response body: %s", _redact_text(body, token))
        response.raise_for_status()
        return data["orbit_session_token"]


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
            client._token = await _log_login_attempt(session, username, password)
            devices = await client.devices
            logging.info("GET %s%s", API_HOST, DEVICES_PATH)
            print(json.dumps(devices, indent=2, sort_keys=True))
            return 0
        except BHyveError as err:
            logging.error("BHyve request failed: %s", err)
            return 1
        except Exception as err:
            logging.exception("Probe failed: %s", err)
            return 1
        finally:
            await client.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
