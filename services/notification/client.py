"""Minimal Discord webhook client for planner notifications."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from aiohttp import ClientSession
else:
    ClientSession = Any

DEFAULT_DISCORD_USERNAME = "Yard Irrigation Agent"
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class NotificationError(RuntimeError):
    """Structured outbound notification failure."""

    status: int
    message: str

    def __str__(self) -> str:
        return f"Notification delivery failed ({self.status}): {self.message}"


class DiscordWebhookClient:
    """Thin async client for sending messages to a Discord webhook."""

    def __init__(
        self,
        webhook_url: str,
        *,
        username: str = DEFAULT_DISCORD_USERNAME,
        session: Optional[ClientSession] = None,
    ) -> None:
        self._webhook_url = webhook_url
        self._username = username
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "DiscordWebhookClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @classmethod
    def from_env(cls, **kwargs: Any) -> "DiscordWebhookClient":
        """Build a client from environment variables."""
        webhook_url = kwargs.pop("webhook_url", None) or os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            raise RuntimeError("DISCORD_WEBHOOK_URL is required for notification delivery")
        if "username" not in kwargs:
            kwargs["username"] = os.getenv("DISCORD_WEBHOOK_USERNAME", DEFAULT_DISCORD_USERNAME)
        return cls(webhook_url, **kwargs)

    async def close(self) -> None:
        """Close an owned aiohttp session."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def probe(self, *, message: str = "Notification probe from irrigation-agent.") -> Dict[str, Any]:
        """Send a simple probe message and return the delivery result."""
        return await self.send(message)

    async def send(self, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Deliver a message and optional metadata to Discord."""
        session = await self._ensure_session()
        body = self._build_payload(message, metadata=metadata)
        _LOG.debug("Sending Discord webhook message")

        async with session.post(self._webhook_url, json=body) as response:
            text = await response.text()
            if response.status >= 400:
                _LOG.error("Discord webhook delivery failed with status %s", response.status)
                raise NotificationError(response.status, text.strip() or "unknown webhook error")
        _LOG.info("Discord webhook delivery succeeded")

        return {
            "delivered": True,
            "status": response.status,
            "message": message,
            "metadata": metadata or {},
        }

    async def _ensure_session(self) -> ClientSession:
        if self._session is None:
            try:
                from aiohttp import ClientSession as AiohttpClientSession
            except ModuleNotFoundError as err:
                raise RuntimeError(
                    "aiohttp is required to create a live Discord webhook session; "
                    "install aiohttp or inject an existing session"
                ) from err
            self._session = AiohttpClientSession()
        return self._session

    def _build_payload(self, message: str, *, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        content_parts = [message.strip() or "(empty notification)"]
        if metadata:
            compact = json.dumps(metadata, sort_keys=True)
            if len(compact) > 1500:
                compact = compact[:1497] + "..."
            content_parts.append(f"```json\n{compact}\n```")

        content = "\n".join(content_parts)
        if len(content) > 2000:
            content = content[:1997] + "..."

        return {
            "username": self._username,
            "content": content,
        }
