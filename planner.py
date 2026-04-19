"""Minimal planner/runtime contracts for MVP experimentation."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Protocol


ActionName = Literal["water_on", "water_off", "no_op", "notify"]


@dataclass
class Event:
    """Canonical external observation unit exchanged with planners."""

    timestamp: datetime
    source: str
    type: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class Decision:
    """Canonical planner output at a decision boundary."""

    timestamp: datetime
    planner: str
    action: ActionName
    rationale: str
    duration_seconds: int | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Record of attempted downstream action emission."""

    timestamp: datetime
    action: ActionName
    status: Literal["dry_run", "executed", "failed"]
    detail: Dict[str, Any] = field(default_factory=dict)


class Planner(Protocol):
    """Minimal planner protocol used by the CLI runtime."""

    name: str

    def observe(self, event: Event) -> None:
        """Ingest a new event into planner-private state."""

    def decide(self, now: datetime) -> Decision:
        """Emit a decision at the current decision boundary."""


def event_to_dict(event: Event) -> Dict[str, Any]:
    payload = asdict(event)
    payload["timestamp"] = event.timestamp.isoformat()
    return payload


def decision_to_dict(decision: Decision) -> Dict[str, Any]:
    payload = asdict(decision)
    payload["timestamp"] = decision.timestamp.isoformat()
    return payload


def action_result_to_dict(result: ActionResult) -> Dict[str, Any]:
    payload = asdict(result)
    payload["timestamp"] = result.timestamp.isoformat()
    return payload


def event_from_dict(payload: Dict[str, Any]) -> Event:
    data = dict(payload)
    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
    return Event(**data)


def append_jsonl(path: str | Path, record_type: str, payload: Dict[str, Any]) -> None:
    """Append one structured record to a JSONL trajectory file."""
    record = {"record_type": record_type, **payload}
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
