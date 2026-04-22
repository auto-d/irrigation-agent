"""LLM-backed planner and tool scaffolding."""

from __future__ import annotations

import datetime as dt
import os
import json
from typing import Any, Dict, List

from llm_agent import LlmAgent, Tool
from llm_backend import Backend
from planner import BasePlanner, Decision, Event, PlannerProxy, PlannerRunResult


class WeatherPerceptionTool(Tool):
    TOOL_DESC = {
        "type": "function",
        "name": "inspect_weather_forecast",
        "description": "Inspect the weather forecast for the next few hours or days. Use this to decide whether rain is expected soon enough that watering would be wasteful.",
        "parameters": {
            "type": "object",
            "properties": {
                "forecast_hours": {"type": "integer"},
                "forecast_days": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    }

    def __init__(self, proxy: PlannerProxy, event_sink: List[Event]):
        self.schema = self.TOOL_DESC
        self.proxy = proxy
        self.event_sink = event_sink
        super().__init__()

    async def run(self, instance, callback=None):
        arguments = self.validate_instance(instance)
        event = await self.proxy.perceive("weather", **arguments)
        self.event_sink.append(event)
        return self.build_result(instance.call_id, "ok", {"event": self._event_to_result(event)})

    @staticmethod
    def _event_to_result(event: Event) -> Dict[str, Any]:
        return {
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
            "type": event.type,
            "payload": event.payload,
            "event_id": event.event_id,
        }


class PrecipitationPerceptionTool(Tool):
    TOOL_DESC = {
        "type": "function",
        "name": "inspect_recent_rainfall",
        "description": "Inspect recent rainfall totals. Use this to decide whether the ground is likely already wet enough that watering should be skipped.",
        "parameters": {
            "type": "object",
            "properties": {
                "window": {"type": "string"},
            },
            "additionalProperties": False,
        },
    }

    def __init__(self, proxy: PlannerProxy, event_sink: List[Event]):
        self.schema = self.TOOL_DESC
        self.proxy = proxy
        self.event_sink = event_sink
        super().__init__()

    async def run(self, instance, callback=None):
        arguments = self.validate_instance(instance)
        event = await self.proxy.perceive("precipitation", **arguments)
        self.event_sink.append(event)
        return self.build_result(instance.call_id, "ok", {"event": WeatherPerceptionTool._event_to_result(event)})


class IrrigationPerceptionTool(Tool):
    TOOL_DESC = {
        "type": "function",
        "name": "inspect_irrigation_state",
        "description": "Inspect whether the irrigation device is connected, currently watering, expected to be watering, or was watered recently.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }

    def __init__(self, proxy: PlannerProxy, event_sink: List[Event]):
        self.schema = self.TOOL_DESC
        self.proxy = proxy
        self.event_sink = event_sink
        super().__init__()

    async def run(self, instance, callback=None):
        self.validate_instance(instance)
        event = await self.proxy.perceive("irrigation")
        self.event_sink.append(event)
        return self.build_result(instance.call_id, "ok", {"event": WeatherPerceptionTool._event_to_result(event)})


class CameraPerceptionTool(Tool):
    TOOL_DESC = {
        "type": "function",
        "name": "inspect_lawn_camera",
        "description": "Inspect the lawn camera. Use this when you need to know whether a person, animal, mower, or other obstacle may be present on the lawn.",
        "parameters": {
            "type": "object",
            "properties": {
                "sample_frames": {"type": "integer"},
                "save_frame": {"type": "string"},
            },
            "additionalProperties": False,
        },
    }

    def __init__(self, proxy: PlannerProxy, event_sink: List[Event]):
        self.schema = self.TOOL_DESC
        self.proxy = proxy
        self.event_sink = event_sink
        super().__init__()

    async def run(self, instance, callback=None):
        arguments = self.validate_instance(instance)
        event = await self.proxy.perceive("camera", **arguments)
        self.event_sink.append(event)
        return self.build_result(instance.call_id, "ok", {"event": WeatherPerceptionTool._event_to_result(event)})


class IrrigationActionTool(Tool):
    TOOL_DESC = {
        "type": "function",
        "name": "set_watering",
        "description": "Control lawn watering. Start watering, stop watering, or run a short cycle for testing. Use this only when you have enough evidence to justify changing the valve state.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start_watering", "stop_watering", "cycle_watering"],
                },
                "duration_seconds": {
                    "type": "integer",
                    "description": "How long to water for when starting or cycling watering.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    }

    def __init__(self, proxy: PlannerProxy):
        self.schema = self.TOOL_DESC
        self.proxy = proxy
        super().__init__()

    async def run(self, instance, callback=None):
        arguments = self.validate_instance(instance)
        action = arguments.pop("action")
        mapped_action = {
            "start_watering": "water_on",
            "stop_watering": "water_off",
            "cycle_watering": "cycle",
        }[action]
        result = await self.proxy.act("irrigation", mapped_action, **arguments)
        return self.build_result(
            instance.call_id,
            "ok",
            {
                "watering_result": {
                    "timestamp": result.timestamp.isoformat(),
                    "action": action,
                    "status": result.status,
                    "detail": result.detail,
                }
            },
        )


class NotificationActionTool(Tool):
    TOOL_DESC = {
        "type": "function",
        "name": "send_human_message",
        "description": "Send a message to a human via Discord. Use this to flag error conditions, warnings, or ambiguous situations that should be surfaced rather than handled silently.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "description": "Optional compact machine-readable context to include with the message.",
                },
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    }

    def __init__(self, proxy: PlannerProxy):
        self.schema = self.TOOL_DESC
        self.proxy = proxy
        super().__init__()

    async def run(self, instance, callback=None):
        arguments = self.validate_instance(instance)
        result = await self.proxy.act("notification", "send", **arguments)
        return self.build_result(
            instance.call_id,
            "ok",
            {
                "message_result": {
                    "timestamp": result.timestamp.isoformat(),
                    "delivery_channel": "discord",
                    "status": result.status,
                    "detail": result.detail,
                }
            },
        )

class NeuralPlanner(LlmAgent, BasePlanner): 
    """LLM-backed planner."""

    name = "neural"

    def __init__(self, api_key: str | None = None) -> None:
        BasePlanner.__init__(self)
        self._events: List[Event] = []
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        backend = Backend(self._api_key, model="gpt-5.4-mini") if self._api_key else None
        LlmAgent.__init__(
            self,
            backend=backend,
            dev_msgs=[
                (
                    "You are an irrigation planning agent responsible for a residential lawn. "
                    "Your job is to decide whether to water now, stop watering, do nothing, or message a human "
                    "when the situation is ambiguous or concerning. "
                    "Use the available tools to inspect only the information you need. "
                    "Do not refer to internal framework concepts such as actors, proxies, or planner internals. "
                    "Think in domain terms: recent watering, recent rain, forecast rain, obstacles on the lawn, "
                    "daylight versus nighttime, "
                    "and whether a human should be notified. "
                    "Never begin watering outside the allowed daylight watering window. "
                    "If you use a messaging tool, reserve it for warnings, failures, or ambiguity worth surfacing to a human. "
                    "After you are done gathering information and taking any necessary tool actions, respond with compact JSON "
                    "containing keys action, duration_seconds, rationale."
                )
            ],
        )

    async def run(self, proxy: PlannerProxy, *, now: dt.datetime) -> PlannerRunResult:
        tools = [
            WeatherPerceptionTool(proxy, self._events),
            PrecipitationPerceptionTool(proxy, self._events),
            IrrigationPerceptionTool(proxy, self._events),
            CameraPerceptionTool(proxy, self._events),
            IrrigationActionTool(proxy),
            NotificationActionTool(proxy),
        ]

        if self.backend is None:
            raise RuntimeError("Neural planner requested without an OpenAI backend. Set OPENAI_API_KEY.")

        prompt = self._decision_prompt(now)
        try:
            # In the live LLM path, tool calls are the only place perception and action happen.
            text_chunks = await self.chat(
                [prompt],
                tools=tools,
            )
            content = "\n".join(text_chunks).strip()
            parsed = json.loads(content)
            decision = Decision(
                timestamp=now,
                planner=self.name,
                action=parsed.get("action", "no_op"),
                duration_seconds=parsed.get("duration_seconds"),
                rationale=parsed.get("rationale", "No rationale provided."),
                metadata={"raw_response": content},
            )
            return PlannerRunResult(
                timestamp=now,
                planner=self.name,
                decision=decision,
                trace=self._consume_trace(),
                perception_count=getattr(proxy, "perception_count", 0),
                action_count=getattr(proxy, "action_count", 0),
            )
        except Exception as err:
            raise RuntimeError(f"Neural planner execution failed: {type(err).__name__}: {err}") from err

    def _decision_prompt(self, now: dt.datetime) -> str:
        recent_events = [
            {
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "type": event.type,
                "payload": event.payload,
            }
            for event in self._events[-12:]
        ]
        return (
            "You are deciding what to do about watering a residential lawn right now. "
            "You may inspect the forecast, recent rainfall, irrigation state, and lawn camera before answering. "
            "You may also start or stop watering, run a short watering cycle, or send a human a Discord message if the situation warrants it. "
            "Watering is only allowed during the local daylight watering window. "
            "Return only valid JSON with keys action, duration_seconds, rationale. "
            "Allowed final action values: water_on, water_off, no_op, notify. "
            "Use water_on when the decision is to water the lawn, water_off when the decision is to stop watering, "
            "no_op when the right move is to leave things alone, and notify when the main outcome is to surface the situation to a human.\n\n"
            f"Decision time: {now.isoformat()}\n"
            f"Wall-time guard: {json.dumps(self._watering_window_summary(now), sort_keys=True)}\n"
            f"Recent events: {json.dumps(recent_events, sort_keys=True)}"
        )
