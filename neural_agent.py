"""LLM-backed planner and tool scaffolding."""

from __future__ import annotations

import datetime as dt
import inspect
import os
from copy import deepcopy
import json 
from typing import Any, Dict, List

from planner import BasePlanner, Decision, Event, PlannerProxy, PlannerRunResult

class Backend(): 
    """Thin wrapper over the Responses API."""

    def __init__(self, api_key, model="gpt-5.4", reasoning={"effort": "low"}): 
        """Create a backend model abstraction."""
        try:
            from openai import OpenAI
        except ImportError as err:
            raise RuntimeError("The OpenAI Python package is required for neural planner execution.") from err

        self.api_key = api_key 
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.reasoning = reasoning

    def clone(self):
        """Create a deep copy of this instance."""
        new = Backend(self.api_key, self.model, self.reasoning)
        return new 

    def send(self, messages, instructions=None, tools=None): 
        """Send the current conversation to the backend."""
        response = self.client.responses.create(
            model=self.model,
            #instructions=None, 
            input=messages, 
            reasoning=self.reasoning,
            tools=tools
        )

        return response
    
    def unpack_response(self, response, include_reasoning=False): 
        """Return plain text chunks and any tool calls."""

        text = [] 
        tool_calls = []
        for r in response.output:
            if r.type == "reasoning":
                if include_reasoning:
                    text.append(r.summary)
            elif r.type == "message":
                for c in r.content:
                    if c.type == "output_text":
                        text.append(c.text)
                    elif c.type == "function_call":
                        tool_calls.append(c)
                    elif c.type == "tool_call":
                        raise NotImplementedError("Need to implement support for newer function calling semantics!")
                    else:
                        raise ValueError(f"Unexpected message type received: {c.type}")
            elif r.type == "function_call":
                tool_calls.append(r)
            elif r.type == "tool_call":
                raise NotImplementedError("Need to implement support for newer function calling semantics!")
            else:
                raise ValueError(f"Unexpected response type received: {r.type}")
                
        return text, tool_calls 

class Memory(): 
    """Conversation state for one planner run."""
    def __init__(self, backend=None): 
        self.backend = backend
        self.messages = []

    def clone(self):
        """Create a duplicate of this instance."""
        new = Memory()
        new.backend = self.backend.clone() 
        new.messages = deepcopy(self.messages)
        return new 
    
    def append_developer(self, messages=[]): 
        """Append developer messages."""        
        new = [ {"role": "developer", "content": msg} for msg in messages ]
        self.messages.extend(new)

    def append_user(self, messages=[]): 
        """Append user messages."""        
        new = [ {"role": "user", "content": msg} for msg in messages ]
        self.messages.extend(new)

    def append_assistant(self, messages=[]): 
        """Append assistant messages."""        
        new = [ {"role": "assistant", "content": msg} for msg in messages ]
        self.messages.extend(new)

    def append_response_output(self, responses=[]):
        """Append raw response objects in Responses API format."""
        for response in responses:
            if isinstance(response, dict):
                payload = deepcopy(response)
            elif hasattr(response, "model_dump"):
                payload = response.model_dump()
            else:
                payload = deepcopy(response.__dict__)

            # Responses history rejects the telemetry-only status field.
            if isinstance(payload, dict) and "status" in payload:
                del payload["status"]

            self.messages.append(payload)

    def append_tool_results(self, results=[]): 
        """Append tool call results in `function_call_output` form."""
        new = [
            {
                "type": "function_call_output",
                "call_id": result["call_id"],
                "output": json.dumps(result["result"]),
            }
            for result in results
        ]
        self.messages.extend(new)

    def summarize(self): 
        """Summarize memory."""
        if self.backend is None: 
            raise ValueError("Summarization requested but no backend configured!")
        
        raise NotImplementedError()
    
class Tool(): 
    """Base type for planner tools."""
    MINIMAL = {
        "type": "function",
        "name": None,
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }

    def __init__(self): 
        """Initialize the tool from its declared schema."""
        Tool.validate(Tool.MINIMAL, self.schema)

    def emit_telemetry(self, instance=None, event=None, text=None, data=None):
        """Placeholder telemetry hook."""
        return None

    def build_result(self, call_id, status, result):
        """Build a standard tool result payload."""
        return {"call_id": call_id, "status": status, "result": result}

    @classmethod 
    def validate(cls, src, target): 
        """Confirm the second object is a subset of the first."""
        if isinstance(src, dict):
            for k,v in src.items(): 
                if k not in target.keys(): 
                    raise ValueError(f"Provided tool schema is missing key '{k}'!")
                cls.validate(v, target[k])

    def validate_instance(self, instance): 
        """Parse JSON arguments from a Responses API tool call."""        
        
        args = json.loads(instance.arguments) 
        print(f"Found {instance.type} {instance.name} with id {instance.id} (call ID: {instance.call_id})...")
        print("Arguments", args)

        return args

    def run(self, instance, callback=None):
        """Execute the tool call."""
        raise NotImplementedError("run() method is an abstract method!")

class LlmAgent(): 
    """Generic async LLM agent loop with tool support."""

    def __init__(self, backend=None, cwd='.', persist=False, dev_msgs=[]): 
        """Create a new agent instance."""
        self.backend = backend 
        self.cwd = cwd
        self.persist = False
        self.memory = Memory(backend=backend)
        self.memory.append_developer(dev_msgs)

    def save(self, path): 
        """Persist agent configuration and state to disk."""
        pass 

    def load(self, path): 
        """Load an agent config off disk."""
        pass 

    async def chat(self, messages, tools=[], trace=False):
        """Send user messages to the backend and satisfy any returned tool calls."""

        self.memory.append_user(messages)

        response = None
        text = None 

        tool_schemas = [ x.schema for x in tools ]
        tool_map = { x.schema['name']: x for x in tools}

        while True: 
            if trace: 
                print("===============\n")
                print("Sending messages to backend...")
                print("Message history:\n",self.memory.messages)
                print("Tool descriptions:\n", tool_schemas)
                print("---------------\n")

            if self.backend is None:
                raise RuntimeError("chat() requested but no backend configured for this agent")

            response = self.backend.send(self.memory.messages, tools=tool_schemas)
            text, tool_calls = self.backend.unpack_response(response)
            self.memory.append_response_output(response.output)
            
            if len(tool_calls) == 0: 
                break 

            for call in tool_calls: 
                tool = tool_map[call.name]
                result = tool.run(call)
                # Tool implementations can be sync or async.
                if inspect.isawaitable(result):
                    result = await result
                self.memory.append_tool_results([result])

        return text

    def run(self): 
        """Run the agent loop."""
        raise NotImplementedError("Agent loop is currently external to this instance!")


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
