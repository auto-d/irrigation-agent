"""LLM-backed planner and tool scaffolding."""

from __future__ import annotations

import datetime as dt
import inspect
import os
from copy import deepcopy
import json 
from typing import Any, Dict, List

from classical_agent import ClassicalPlanner
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
        "name": "perceive_weather",
        "description": "Fetch a normalized weather forecast perception event.",
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
        "name": "perceive_precipitation",
        "description": "Fetch a normalized precipitation-history perception event.",
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
        "name": "perceive_irrigation",
        "description": "Fetch a normalized irrigation-status perception event.",
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
        "name": "perceive_camera",
        "description": "Fetch a normalized camera scene-activity perception event.",
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
        "name": "act_irrigation",
        "description": "Actuate the irrigation actor.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["water_on", "water_off", "cycle"],
                },
                "duration_seconds": {"type": "integer"},
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
        result = await self.proxy.act("irrigation", action, **arguments)
        return self.build_result(
            instance.call_id,
            "ok",
            {
                "action_result": {
                    "timestamp": result.timestamp.isoformat(),
                    "actor": result.actor,
                    "action": result.action,
                    "status": result.status,
                    "detail": result.detail,
                }
            },
        )


class NotificationActionTool(Tool):
    TOOL_DESC = {
        "type": "function",
        "name": "act_notification",
        "description": "Invoke the notification actor.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["send"]},
                "message": {"type": "string"},
            },
            "required": ["action", "message"],
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
        result = await self.proxy.act("notification", action, **arguments)
        return self.build_result(
            instance.call_id,
            "ok",
            {
                "action_result": {
                    "timestamp": result.timestamp.isoformat(),
                    "actor": result.actor,
                    "action": result.action,
                    "status": result.status,
                    "detail": result.detail,
                }
            },
        )

class NeuralPlanner(LlmAgent, BasePlanner): 
    """LLM-backed planner with a classical fallback."""

    name = "neural"

    def __init__(self, api_key: str | None = None) -> None:
        BasePlanner.__init__(self)
        self._events: List[Event] = []
        self._fallback = ClassicalPlanner()
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        backend = Backend(self._api_key, model="gpt-5.4-mini") if self._api_key else None
        LlmAgent.__init__(
            self,
            backend=backend,
            dev_msgs=[
                (
                    "You are an irrigation planning agent. Use the provided tools to gather only the "
                    "information you need, then optionally invoke an action tool, and finally respond "
                    "with compact JSON containing keys action, duration_seconds, rationale."
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
            # No model configured, so run the same perception sequence and fall back.
            events = [
                await proxy.perceive("weather"),
                await proxy.perceive("precipitation"),
                await proxy.perceive("irrigation"),
                await proxy.perceive("camera"),
            ]
            self._events.extend(events)
            weather, precipitation, irrigation, camera = events
            decision = self._fallback.decision_from_events(
                now,
                weather=weather,
                precipitation=precipitation,
                irrigation=irrigation,
                camera=camera,
            )
            decision.planner = self.name
            decision.metadata["fallback"] = "no_api_key"
            action_count = 0
            if decision.action == "water_on":
                await proxy.act("irrigation", "water_on", duration_seconds=decision.duration_seconds or 300)
                action_count += 1
            elif decision.action == "water_off":
                await proxy.act("irrigation", "water_off")
                action_count += 1
            elif decision.action == "notify":
                await proxy.act("notification", "send", message=decision.rationale, metadata=decision.metadata)
                action_count += 1
            return PlannerRunResult(
                timestamp=now,
                planner=self.name,
                decision=decision,
                trace=self._consume_trace(),
                perception_count=len(events),
                action_count=action_count,
            )

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
            events = [
                await proxy.perceive("weather"),
                await proxy.perceive("precipitation"),
                await proxy.perceive("irrigation"),
                await proxy.perceive("camera"),
            ]
            self._events.extend(events)
            weather, precipitation, irrigation, camera = events
            decision = self._fallback.decision_from_events(
                now,
                weather=weather,
                precipitation=precipitation,
                irrigation=irrigation,
                camera=camera,
            )
            decision.planner = self.name
            decision.metadata["fallback"] = f"backend_error:{type(err).__name__}"
            action_count = 0
            if decision.action == "water_on":
                await proxy.act("irrigation", "water_on", duration_seconds=decision.duration_seconds or 300)
                action_count += 1
            elif decision.action == "water_off":
                await proxy.act("irrigation", "water_off")
                action_count += 1
            elif decision.action == "notify":
                await proxy.act("notification", "send", message=decision.rationale, metadata=decision.metadata)
                action_count += 1
            return PlannerRunResult(
                timestamp=now,
                planner=self.name,
                decision=decision,
                trace=self._consume_trace(),
                perception_count=len(events),
                action_count=action_count,
            )

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
            "You are a lawn irrigation planner. "
            "You may call perception tools and actor tools before you answer. "
            "Return only valid JSON with keys action, duration_seconds, rationale. "
            "Allowed actions: water_on, water_off, no_op, notify.\n\n"
            f"Decision time: {now.isoformat()}\n"
            f"Recent events: {json.dumps(recent_events, sort_keys=True)}"
        )
