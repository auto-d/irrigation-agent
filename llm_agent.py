"""Shared LLM agent, memory, and tool scaffolding."""

from __future__ import annotations

import inspect
import json
import logging
from copy import deepcopy
from typing import Any, Dict, List

from llm_backend import Backend

_LOG = logging.getLogger(__name__)


class Memory:
    """Conversation state for one planner run."""

    def __init__(self, backend: Backend | None = None):
        self.backend = backend
        self.messages: List[Any] = []

    def clone(self) -> "Memory":
        """Create a duplicate of this instance."""
        new = Memory()
        new.backend = self.backend.clone() if self.backend is not None else None
        new.messages = deepcopy(self.messages)
        return new

    def append_developer(self, messages: List[str] | None = None) -> None:
        """Append developer messages."""
        for msg in messages or []:
            self.messages.append({"role": "developer", "content": msg})

    def append_user(self, messages: List[str] | None = None) -> None:
        """Append user messages."""
        for msg in messages or []:
            self.messages.append({"role": "user", "content": msg})

    def append_assistant(self, messages: List[str] | None = None) -> None:
        """Append assistant messages."""
        for msg in messages or []:
            self.messages.append({"role": "assistant", "content": msg})

    def append_response_output(self, responses: List[Any] | None = None) -> None:
        """Append raw response objects in Responses API format."""
        for response in responses or []:
            if isinstance(response, dict):
                payload = deepcopy(response)
            elif hasattr(response, "model_dump"):
                payload = response.model_dump()
            else:
                payload = deepcopy(response.__dict__)
            if isinstance(payload, dict) and "status" in payload:
                del payload["status"]
            self.messages.append(payload)

    def append_tool_results(self, results: List[Dict[str, Any]] | None = None) -> None:
        """Append tool call results in function-call-output form."""
        for result in results or []:
            self.messages.append(
                {
                    "type": "function_call_output",
                    "call_id": result["call_id"],
                    "output": json.dumps(result["result"]),
                }
            )


class Tool:
    """Base type for planner tools."""

    MINIMAL = {
        "type": "function",
        "name": None,
        "parameters": {
            "type": "object",
            "properties": {},
        },
    }

    def __init__(self) -> None:
        """Initialize the tool from its declared schema."""
        Tool.validate(Tool.MINIMAL, self.schema)

    def emit_telemetry(self, instance: Any = None, event: str | None = None, text: str | None = None, data: Dict[str, Any] | None = None) -> None:
        """Placeholder telemetry hook."""
        return None

    def build_result(self, call_id: str, status: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Build a standard tool result payload."""
        return {"call_id": call_id, "status": status, "result": result}

    @classmethod
    def validate(cls, src: Any, target: Any) -> None:
        """Confirm the second object is a subset of the first."""
        if isinstance(src, dict):
            for key, value in src.items():
                if key not in target.keys():
                    raise ValueError(f"Provided tool schema is missing key '{key}'!")
                cls.validate(value, target[key])

    def validate_instance(self, instance: Any) -> Dict[str, Any]:
        """Parse JSON arguments from a Responses API tool call."""
        args = json.loads(instance.arguments)
        _LOG.debug(
            "Tool call received: type=%s name=%s id=%s call_id=%s args=%s",
            instance.type,
            instance.name,
            instance.id,
            instance.call_id,
            args,
        )
        return args

    def run(self, instance: Any, callback: Any = None) -> Any:
        """Execute the tool call."""
        raise NotImplementedError("run() method is an abstract method!")


class LlmAgent:
    """Generic async LLM agent loop with tool support."""

    def __init__(self, backend: Backend | None = None, cwd: str = ".", persist: bool = False, dev_msgs: List[str] | None = None):
        """Create a new agent instance."""
        self.backend = backend
        self.cwd = cwd
        self.persist = persist
        self.memory = Memory(backend=backend)
        self.memory.append_developer(dev_msgs or [])

    async def chat(
        self,
        messages: List[str],
        tools: List[Tool] | None = None,
        trace: bool = False,
        text_format: Dict[str, Any] | None = None,
    ) -> List[str]:
        """Send user messages to the backend and satisfy any returned tool calls."""
        self.memory.append_user(messages)

        tool_schemas = [tool.schema for tool in (tools or [])]
        tool_map = {tool.schema["name"]: tool for tool in (tools or [])}

        while True:
            if trace:
                print("===============\n")
                print("Sending messages to backend...")
                print("Message history:\n", self.memory.messages)
                print("Tool descriptions:\n", tool_schemas)
                print("---------------\n")

            if self.backend is None:
                raise RuntimeError("chat() requested but no backend configured for this agent")

            response = self.backend.send(
                self.memory.messages,
                tools=tool_schemas,
                text_format=text_format,
            )
            text, tool_calls = self.backend.unpack_response(response)
            self.memory.append_response_output(response.output)

            if not tool_calls:
                return text

            for call in tool_calls:
                tool = tool_map[call.name]
                result = tool.run(call)
                if inspect.isawaitable(result):
                    result = await result
                self.memory.append_tool_results([result])
