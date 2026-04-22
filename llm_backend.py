"""Shared OpenAI backend wrapper for planner and perception code."""

from __future__ import annotations

import copy
from typing import Any, Dict, List


class Backend:
    """Thin wrapper over the OpenAI client surfaces we use."""

    def __init__(self, api_key: str, model: str = "gpt-5.4", reasoning: Dict[str, Any] | None = None):
        """Create a backend model abstraction."""
        try:
            from openai import OpenAI
        except ImportError as err:
            raise RuntimeError("The OpenAI Python package is required for neural execution.") from err

        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.reasoning = reasoning or {"effort": "low"}

    def clone(self) -> "Backend":
        """Create a deep copy of this instance."""
        return Backend(self.api_key, self.model, copy.deepcopy(self.reasoning))

    def send(
        self,
        messages: List[Dict[str, Any]],
        instructions: str | None = None,
        tools: List[Dict[str, Any]] | None = None,
        text_format: Dict[str, Any] | None = None,
    ) -> Any:
        """Send the current conversation to the Responses API."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": messages,
            "reasoning": self.reasoning,
            "tools": tools,
        }
        if text_format is not None:
            kwargs["text"] = {"format": text_format}
        return self.client.responses.create(
            **kwargs,
        )

    def unpack_response(self, response: Any, include_reasoning: bool = False) -> tuple[List[str], List[Any]]:
        """Return plain text chunks and any tool calls."""
        text: List[str] = []
        tool_calls: List[Any] = []
        for item in response.output:
            if item.type == "reasoning":
                if include_reasoning:
                    text.append(item.summary)
            elif item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        text.append(content.text)
                    elif content.type == "function_call":
                        tool_calls.append(content)
                    elif content.type == "tool_call":
                        raise NotImplementedError("New tool_call semantics are not implemented yet.")
                    else:
                        raise ValueError(f"Unexpected message type received: {content.type}")
            elif item.type == "function_call":
                tool_calls.append(item)
            elif item.type == "tool_call":
                raise NotImplementedError("New tool_call semantics are not implemented yet.")
            else:
                raise ValueError(f"Unexpected response type received: {item.type}")
        return text, tool_calls

    def structured_image_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_url: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a structured multimodal completion against one image."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            response_format={"type": "json_schema", "json_schema": schema},
        )
        import json

        content = response.choices[0].message.content or "{}"
        return json.loads(content)
