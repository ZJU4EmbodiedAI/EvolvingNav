"""Frozen GPT-5.6-Luna function-call controller for legal high-level actions."""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen


class LunaToolController:
    def __init__(self, *, requester=None) -> None:
        self.requester = requester or self._request
        self.handlers = {}

    def bind_tools(self, handlers: dict) -> None:
        self.handlers = handlers

    @staticmethod
    def _request(payload: dict) -> dict:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for the Luna controller")
        request = Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    def choose(self, legal_actions: list[str], public_context: dict) -> str:
        if not legal_actions:
            raise ValueError("controller needs at least one legal action")
        forbidden = ("private", "ground_truth", "true_state", "target_position_xyz",
                     "oracle_shortest_path")
        def contains_private(value):
            if isinstance(value, dict):
                return any(any(token in str(key).lower() for token in forbidden)
                           or contains_private(item) for key, item in value.items())
            return isinstance(value, list) and any(contains_private(item) for item in value)
        if contains_private(public_context):
            raise ValueError("evaluator-private field in controller context")
        context = {**public_context, "observation": dict(public_context.get("observation", {}))}
        image_url = context["observation"].pop("image_url", None)
        content = [{"type": "input_text", "text": json.dumps(context, ensure_ascii=False)}]
        if image_url:
            content.append({"type": "input_image", "image_url": image_url})
        payload = {
            "model": "gpt-5.6-luna",
            "input": [
                {"role": "system", "content": (
                    "You are a frozen embodied-search tool controller. Choose one legal high-level "
                    "action using only the supplied public belief, memory, costs and evidence. "
                    "Prefer the maximum utility and never infer evaluator-private truth."
                )},
                {"role": "user", "content": content},
            ],
            "tools": [{
                "type": "function", "name": "select_action",
                "description": "Choose one legal EvolvingNav high-level action.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"action": {"type": "string", "enum": legal_actions}},
                    "required": ["action"], "additionalProperties": False,
                },
            }],
            "tool_choice": {"type": "function", "name": "select_action"},
        }
        for name in self.handlers:
            parameters = ({"type": "object", "properties": {"filters": {"type": "object"}},
                           "required": ["filters"]} if name == "query_memory"
                          else {"type": "object", "properties": {}})
            payload["tools"].append({"type": "function", "name": name,
                                     "description": name.replace("_", " "),
                                     "parameters": parameters, "strict": False})
        for epoch in range(8):
            if self.handlers and epoch < 7:
                payload["tool_choice"] = "required"
            else:
                payload["tool_choice"] = {"type": "function", "name": "select_action"}
            response = self.requester(payload)
            calls = [item for item in response.get("output", []) if item.get("type") == "function_call"]
            if not calls:
                raise ValueError("Luna returned no tool call")
            payload["input"].extend(response["output"])
            for call in calls:
                arguments = json.loads(call["arguments"])
                if call["name"] == "select_action":
                    action = arguments["action"]
                    if action not in legal_actions:
                        raise ValueError("Luna selected an illegal action")
                    return action
                if call["name"] not in self.handlers:
                    raise ValueError("Luna selected an unknown tool")
                result = self.handlers[call["name"]](arguments.get("filters", arguments))
                payload["input"].append({"type": "function_call_output", "call_id": call["call_id"],
                                         "output": json.dumps(result, ensure_ascii=False)})
        raise RuntimeError("Luna tool-call budget exhausted")
