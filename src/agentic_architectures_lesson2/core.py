"""LLM adapter and offline demonstration model for Lesson 2.

The Ollama adapter uses the OpenAI-compatible chat completions API. Ollama
supports 'tools', but its OpenAI-compatible endpoint does not guarantee
support for the 'tool_choice' argument; do not send it.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Reply:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


class Model(Protocol):
    def respond(self, role: str, messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None) -> Reply: ...


class OllamaModel:
    """Real local model. No paid API key required."""

    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434/v1",
                 timeout: float = 180.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=timeout,
                             max_retries=1)
        self.model = model

    def respond(self, role: str, messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None) -> Reply:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            request["tools"] = tools
        result = self.client.chat.completions.create(**request)
        msg = result.choices[0].message
        calls = tuple(
            ToolCall(id=c.id or f"tool-{index}", name=c.function.name,
                     arguments=c.function.arguments)
            for index, c in enumerate(msg.tool_calls or [])
        )
        # Thinking output is not a final answer. Many Ollama models return it
        # separately; strip visible <think> markup when present in content.
        content = re.sub(r"<think>.*?</think>\s*", "", msg.content or "", flags=re.S).strip()
        return Reply(content=content, tool_calls=calls)


def parse_json_object(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return obj


class DemoModel:
    """Deterministic simulation so demos/tests work with zero downloads.

    Does NOT simulate the quality, planning skill, or tool accuracy of a real LLM.
    """
    def respond(self, role: str, messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None) -> Reply:
        latest = next((m["content"] for m in reversed(messages)
                       if m.get("role") == "user"), "")
        if role == "single":
            used_tools = [m.get("name") for m in messages if m.get("role") == "tool"]
            if "search_documents" not in used_tools:
                return Reply(tool_calls=(ToolCall("search-call", "search_documents", '{"query":"PLC"}'),))
            if "estimate_cost" not in used_tools:
                return Reply(tool_calls=(ToolCall("cost-call", "estimate_cost", '{"slides":3,"per_slide":0.05}'),))
            return Reply("PLCs read inputs and control outputs [doc-plc]. Three slides cost $0.15 in the toy estimate.")

        if role == "supervisor":
            data = json.loads(latest)
            completed = [x["worker"] for x in data.get("worker_results", [])]
            if "researcher" not in completed:
                return Reply(json.dumps({"delegate": "researcher", "instruction": "Research PLC basics"}))
            if "writer" not in completed:
                return Reply(json.dumps({"delegate": "writer", "instruction": "Explain PLC basics with citations"}))
            if "cost" not in completed:
                return Reply(json.dumps({"delegate": "cost", "instruction": "Cost 3 slides at 0.05 USD per slide"}))
            return Reply(json.dumps({"delegate": "finish", "final_answer": "PLC training content researched, drafted and estimated at $0.15."}))

        if role == "supervisor_researcher":
            return Reply("PLC")  # a suggested lookup query (actual retrieval is code-controlled)
        if role == "supervisor_writer":
            return Reply("A PLC reads sensor inputs, executes control logic, and controls equipment outputs [doc-plc].")

        if role == "handoff:triage":
            return Reply('{"action":"handoff","target":"researcher","payload":"Look up PLC safety evidence"}')
        if role == "handoff:researcher":
            return Reply('{"action":"handoff","target":"writer","payload":"Write a short grounded PLC safety slide"}')
        if role == "handoff:writer":
            return Reply('{"action":"finish","answer":"PLC safety: assess hazards and use tested interlocks [doc-safety]."}')

        if role == "parallel_worker":
            data = json.loads(latest)
            return Reply(f"{data['topic']}: concise training takeaway based on {', '.join(data['source_ids'])}.")

        if role == "critic_writer":
            data = json.loads(latest)
            if data["attempt"] == 1:
                return Reply("PLCs coordinate manufacturing equipment.")  # missing citation: deterministic failure
            return Reply("PLCs monitor input signals and control outputs using programmed logic [doc-plc].")
        if role == "critic_reviewer":
            return Reply('{"approved":true,"feedback":"Clear, relevant, and supported by the supplied evidence."}')
        raise ValueError(f"Unknown mock role: {role}")


def create_model(mode: str, model_name: str | None = None) -> Model:
    if mode == "mock":
        return DemoModel()
    if mode == "ollama":
        return OllamaModel(
            model=model_name or os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    raise ValueError("mode must be mock or ollama")
