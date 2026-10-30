"""Small real Python tools with allowlisted dispatch and bounded inputs."""
from __future__ import annotations

import json
import re
from typing import Any

DOCUMENTS = {
    "doc-plc": "A programmable logic controller (PLC) reads inputs, executes programmed control logic, and controls equipment outputs.",
    "doc-hmi": "A human machine interface (HMI) displays industrial process status, alarms and operator controls.",
    "doc-scada": "SCADA collects industrial equipment data to support remote supervision and monitoring.",
    "doc-safety": "Machine safety uses risk assessment, interlocks and tested safety functions to reduce hazards.",
}
STOP = {"about", "with", "from", "this", "that", "into", "slide", "slides", "training", "basics", "the", "and"}


def tokens(text: str) -> set[str]:
    return {x for x in re.findall(r"[a-z]{3,}", text.lower()) if x not in STOP}


def search_documents(query: str, top_k: int = 2) -> list[dict[str, str]]:
    if not isinstance(query, str) or not query.strip() or len(query) > 200:
        raise ValueError("query must be non-empty text up to 200 characters")
    if type(top_k) is not int or not 1 <= top_k <= 4:
        raise ValueError("top_k must be an integer from 1 to 4")
    q = tokens(query)
    ranked = [(len(q & tokens(v)), k, v) for k, v in DOCUMENTS.items()]
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [{"source_id": key, "text": text} for score, key, text in ranked if score > 0][:top_k]


def estimate_cost(slides: int, per_slide: float) -> dict[str, Any]:
    if type(slides) is not int or not 1 <= slides <= 100:
        raise ValueError("slides must be an integer from 1 to 100")
    if type(per_slide) not in (float, int) or not 0 <= per_slide <= 100:
        raise ValueError("per_slide must be a nonnegative number at most 100")
    return {"slides": slides, "cost_usd": round(slides * per_slide, 2)}


TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "search_documents",
      "description": "Look up supporting evidence in a small local industrial automation corpus. This is NOT a web search.",
      "parameters": {"type": "object", "properties": {
          "query": {"type": "string", "description": "Search keywords, e.g. PLC"},
          "top_k": {"type": "integer", "description": "Number of excerpts, 1 to 4"}
      }, "required": ["query"]}}},
    {"type": "function", "function": {"name": "estimate_cost",
      "description": "Estimate slide cost using a given number of slides and USD cost per slide.",
      "parameters": {"type": "object", "properties": {
          "slides": {"type": "integer"}, "per_slide": {"type": "number"}
      }, "required": ["slides", "per_slide"]}}},
]


def execute_tool(name: str, raw_arguments: str) -> Any:
    """Never eval arbitrary model text or call arbitrary Python attributes."""
    try:
        args = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("Tool arguments must be valid JSON") from exc
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be a JSON object")
    if name == "search_documents":
        if set(args) - {"query", "top_k"}:
            raise ValueError("Unexpected search argument")
        return search_documents(**args)
    if name == "estimate_cost":
        if set(args) != {"slides", "per_slide"}:
            raise ValueError("Expected slides and per_slide arguments")
        return estimate_cost(**args)
    raise ValueError(f"Tool '{name}' is not allowed")
