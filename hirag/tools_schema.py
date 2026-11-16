"""Turn MCPBench tools into OpenAI-style function schemas for LLM tool calling.

The authors expose each service as an MCP server and Qwen-Agent lists all of that
service's tools to the LLM. We pass the same tools directly as function schemas,
which needs no running servers (the paper only scores which tool is called).
"""
from __future__ import annotations

import re
from typing import Dict, List

from .data import MCPBench, Tool

_BASIC = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
          "list": "array", "dict": "object"}


def _json_type(py_type: str) -> Dict:
    t = (py_type or "").strip()
    m = re.fullmatch(r"Optional\[(.*)\]", t)
    if m:
        t = m.group(1)
    head = re.split(r"[\[\]]", t)[0].lower()
    if head in ("list", "sequence", "tuple", "set"):
        return {"type": "array", "items": {}}
    if head in ("dict", "mapping"):
        return {"type": "object"}
    if head in _BASIC:
        return {"type": _BASIC[head]}
    return {}                                    # Any / unknown: no type constraint


def tool_to_function(tool: Tool) -> Dict:
    props, required = {}, []
    for p in tool.parameters or []:
        name = p.get("name")
        if not name:
            continue
        spec = _json_type(p.get("type", ""))
        if p.get("description"):
            spec["description"] = p["description"]
        props[name] = spec
        if p.get("required"):
            required.append(name)
    return {"type": "function", "function": {
        "name": tool.tool_id[:64],
        "description": tool.summary,
        "parameters": {"type": "object", "properties": props, "required": required}}}


def functions_for_services(bench: MCPBench, services: List[str]) -> List[Dict]:
    """All tools of the given services, in service order (what an MCP server would expose)."""
    by_id = bench.tool_by_id
    out = []
    for s in services:
        out.extend(tool_to_function(by_id[t]) for t in bench.services[s].tools)
    return out
