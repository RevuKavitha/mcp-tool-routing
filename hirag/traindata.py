"""Phase 3: Stage 2 training data (paper Sec. 3.4 / 4.1: LLM-generated queries per tool).

For every tool the LLM writes N realistic user requests that need exactly that tool.
Queries too similar to a MCPBench test query are removed (no test leakage), then the
rest is split 80/20 per tool into train / validation.
"""
from __future__ import annotations

import json
import random
import re
from typing import Dict, List

import numpy as np

from .data import MCPBench, Tool

GEN_SYSTEM = "You write realistic test data for a tool-retrieval system. Answer with JSON only."

GEN_TEMPLATE = """A user talks to an AI assistant that can call tools. Here is ONE tool:

Service: {service} (category: {type})
Tool: {tool}
What it does: {summary}
Parameters: {params}

Write {n} different user requests that can ONLY be solved by calling this tool.
Rules:
- English, written the way real users type (some short, some long, some polite, some direct).
- Include concrete values for the parameters (names, IDs, paths, cities, dates, ...).
- Do NOT mention the tool name, the service name or the word "tool".
- Vary the wording; do not start every request the same way.
Return a JSON list of {n} strings and nothing else."""


def format_params(tool: Tool) -> str:
    if not tool.parameters:
        return "none"
    return ", ".join(f"{p['name']} ({p.get('type', 'str')}{', required' if p.get('required') else ''})"
                     for p in tool.parameters)


def build_prompt(tool: Tool, n: int) -> str:
    return GEN_TEMPLATE.format(service=tool.service, type=tool.type.replace("-", " ").replace("_", " "),
                               tool=tool.path.strip("/"), summary=tool.summary,
                               params=format_params(tool), n=n)


def parse_queries(text: str) -> List[str]:
    """Pull a JSON list of strings out of an LLM answer (tolerates <think> blocks and ``` fences)."""
    if not text:
        return []
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in items if isinstance(s, str) and len(s.strip()) > 8]


def generate(bench: MCPBench, llm, n: int = 8, limit_tools: int | None = None, log_every: int = 20) -> List[Dict]:
    out: List[Dict] = []
    tools = bench.tools[:limit_tools]
    for i, t in enumerate(tools, 1):
        resp = llm._chat([{"role": "system", "content": GEN_SYSTEM},
                          {"role": "user", "content": build_prompt(t, n)}], [])
        qs = parse_queries(resp.get("content") or "")
        out.extend({"query": q, "tool_id": t.tool_id, "service": t.service, "type": t.type} for q in qs)
        if i % log_every == 0 or i == len(tools):
            print(f"[{i}/{len(tools)} tools] {len(out)} queries so far", flush=True)
    return out


def dedupe_and_split(items: List[Dict], test_texts: List[str], embedder=None, threshold: float = 0.95,
                     val_frac: float = 0.2, seed: int = 0) -> Dict[str, List[Dict]]:
    """Drop duplicates and near-copies of test queries; split per tool into train / val."""
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
    test_norm = {norm(t) for t in test_texts}
    seen, kept = set(), []
    for it in items:
        k = norm(it["query"])
        if k in seen or k in test_norm:
            continue
        seen.add(k)
        kept.append(it)
    removed_near = 0
    if embedder is not None and kept:
        q = embedder.encode([it["query"] for it in kept], is_query=True)
        t = embedder.encode(test_texts, is_query=True)
        too_close = (q @ t.T).max(axis=1) >= threshold
        removed_near = int(too_close.sum())
        kept = [it for it, bad in zip(kept, too_close) if not bad]
    rng = random.Random(seed)
    by_tool: Dict[str, List[Dict]] = {}
    for it in kept:
        by_tool.setdefault(it["tool_id"], []).append(it)
    train, val = [], []
    for tid in sorted(by_tool):
        lst = by_tool[tid]
        rng.shuffle(lst)
        n_val = max(1, round(len(lst) * val_frac)) if len(lst) >= 3 else 0
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    print(f"generated {len(items)}, kept {len(kept)} (near-copies of test removed: {removed_near}); "
          f"train {len(train)}, val {len(val)}, tools covered {len(by_tool)}")
    return {"train": train, "val": val}
