"""ToolBench (ToolLLM) as a second, large-scale benchmark for the router (paper Table 6 setting).

Tool catalogue: the Hi-RAG authors' data/tool_bench/tool_bench_summary.json (10,657 entries ->
10,388 unique RapidAPI tools used as "services", 50 categories used as "types").
Test queries: StableToolBench solvable test sets G1/G2/G3 instruction (163 / 106 / 61 queries),
because the original ToolBench download links no longer work (same choice as the reproduction).
Gold = the tools named in each query's "relevant APIs" (binary relevance, NDCG@k as in the authors' script).

The result is an MCPBench-shaped object, so the registry / router / Stage 1 code runs unchanged.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Dict, List

from .data import DEFAULT_ROOT, MCPBench, Query, Service, Tool, ensure_repo

STB_URL = "https://github.com/THUNLP-MT/StableToolBench.git"
STB_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stb", "repo")
SETS = ("G1_instruction", "G2_instruction", "G3_instruction")


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def ensure_stb(root: str = STB_ROOT) -> str:
    """Sparse-clone only StableToolBench's solvable_queries folder (~6 MB)."""
    qdir = os.path.join(root, "solvable_queries", "test_instruction")
    if not os.path.isdir(qdir):
        os.makedirs(os.path.dirname(root), exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", STB_URL, root], check=True)
        subprocess.run(["git", "-C", root, "sparse-checkout", "set", "solvable_queries"], check=True)
    return root


def load_toolbench(sets=SETS, max_desc_chars: int = 400, hirag_root: str | None = None,
                   stb_root: str | None = None) -> MCPBench:
    root = ensure_repo(hirag_root or DEFAULT_ROOT)
    entries = json.load(open(os.path.join(root, "data/tool_bench/tool_bench_summary.json"), encoding="utf-8"))
    merged: Dict[str, Dict] = {}
    for e in entries:                                   # 184 names occur more than once: merge them
        key = norm_name(e["service"])
        if not key:
            continue
        m = merged.setdefault(key, {"name": e["service"], "type": e.get("type") or "Other", "desc": []})
        d = (e.get("tool") or "").replace("\r", " ").replace("\n", " ").strip()
        if d and d not in m["desc"]:
            m["desc"].append(d)
    tools: List[Tool] = []
    services: Dict[str, Service] = {}
    for key, m in merged.items():
        summary = " ".join(m["desc"])[:max_desc_chars] or m["name"]
        tools.append(Tool(key, key, m["type"], summary, "/" + m["name"], "GET", []))
        services[key] = Service(key, m["name"], m["type"], [key])

    qroot = os.path.join(ensure_stb(stb_root or STB_ROOT), "solvable_queries", "test_instruction")
    queries: List[Query] = []
    for s in sets:
        for x in json.load(open(os.path.join(qroot, s + ".json"), encoding="utf-8")):
            gold = list(dict.fromkeys(norm_name(t) for t, _ in x["relevant APIs"]))
            gold = [g for g in gold if g in services]
            if gold:
                queries.append(Query(f"{s.split('_')[0]}-{x['query_id']}", x["query"], gold, gold, s.split("_")[0]))
    # .single holds every query (kind = G1/G2/G3); .multi is unused
    return MCPBench(tools, services, queries, [])
