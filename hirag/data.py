"""Load MCPBench (tools, services, types, queries) from the Hi-RAG authors' repository.

The authors' repository has no licence file, so we do not copy their data.
`ensure_repo()` clones it from GitHub the first time it is needed.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List

REPO_URL = "https://github.com/TW-NLP/Hierarchical-RAG-MCP"
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Hierarchical-RAG-MCP")


def ensure_repo(root: str = DEFAULT_ROOT) -> str:
    """Clone the authors' repository if it is not already present. Returns its path."""
    if not os.path.isdir(os.path.join(root, "data", "query_test")):
        os.makedirs(os.path.dirname(root), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, root], check=True)
    return root


def make_tool_id(service: str, path: str) -> str:
    """Tool name as used in the gold labels, e.g. ('docx_mcp', '/read-docx') -> 'docx_mcp_read_docx'."""
    return (service + "_" + path.strip("/").replace("-", "_").replace("/", "_")).lower()


@dataclass
class Tool:
    tool_id: str          # e.g. docx_mcp_read_docx
    service: str          # e.g. docx_mcp
    type: str             # e.g. file_systems
    summary: str          # one-line description (the text the authors index)
    path: str
    method: str
    parameters: list = field(default_factory=list)


@dataclass
class Service:
    name: str
    title: str            # e.g. "docx_mcp service" (what the authors embed for the service node)
    type: str
    tools: List[str] = field(default_factory=list)   # tool_ids


@dataclass
class Query:
    qid: str
    text: str
    gold_tools: List[str]       # tool_ids
    gold_services: List[str]
    kind: str                   # "single" or "multi"


@dataclass
class MCPBench:
    tools: List[Tool]
    services: Dict[str, Service]
    single: List[Query]
    multi: List[Query]

    @property
    def tool_by_id(self) -> Dict[str, Tool]:
        return {t.tool_id: t for t in self.tools}

    @property
    def types(self) -> List[str]:
        return sorted({s.type for s in self.services.values()})

    def summary(self) -> str:
        return (f"MCPBench: {len(self.tools)} tools, {len(self.services)} services with tools, "
                f"{len(self.types)} types, {len(self.single)} single + {len(self.multi)} multi queries")


def load_mcpbench(root: str | None = None) -> MCPBench:
    root = ensure_repo(root or DEFAULT_ROOT)
    sig = json.load(open(os.path.join(root, "data/query_test/sig_mcp_test.json"), encoding="utf-8"))
    mul = json.load(open(os.path.join(root, "data/query_test/mul_mcp_test.json"), encoding="utf-8"))
    s2o = json.load(open(os.path.join(root, "data/service_info/summary2other.json"), encoding="utf-8"))

    tools: List[Tool] = []
    services: Dict[str, Service] = {}
    single: List[Query] = []
    for svc in sig:
        for ep in svc["endpoints"]:
            info = s2o[ep["summary"]]
            name = info["service_name"]
            tid = make_tool_id(name, info["path"])
            tools.append(Tool(tid, name, info["type"], ep["summary"], info["path"],
                              info["method"], ep.get("parameters", [])))
            if name not in services:
                services[name] = Service(name, info.get("title", name), info["type"])
            services[name].tools.append(tid)
            single.append(Query(f"S{len(single):03d}", ep["query"], [tid], [name], "single"))

    by_id = {t.tool_id: t for t in tools}
    multi: List[Query] = []
    for i, m in enumerate(mul):
        gold = [t.lower() for t in m["tool_list"]]
        missing = [g for g in gold if g not in by_id]
        if missing:
            raise ValueError(f"multi query {i}: unknown gold tools {missing}")
        svcs = list(dict.fromkeys(by_id[g].service for g in gold))
        multi.append(Query(f"M{i:03d}", m["query"], gold, svcs, "multi"))

    return MCPBench(tools, services, single, multi)


if __name__ == "__main__":
    print(load_mcpbench().summary())
