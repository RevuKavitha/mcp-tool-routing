"""Which services each method hands to the LLM (mirrors the authors' sigmcp.py / mulmcp.py).

- full:    all services (the paper's "Full Service" / full-context injection)
- flat:    Flat-RAG: dense top-20 tools -> first K distinct services (rag_test / rag_test_top3)
- colbert: ColBERT-RAG: MaxSim top-20 tools -> first K distinct services
- stage1:  Hi-RAG Stage 1 only (our ablation): S_cand order -> first K services
- hirag:   Hi-RAG Stage 1 + Stage 2 re-ranker (added in Phase 4)
"""
from __future__ import annotations

from typing import Callable, Dict, List

from .data import MCPBench
from .metrics import unique_in_order


def first_k_services(tool_ids: List[str], service_of: Dict[str, str], k: int) -> List[str]:
    return unique_in_order([service_of[t] for t in tool_ids])[:k]


def make_selector(method: str, bench: MCPBench, stage1=None, colbert=None,
                  reranker=None) -> Callable[[str, int], List[str]]:
    tool_ids = [t.tool_id for t in bench.tools]
    service_of = {t.tool_id: t.service for t in bench.tools}

    if method == "full":
        return lambda q, k: list(bench.services)
    if method == "flat":
        return lambda q, k: first_k_services([tool_ids[i] for i in stage1.dense_list(q)], service_of, k)
    if method == "colbert":
        return lambda q, k: first_k_services([tool_ids[i] for i in colbert.rank(q, 20)], service_of, k)
    if method == "stage1":
        return lambda q, k: list(stage1.run(q).candidates)[:k]
    if method == "hirag":
        return lambda q, k: reranker.rerank(q, list(stage1.run(q).candidates))[:k]
    raise ValueError(f"unknown method {method!r}")
