"""Stage 1 of Hi-RAG: Tool-as-Proxy hybrid retrieval -> candidate services (S_cand).

Defaults reproduce the authors' released code:
  jieba BM25 top-20 + dense top-20, fused with rrf_code(w=0.1), keep top M=10 tools,
  widen to 20 if those tools cover fewer than 3 services.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .data import MCPBench
from .dense import DenseRetriever, Embedder
from .fusion import fuse
from .sparse import BM25Retriever


@dataclass
class Stage1Config:
    tokenizer: str = "jieba"            # "jieba" (authors) | "clean" | "whitespace"
    k1: float = 1.5
    b: float = 0.75
    bm25_tie_break: str = "authors"     # "authors" | "stable"
    fusion: str = "code"                # "code" | "paper" | "bm25" | "dense"
    weight: float = 0.1                 # w (code) or alpha (paper)
    rrf_k: int = 60
    search_topk: int = 20               # SEARCH_TOPK in the authors' config.py
    top_m: int = 10                     # tools kept after fusion
    widen_to: int = 20                  # authors widen to 20 if < min_services
    min_services: int = 3


@dataclass
class Stage1Result:
    tools: List[str]                    # fused ranked tool_ids (full fused list)
    kept_tools: List[str]               # top-M (after widening rule)
    candidates: Dict[str, List[str]]    # service -> its retrieved tools, in rank order


class Stage1:
    def __init__(self, bench: MCPBench, embedder: Optional[Embedder], cfg: Stage1Config = Stage1Config()):
        self.bench, self.cfg = bench, cfg
        self.tool_ids = [t.tool_id for t in bench.tools]
        self.service_of = {t.tool_id: t.service for t in bench.tools}
        docs = [t.summary for t in bench.tools]          # the authors index the one-line summary
        self.bm25 = BM25Retriever(docs, cfg.tokenizer, cfg.k1, cfg.b)
        self.dense = DenseRetriever(docs, embedder) if embedder is not None else None
        self._dense_cache: Dict[str, np.ndarray] = {}

    def precompute_dense(self, queries: List[str]) -> None:
        """Encode many queries in one batch (fast on GPU)."""
        if self.dense is None:
            return
        todo = [q for q in queries if q not in self._dense_cache]
        if todo:
            for q, s in zip(todo, self.dense.scores_many(todo)):
                self._dense_cache[q] = s

    def bm25_list(self, query: str) -> List[int]:
        return self.bm25.rank(query, self.cfg.search_topk, self.cfg.bm25_tie_break)

    def dense_list(self, query: str) -> List[int]:
        if self.dense is None:
            return []
        if query not in self._dense_cache:
            self.precompute_dense([query])
        return DenseRetriever.rank_from_scores(self._dense_cache[query], self.cfg.search_topk)

    def run(self, query: str) -> Stage1Result:
        c = self.cfg
        fused = fuse(c.fusion, self.bm25_list(query), self.dense_list(query), c.weight, c.rrf_k)
        tools = [self.tool_ids[i] for i in fused]
        kept = tools[:c.top_m]
        if len({self.service_of[t] for t in kept}) < c.min_services:
            kept = tools[:c.widen_to]
        cands: Dict[str, List[str]] = {}
        for t in kept:
            cands.setdefault(self.service_of[t], []).append(t)
        return Stage1Result(tools, kept, cands)
