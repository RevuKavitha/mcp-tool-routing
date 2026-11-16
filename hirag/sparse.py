"""BM25 sparse retrieval (rank_bm25.BM25Okapi, as in the authors' code)."""
from __future__ import annotations

from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from .tokenize import get_tokenizer


class BM25Retriever:
    def __init__(self, docs: List[str], tokenizer: str = "jieba",
                 k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.tok = get_tokenizer(tokenizer)
        self.bm25 = BM25Okapi([self.tok(d) for d in docs], k1=k1, b=b, epsilon=epsilon)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(self.tok(query)), dtype=float)

    def rank(self, query: str, top_n: int | None = None, tie_break: str = "authors") -> List[int]:
        """Indices of documents, best first.

        tie_break="authors" reproduces np.argsort(scores)[::-1] from keyword_search.py
        (among equal scores, the later document comes first); "stable" keeps list order.
        """
        s = self.scores(query)
        order = np.argsort(s)[::-1] if tie_break == "authors" else np.argsort(-s, kind="stable")
        order = [int(i) for i in order]
        return order[:top_n] if top_n else order
