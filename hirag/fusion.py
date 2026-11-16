"""Rank fusion for Stage 1.

- rrf_code:  the authors' released rrf_fusion(): w1/(k+rank_bm25) + w2/(k+rank_dense), w2 = 1.0
- wrrf_paper: the paper's Eq. 5: alpha/(k+rank_bm25) + (1-alpha)/(k+rank_dense)

Both take two ranked lists of item ids (already truncated, e.g. top-20 each);
an item missing from a list gets 0 from that list.
"""
from __future__ import annotations

from typing import Dict, List, Sequence


def _fuse(list1: Sequence[int], list2: Sequence[int], w1: float, w2: float, k: int) -> List[int]:
    scores: Dict[int, float] = {}
    for rank, item in enumerate(list1, start=1):
        scores[item] = scores.get(item, 0.0) + w1 / (k + rank)
    for rank, item in enumerate(list2, start=1):
        scores[item] = scores.get(item, 0.0) + w2 / (k + rank)
    # sorted() is stable: ties keep first-insertion order, as in the authors' code
    return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def rrf_code(bm25_list, dense_list, w: float = 0.1, k: int = 60) -> List[int]:
    return _fuse(bm25_list, dense_list, w, 1.0, k)


def wrrf_paper(bm25_list, dense_list, alpha: float = 0.1, k: int = 60) -> List[int]:
    return _fuse(bm25_list, dense_list, alpha, 1.0 - alpha, k)


def fuse(method: str, bm25_list, dense_list, weight: float, k: int = 60) -> List[int]:
    if method == "code":
        return rrf_code(bm25_list, dense_list, weight, k)
    if method == "paper":
        return wrrf_paper(bm25_list, dense_list, weight, k)
    if method == "bm25":
        return list(bm25_list)
    if method == "dense":
        return list(dense_list)
    raise ValueError(f"unknown fusion {method!r}")
