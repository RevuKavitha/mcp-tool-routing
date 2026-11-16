"""Evaluation metrics."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1 if any gold item is in the top k."""
    return float(any(g in ranked[:k] for g in gold))


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Share of gold items found in the top k."""
    return sum(g in ranked[:k] for g in gold) / len(gold)


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Binary-relevance NDCG@k."""
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def unique_in_order(items: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(items))


def mean(xs: Sequence[float]) -> float:
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def format_table(rows: List[Dict], cols: List[str]) -> str:
    widths = {c: max(len(c), *(len(f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols}
    fmt = lambda r, c: (f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c])).rjust(widths[c]) if c != cols[0] else str(r[c]).ljust(widths[c])
    head = " | ".join(c.ljust(widths[c]) if c == cols[0] else c.rjust(widths[c]) for c in cols)
    return "\n".join([head, "-" * len(head)] + [" | ".join(fmt(r, c) for c in cols) for r in rows])
