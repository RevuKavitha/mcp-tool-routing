"""Calibration of the confidence gate and statistics for the router evaluation.

risk_control_tau: smallest threshold tau whose deterministic path has a conservative error estimate
    (errors + 1) / (covered + 1) <= eps on the calibration set (finite-sample risk control in the
    spirit of conformal risk control). Smaller eps -> stricter gate -> fewer queries skip the LLM.
cross_fit: K-fold cross-fitting so every test query is gated by a tau chosen WITHOUT that query.
mcnemar_exact: exact two-sided McNemar test on paired correct/incorrect outcomes.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple


def risk_control_tau(conf: Sequence[float], correct: Sequence[bool], eps: float) -> float:
    pairs = sorted(zip(conf, correct), key=lambda x: -x[0])
    best = math.inf                                   # inf = never deterministic
    err = cov = 0
    for i, (c, ok) in enumerate(pairs):
        cov += 1
        err += (not ok)
        # only cut between distinct confidence values
        if i + 1 < len(pairs) and pairs[i + 1][0] == c:
            continue
        if c > 0 and (err + 1) / (cov + 1) <= eps:
            best = c                                  # keep lowering tau while the risk bound holds
    return best


def tau_by_shape(cal: List[Dict], eps: float) -> Dict[int, float]:
    """One tau per request shape (1 = single sub-task, 2 = several), each with the same risk target."""
    out = {}
    for sh in (1, 2):
        grp = [c for c in cal if c["shape"] == sh]
        out[sh] = risk_control_tau([c["conf"] for c in grp], [c["correct"] for c in grp], eps) if grp else math.inf
    return out


def cross_fit(items: List[Dict], eps: float, k: int = 5, seed: int = 0) -> List[float]:
    """items: dicts with 'conf', 'correct', 'kind', 'shape'. Returns the tau applied to every item.
    Folds are stratified by the benchmark split; tau is chosen per shape on the other folds."""
    idx = list(range(len(items)))
    rnd = random.Random(seed)
    folds: List[List[int]] = [[] for _ in range(k)]
    for kind in sorted({it["kind"] for it in items}):          # stratified by single / multi
        group = [i for i in idx if items[i]["kind"] == kind]
        rnd.shuffle(group)
        for j, i in enumerate(group):
            folds[j % k].append(i)
    taus = [math.inf] * len(items)
    for f in range(k):
        held = set(folds[f])
        cal = [items[i] for i in idx if i not in held]
        taus_f = tau_by_shape(cal, eps)
        for i in folds[f]:
            taus[i] = taus_f[items[i]["shape"]]
    return taus


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> Tuple[int, int, float]:
    """Returns (a right & b wrong, a wrong & b right, two-sided exact p-value)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    rnd = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rnd.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]
