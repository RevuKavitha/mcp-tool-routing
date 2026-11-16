"""Self-distillation: the LLM fallback teaches the deterministic tier.

Every escalated request is answered by Hi-RAG + LLM; its (query, chosen tool) pair is logged. From the log we
learn which query words are reliable evidence for each tool and add them to that tool's rule:

  feature f = a content-word stem, or a pair of adjacent stems ("credit note")
  keep (f -> tool t) if  count(f, t) >= min_count  and  purity p = count(f, t) / (count(f) + prior) >= min_purity
  learned score  L(q, t) = 1 - prod_{f in q, f -> t} (1 - p_f)          (noisy-OR, in [0, 1])

The router then scores  s = 0.30 action + 0.25 object + 0.15 hint + 0.05 entity + 0.25 learned
(the static weights are used when nothing has been learned). Learned rules stay readable (word -> tool, p),
and the confidence gate is re-calibrated on held-out logs after every update, so the error target still holds.
"""
from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

from .registry import Registry, norm_tokens


def _wordlike(s: str) -> bool:
    """Real words only: IDs and values ('g9_c3', 'o73n', '2401') are request-specific, not evidence of a tool.
    (Generated logs reuse such values, so without this filter the router would memorise them.)"""
    return len(s) >= 3 and s.isalpha()


def query_feature_set(text: str) -> Set[str]:
    stems = [s for s in norm_tokens(text)]             # content words (stop-words removed), stemmed
    feats = {"w:" + s for s in stems if _wordlike(s)}
    feats |= {"b:" + a + "_" + b for a, b in zip(stems, stems[1:]) if _wordlike(a) and _wordlike(b)}
    return feats


@dataclass
class Distiller:
    min_count: int = 2
    min_purity: float = 0.6
    prior: float = 1.0
    max_per_tool: int = 40

    def fit(self, logs: Iterable[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
        pair: Counter = Counter()
        total: Counter = Counter()
        for text, tool in logs:
            for f in query_feature_set(text):
                pair[(f, tool)] += 1
                total[f] += 1
        learned: Dict[str, Dict[str, float]] = defaultdict(dict)
        for (f, t), c in pair.items():
            p = c / (total[f] + self.prior)
            if c >= self.min_count and p >= self.min_purity:
                learned[t][f] = round(p, 4)
        for t in list(learned):                        # keep the strongest features per tool
            top = sorted(learned[t].items(), key=lambda x: -x[1])[: self.max_per_tool]
            learned[t] = dict(top)
        return dict(learned)


def with_learned(reg: Registry, learned: Dict[str, Dict[str, float]]) -> Registry:
    """A copy of the registry carrying learned rules (the static rules are unchanged)."""
    new = dataclasses.replace(reg)
    new.learned = learned
    if hasattr(new, "_index"):
        del new._index
    return new


def explain_learned(learned: Dict[str, Dict[str, float]], tool: str, k: int = 8) -> str:
    feats = sorted(learned.get(tool, {}).items(), key=lambda x: -x[1])[:k]
    return f"{tool}: " + ", ".join(f"{f[2:]}({p:.2f})" for f, p in feats)
