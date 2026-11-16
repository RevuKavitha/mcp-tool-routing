"""Proposed system: Deterministic-LLM hybrid tool router (Tiers 1-3 + orchestrator).

  Tier 1  deterministic rule matching (no LLM, no neural model), registry from hirag/registry.py
          s(q, t) = 0.4 * action + 0.3 * object + 0.2 * service_hint + 0.1 * entity
  Tier 2  confidence gate: c(q) = s1 - s2 (margin between best and second-best tool, ties -> 0).
          Consensus gate (default when a retriever is given): the rule's tool must ALSO be the top-1
          tool of a cheap retriever (BM25, or Hi-RAG Stage 1 = BM25 + dense) -- two independent signals
          agreeing -- and c(q) >= tau. Then the tool is called directly (0 LLM calls).
          tau is calibrated on held-out labelled queries with a risk-control rule (hirag/calibrate.py).
  Tier 3  fallback: Hi-RAG (Stage 1 + Stage 2) + LLM tool calling over the top-K services only.
  Orchestrator  multi-step / multi-domain requests are split into sub-requests ("..., then ...",
          "... and create ..."); the request is deterministic only if EVERY sub-request passes the gate.

Every decision is returned as a RouteDecision that records the tier, the rule evidence and the
confidence, so the deterministic path is auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .registry import (ENTITY_REGEX, EXT_REGEX, VERB_INDEX, Registry, ToolRule,
                       norm_tokens, split_words)

WEIGHTS = {"action": 0.4, "object": 0.3, "hint": 0.2, "entity": 0.1}
WEIGHTS_DISTILL = {"action": 0.30, "object": 0.25, "hint": 0.15, "entity": 0.05, "learned": 0.25}


# ---------------------------------------------------------------------------------------------
# Query features
# ---------------------------------------------------------------------------------------------

@dataclass
class QueryFeatures:
    text: str
    stems: List[str]
    actions: Dict[str, int]          # canonical action -> first word position
    raw_words: List[str]             # lowercase words without camelCase splitting (for brand names)
    compact: set                     # raw words + adjacent pairs joined ("hugging face" -> "huggingface")
    entities: set
    extensions: set
    feats: set = field(default_factory=set)      # word / word-pair features for learned rules


def query_features(text: str) -> QueryFeatures:
    stems = norm_tokens(text, keep_stop=True)
    actions: Dict[str, int] = {}
    for i, s in enumerate(stems):
        for a in VERB_INDEX.get(s, ()):
            actions.setdefault(a, i)
    raw = re.findall(r"[a-z0-9]+", text.lower())
    compact = set(raw) | {raw[i] + raw[i + 1] for i in range(len(raw) - 1)}
    ents = {e for e, rx in ENTITY_REGEX.items() if rx.search(text)}
    exts = {m.lower() for m in EXT_REGEX.findall(text)}
    from .distill import query_feature_set
    return QueryFeatures(text, stems, actions, raw, compact, ents, exts, query_feature_set(text))


# ---------------------------------------------------------------------------------------------
# Tier 1: rule matching
# ---------------------------------------------------------------------------------------------

@dataclass
class ToolScore:
    tool_id: str
    score: float
    parts: Dict[str, float]


def _anchor_hit(anchors, qf: QueryFeatures) -> Optional[str]:
    """Exact match of a service word (or joined word pair). Fuzzy typo matching was dropped: it matched
    'table' to the anchor 'stable' (Stable Diffusion) and never fixed a real typo on MCPBench."""
    for a in anchors:
        if a in qf.compact:
            return a
    return None


def score_tool(rule: ToolRule, qf: QueryFeatures, reg: Registry) -> ToolScore:
    qstems = set(qf.stems)
    a = 1.0 if rule.actions & set(qf.actions) else 0.0
    num = sum(w * reg.idf.get(o, 1.0) for o, w in rule.objects.items() if o in qstems)
    den = sum(w * reg.idf.get(o, 1.0) for o, w in rule.objects.items())
    o = num / den if den else 0.0
    h = 1.0 if _anchor_hit(rule.anchors, qf) else 0.0
    e = 1.0 if (rule.entities & qf.entities) or (rule.extensions & qf.extensions) else 0.0
    parts = {"action": a, "object": o, "hint": h, "entity": e}
    if reg.learned:
        lw = reg.learned.get(rule.tool_id)
        miss = 1.0
        if lw:
            for f in qf.feats:
                p = lw.get(f)
                if p:
                    miss *= 1.0 - p
        parts["learned"] = 1.0 - miss
        return ToolScore(rule.tool_id, sum(WEIGHTS_DISTILL[k] * v for k, v in parts.items()), parts)
    return ToolScore(rule.tool_id, sum(WEIGHTS[k] * v for k, v in parts.items()), parts)


INDEX_MIN_RULES = 1000          # above this size, score only tools that share a word/entity with the query


def _build_index(reg: Registry) -> Dict[str, List[int]]:
    idx: Dict[str, List[int]] = {}
    for i, r in enumerate(reg.rules):
        keys = {"o:" + o for o in r.objects} | {"a:" + a for a in r.anchors} | \
               {"e:" + e for e in r.entities} | {"x:" + x for x in r.extensions} | \
               {"l:" + f for f in reg.learned.get(r.tool_id, {})}
        for k in keys:
            idx.setdefault(k, []).append(i)
    return idx


def tier1_rank(text: str, reg: Registry) -> List[ToolScore]:
    """Score every tool (small registries) or, for large ones (e.g. ToolBench, 10k tools), only the tools
    that share an object word, service anchor or entity with the query. A tool outside that set can only
    score the action weight, so one placeholder with that score keeps the margin c = s1 - s2 exact.
    (Fuzzy typo matching of anchors is used only in the full scan.)"""
    qf = query_features(text)
    if len(reg.rules) < INDEX_MIN_RULES:
        scores = [score_tool(r, qf, reg) for r in reg.rules]
    else:
        idx = getattr(reg, "_index", None)
        if idx is None:
            idx = reg._index = _build_index(reg)
        keys = {"o:" + s for s in qf.stems} | {"a:" + w for w in qf.compact} | \
               {"e:" + e for e in qf.entities} | {"x:" + x for x in qf.extensions} | {"l:" + f for f in qf.feats}
        cand = sorted({i for k in keys for i in idx.get(k, ())})
        scores = [score_tool(reg.rules[i], qf, reg) for i in cand]
        cset = set(cand)
        rest = (i for i, r in enumerate(reg.rules) if i not in cset and r.actions & set(qf.actions))
        for _, j in zip(range(2), rest):                   # up to 2 action-only tools: ties give margin 0
            w_a = (WEIGHTS_DISTILL if reg.learned else WEIGHTS)["action"]
            scores.append(ToolScore(reg.rules[j].tool_id, w_a, {"action": 1.0, "object": 0.0,
                                                                  "hint": 0.0, "entity": 0.0}))
    scores.sort(key=lambda s: s.score, reverse=True)
    if len(scores) < 2:
        scores.append(ToolScore("<none>", 0.0, {"action": 0.0, "object": 0.0, "hint": 0.0, "entity": 0.0}))
    return scores


def confidence(ranked: Sequence[ToolScore]) -> float:
    """Margin between the best and second-best tool (0 when tied)."""
    if not ranked:
        return 0.0
    s2 = ranked[1].score if len(ranked) > 1 else 0.0
    return ranked[0].score - s2


# ---------------------------------------------------------------------------------------------
# Orchestrator: split multi-step / multi-domain requests
# ---------------------------------------------------------------------------------------------

_SPLIT_RX = re.compile(r"\s*(?:,?\s*(?:and\s+)?then\b|,?\s*after\s+that\b|,?\s*afterwards\b|,?\s*finally\b|"
                       r";|\.\s+(?=[A-Z]))\s*", re.I)


def split_request(text: str) -> List[str]:
    """Split on sequencing words ("then", "after that", ";", sentence breaks), then on
    "and <verb>" / ", <verb>" when the next word is an action verb of the lexicon."""
    parts = [p for p in _SPLIT_RX.split(text) if p and p.strip(" ,.")]
    out: List[str] = []
    for p in parts:
        words = p.split()
        cur: List[str] = []
        for i, w in enumerate(words):
            nxt = re.sub(r"[^a-z]", "", words[i + 1].lower()) if i + 1 < len(words) else ""
            joiner = w.lower() in ("and", "&") or w.endswith(",")
            if joiner and nxt and VERB_INDEX.get(norm_tokens(nxt, keep_stop=True)[0] if norm_tokens(nxt, keep_stop=True) else "") \
                    and len(cur) >= 3:
                if w.endswith(","):
                    cur.append(w.rstrip(","))
                out.append(" ".join(cur))
                cur = []
                continue
            cur.append(w)
        if cur:
            out.append(" ".join(cur))
    return [s.strip(" ,.") for s in out if s.strip(" ,.")] or [text]


# ---------------------------------------------------------------------------------------------
# Router (Tiers 1-3)
# ---------------------------------------------------------------------------------------------

@dataclass
class SubDecision:
    text: str
    tool: str
    confidence: float
    parts: Dict[str, float]
    runner_up: Optional[str]
    retriever_tool: Optional[str] = None      # top-1 tool of the retriever (consensus gate)

    @property
    def agrees(self) -> bool:
        return self.retriever_tool is None or self.retriever_tool == self.tool


@dataclass
class RouteDecision:
    query: str
    tier: str                          # "deterministic" | "fallback"
    tools: List[str]                   # tools called, in order
    confidence: float                  # min confidence over sub-requests
    subs: List[SubDecision] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    fallback_services: List[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [f"[{self.tier}] c={self.confidence:.3f} -> {self.tools}"]
        for s in self.subs:
            lines.append(f"   '{s.text}' -> {s.tool} (c={s.confidence:.3f}, runner-up {s.runner_up}) "
                         + " ".join(f"{k}={v:.2f}" for k, v in s.parts.items()))
        return "\n".join(lines)


FallbackFn = Callable[[str], Tuple[List[str], Dict]]     # query -> (tools called, usage dict)


def actionable(parts: Dict[str, float]) -> bool:
    """A fragment is a separate sub-task only if it names an action AND an object or a service."""
    return parts["action"] > 0 and (parts["object"] > 0 or parts["hint"] > 0 or parts.get("learned", 0) >= 0.5)


class HybridRouter:
    """tau can be one float, or a dict {1: tau_single_step, 2: tau_multi_step} keyed by the number of
    sub-tasks the orchestrator found (known at run time, unlike the benchmark's single/multi label)."""

    def __init__(self, reg: Registry, tau, fallback: Optional[FallbackFn] = None,
                 split: bool = True, max_subs: int = 4,
                 retriever: Optional[Callable[[str], str]] = None):
        """retriever: text -> top-1 tool id (enables the consensus gate); None = rules-only gate."""
        self.reg, self.tau, self.fallback = reg, tau, fallback
        self.split, self.max_subs, self.retriever = split, max_subs, retriever

    @staticmethod
    def shape(subs: Sequence["SubDecision"]) -> int:
        return 1 if len(subs) <= 1 else 2

    def tau_for(self, subs) -> float:
        return self.tau.get(self.shape(subs), float("inf")) if isinstance(self.tau, dict) else self.tau

    def _decide(self, text: str) -> SubDecision:
        ranked = tier1_rank(text, self.reg)
        return SubDecision(text, ranked[0].tool_id, confidence(ranked), ranked[0].parts,
                           ranked[1].tool_id if len(ranked) > 1 else None)

    def tier1(self, text: str) -> Tuple[List[SubDecision], float]:
        pieces = split_request(text) if self.split else [text]
        # merge non-actionable fragments ("display maximum 10 items", "Is there ... software?")
        # into the neighbouring sub-task: they carry parameters or context, not a new task
        groups: List[str] = []
        pending = ""
        for p in pieces:
            if actionable(self._decide(p).parts):
                groups.append((pending + " " + p).strip() if pending else p)
                pending = ""
            elif groups:
                groups[-1] = groups[-1] + ", " + p
            else:
                pending = (pending + " " + p).strip()
        if pending:
            groups = [pending] if not groups else groups[:-1] + [groups[-1] + ", " + pending]
        groups = groups[: self.max_subs] or [text]
        subs = [self._decide(g) for g in groups]
        if self.retriever is not None:
            for sd in subs:
                sd.retriever_tool = self.retriever(sd.text)
        # confidence of the request = weakest sub-task; a sub-task without consensus has confidence 0
        conf = min((s.confidence if s.agrees else 0.0) for s in subs)
        # sub-tasks that resolve to the same tool become one call
        uniq, seen = [], set()
        for s in subs:
            if s.tool not in seen:
                seen.add(s.tool)
                uniq.append(s)
        return uniq, conf

    def route(self, text: str) -> RouteDecision:
        subs, conf = self.tier1(text)
        if all(s.agrees for s in subs) and conf >= self.tau_for(subs):
            return RouteDecision(text, "deterministic", [s.tool for s in subs], conf, subs)
        d = RouteDecision(text, "fallback", [], conf, subs)
        if self.fallback is not None:
            tools, usage = self.fallback(text)
            d.tools = list(tools)
            d.llm_calls = usage.get("llm_calls", 1)
            d.prompt_tokens = usage.get("prompt_tokens", 0)
            d.completion_tokens = usage.get("completion_tokens", 0)
            d.fallback_services = usage.get("services", [])
        return d


# ---------------------------------------------------------------------------------------------
# Online fallback: Hi-RAG retrieval + LLM tool calling (uses the reproduction's own modules)
# ---------------------------------------------------------------------------------------------

def make_hirag_fallback(bench, selector, llm, k: int, prompt: str, max_turns: int) -> FallbackFn:
    """selector = hirag.selectors.make_selector('hirag', ...); llm = hirag.llm.LLMClient(...)."""
    from .tools_schema import functions_for_services

    def fb(text: str):
        services = selector(text, k)
        res = llm.select_tools(text, functions_for_services(bench, services), prompt, max_turns)
        return [c.lower() for c in res.tool_calls], {"llm_calls": res.turns, "services": services,
                                                     "prompt_tokens": res.prompt_tokens,
                                                     "completion_tokens": res.completion_tokens}
    return fb
