"""Tier 1 rule registry for the proposed Deterministic-LLM router.

The registry is MINED AUTOMATICALLY from tool metadata only (tool path, one-line summary,
service name/title, parameter names). No test query is read while building it, so it can be
evaluated on MCPBench without test leakage, and it can be rebuilt for a new tool set (e.g. the
enterprise CPQ / Billing / FSM / ServiceDesk tools) with no manual rule writing.

For every tool we store
  action    canonical verb of the tool        e.g. get_pull_request   -> "get"
  objects   object words with weights          e.g. {"pull": 1.0, "request": 1.0, "specif": 0.5}
  anchors   distinctive service words          e.g. {"github"}           (service hint)
  entities  typed parameters with a pattern    e.g. url, ip, file path, file extension

The only hand-written parts are GENERIC English lexicons (verb synonyms, parameter-type patterns,
file-extension families). They are not tied to MCPBench or to any test query.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .data import MCPBench, Tool

# ---------------------------------------------------------------------------------------------
# Generic lexicons (domain-independent)
# ---------------------------------------------------------------------------------------------

# canonical action -> surface verbs (unstemmed; stemmed at load time)
VERB_LEXICON: Dict[str, List[str]] = {
    "search":   ["search", "find", "look", "lookup", "discover", "seek", "locate", "browse", "query"],
    "get":      ["get", "fetch", "retrieve", "obtain", "show", "view", "see", "display", "check",
                 "tell", "detail", "info", "information", "know", "learn", "lookup"],
    "list":     ["list", "enumerate", "all", "which"],
    "read":     ["read", "open", "extract", "load", "content", "contents"],
    "write":    ["write", "save", "export", "store", "output"],
    "create":   ["create", "add", "new", "make", "generate", "book", "schedule", "start", "init"],
    "update":   ["update", "edit", "modify", "change", "replace", "insert", "revise", "reschedule"],
    "delete":   ["delete", "remove", "cancel", "clear", "erase", "drop"],
    "send":     ["send", "post", "notify", "message"],
    "convert":  ["convert", "translate", "transform", "turn"],
    "count":    ["count", "number", "how many"],
    "analyze":  ["analyze", "analyse", "analysis", "examine", "inspect", "describe"],
    "plan":     ["plan", "route", "directions", "navigate", "go"],
    "move":     ["move", "rename", "relocate"],
    "merge":    ["merge"],
    "fork":     ["fork"],
    "push":     ["push", "commit"],
    "download": ["download"],
    "execute":  ["execute", "run", "evaluate"],
    "compare":  ["compare", "versus", "vs"],
    "calculate": ["calculate", "compute", "measure"],
    "set":      ["set", "assign", "put"],
    "click":    ["click", "press"],
    "hover":    ["hover"],
    "fill":     ["fill", "type", "enter"],
    "select":   ["select", "choose", "pick"],
    "screenshot": ["screenshot", "capture"],
}

# parameter-name pattern -> entity type; entity type -> regex over the raw query
PARAM_ENTITY = [
    (re.compile(r"(^|_)(url|link|uri)$|video_url|image_url", re.I), "url"),
    (re.compile(r"^ip$", re.I), "ip"),
    (re.compile(r"path|filepath|file_path|directory", re.I), "path"),
    (re.compile(r"^(symbol|ticker)$", re.I), "ticker"),
    (re.compile(r"timezone", re.I), "timezone"),
    (re.compile(r"latitude|longitude|^location$|^locations$", re.I), "coords"),
    (re.compile(r"sql", re.I), "sql"),
    (re.compile(r"email", re.I), "email"),
]
ENTITY_REGEX = {
    "url": re.compile(r"https?://\S+|www\.\S+", re.I),
    "ip": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "path": re.compile(r"(?:^|\s)(?:~?/[\w.\-]+)+|[A-Za-z]:\\\S+"),
    "ticker": re.compile(r"\b[A-Z]{2,5}\b"),
    "timezone": re.compile(r"\b(?:[A-Z][a-z]+/[A-Z][A-Za-z_]+|UTC|GMT|[A-Z]{3,4}T)\b"),
    "coords": re.compile(r"-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+"),
    "sql": re.compile(r"\b(?:select|insert|update|delete)\b.+\b(?:from|into|set)\b", re.I),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
}
# file families: a word in the tool text -> extensions that point to it
FILE_FAMILIES = {
    "docx": ["docx", "doc"], "pdf": ["pdf"], "excel": ["xlsx", "xls"],
    "image": ["png", "jpg", "jpeg", "gif", "webp"],
}
EXT_REGEX = re.compile(r"\.([a-z0-9]{2,5})\b", re.I)

GENERIC_SERVICE_WORDS = {"mcp", "service", "tools", "tool", "api", "server", "the", "com", "find",
                         "management", "and", "of"}

STOPWORDS = None
_STEM = None


def _stem(w: str) -> str:
    global _STEM
    if _STEM is None:
        from nltk.stem import PorterStemmer
        _STEM = PorterStemmer()
    return _STEM.stem(w)


def _stop() -> Set[str]:
    global STOPWORDS
    if STOPWORDS is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        STOPWORDS = set(ENGLISH_STOP_WORDS) - {"all", "find", "get", "show", "see", "move", "fill",
                                                 "detail", "top", "first", "last", "back", "describe",
                                                 "system", "call", "name", "found"}
    return STOPWORDS


def split_words(text: str) -> List[str]:
    """Lowercase words; splits camelCase, snake_case and kebab-case ('getParkDetails' -> get park details)."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " ").replace("-", " "))


def norm_tokens(text: str, keep_stop: bool = False) -> List[str]:
    stop = _stop()
    return [_stem(w) for w in split_words(text) if keep_stop or w not in stop]


# stemmed verb -> set of canonical actions
VERB_INDEX: Dict[str, Set[str]] = {}
for _act, _words in VERB_LEXICON.items():
    for _w in _words:
        for _t in [_stem(x) for x in _w.split()][:1]:
            VERB_INDEX.setdefault(_t, set()).add(_act)


def damerau1(a: str, b: str) -> bool:
    """True if a and b are within Damerau-Levenshtein distance 1 (handles typos like arvix/arxiv)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    if la > lb:
        a, b = b, a
    for i in range(len(b)):
        if a == b[:i] + b[i + 1:]:
            return True
    return False


# ---------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------

@dataclass
class ToolRule:
    tool_id: str
    service: str
    type: str
    action: Optional[str]                     # canonical action from the tool path, or None
    objects: Dict[str, float]                 # stemmed object word -> weight (path 1.0, summary 0.5)
    anchors: Set[str]                         # distinctive service words (raw lowercase, unstemmed)
    entities: Set[str]                        # entity types expected from parameters
    extensions: Set[str] = field(default_factory=set)
    n_required: int = 0
    alt_actions: Set[str] = field(default_factory=set)   # action of the summary's first verb

    @property
    def actions(self) -> Set[str]:
        return ({self.action} if self.action else set()) | self.alt_actions

    def explain(self) -> str:
        return (f"{self.tool_id}: actions={sorted(self.actions)} objects={sorted(self.objects)} "
                f"anchors={sorted(self.anchors)} entities={sorted(self.entities)} ext={sorted(self.extensions)}")


@dataclass
class Registry:
    rules: List[ToolRule]
    idf: Dict[str, float]                     # IDF of object words over tools
    anchor_df: Dict[str, int]                 # in how many services an anchor word appears
    services: Dict[str, str]                  # service -> type
    learned: Dict[str, Dict[str, float]] = field(default_factory=dict)   # tool -> {feature: purity} (distill.py)

    @property
    def by_id(self) -> Dict[str, ToolRule]:
        return {r.tool_id: r for r in self.rules}

    def to_json(self) -> Dict:
        return {"rules": [{"tool_id": r.tool_id, "service": r.service, "type": r.type, "action": r.action,
                           "objects": r.objects, "anchors": sorted(r.anchors), "entities": sorted(r.entities),
                           "extensions": sorted(r.extensions)} for r in self.rules]}


def _service_anchors(name: str, title: str) -> Set[str]:
    """Service-name words. A multi-word name ('GoogleDrive', 'hugging_face') counts only as a whole
    ('googledrive', 'huggingface'), so that 'drive from A to B' does not hint GoogleDrive."""
    name_words = [w for w in split_words(re.sub(r"(?i)_?mcp$", "", name)) if w not in GENERIC_SERVICE_WORDS]
    words = name_words if len(name_words) == 1 else []
    words += [w for w in split_words(title) if w not in GENERIC_SERVICE_WORDS and w not in name_words]
    words = [w for w in words if len(w) >= 3]
    compact = re.sub(r"[^a-z0-9]", "", re.sub(r"(?i)_?mcp$", "", name).lower())   # hugging_face -> huggingface
    if len(compact) >= 3 and compact not in GENERIC_SERVICE_WORDS:
        words.append(compact)
    return set(words)


def _tool_action(path_words: List[str], service_words: Set[str]) -> Optional[str]:
    """First verb of the tool path; words of the service name are skipped unless nothing else is a verb
    (e.g. bing_search/bing_search -> "search", fetch/fetch -> "get")."""
    for skip in (True, False):
        for w in path_words:
            if skip and w in service_words:
                continue
            s = _stem(w)
            acts = VERB_INDEX.get(s)
            if acts:
                return s if s in acts else sorted(acts)[0]     # prefer the action named by the word itself
    return None


def _proper_nouns(summary: str) -> Set[str]:
    """Brand-like words in a summary (capitalised, not sentence-initial): 'LeetCode', 'arXiv', 'GitHub'."""
    toks = re.findall(r"[A-Za-z][A-Za-z0-9.]*", summary or "")
    out = set()
    for i, w in enumerate(toks):
        w = w.rstrip(".")
        if i > 0 and len(w) >= 3 and any(c.isupper() for c in w) and w.upper() not in {"API", "ID", "URL", "MCP", "IP", "POI", "SDK"}:
            out.add(w.lower())
    return out


def build_registry(bench: MCPBench) -> Registry:
    # service anchors, kept only if distinctive (appear in <= 2 services)
    raw_anchors = {s.name: _service_anchors(s.name, s.title) for s in bench.services.values()}
    for t in bench.tools:                              # brand names that appear in the tool summaries
        raw_anchors[t.service] |= {w for w in _proper_nouns(t.summary) if w not in GENERIC_SERVICE_WORDS}
    anchor_df = Counter(w for ws in raw_anchors.values() for w in ws)
    stop = _stop()

    # pass 1: action + object words of every tool
    mined = []
    for t in bench.tools:
        name_words = set(split_words(t.service))
        path_words = split_words(t.path)
        action = _tool_action(path_words, name_words)
        act_stems = {_stem(w) for w in VERB_LEXICON.get(action, [])} if action else set()
        objects: Dict[str, float] = {}
        for w in path_words:                           # path words: weight 1.0
            if w in stop or len(w) < 2:
                continue
            s = _stem(w)
            if s not in act_stems:
                objects[s] = 1.0
        for w in split_words(t.summary):               # summary words: weight 0.5
            if w in stop or len(w) < 3:
                continue
            s = _stem(w)
            if s not in act_stems and s not in VERB_INDEX:
                objects.setdefault(s, 0.5)
        first = split_words(t.summary)[:1]
        alt = set()
        if first:
            fs = _stem(first[0])
            if fs in VERB_INDEX:
                alt = {fs} if fs in VERB_INDEX[fs] else {sorted(VERB_INDEX[fs])[0]}
        mined.append((t, action, objects, alt))

    # an anchor must be distinctive: in <= 2 services' names, not a verb, and not an ordinary
    # object word used by tools of 3+ other services (drops 'search', 'file', 'text', 'update', ...)
    obj_services: Dict[str, Set[str]] = {}
    for t, _, objects, _ in mined:
        for s in objects:
            obj_services.setdefault(s, set()).add(t.service)

    def distinctive(w: str, service: str) -> bool:
        s = _stem(w)
        return (anchor_df[w] <= 2 and s not in VERB_INDEX
                and len(obj_services.get(s, set()) - {service}) < 3)

    rules: List[ToolRule] = []
    obj_lists = []
    for t, action, objects, alt in mined:
        ents: Set[str] = set()
        for p in t.parameters or []:
            for rx, ent in PARAM_ENTITY:
                if rx.search(p.get("name", "")):
                    ents.add(ent)
        text_words = set(split_words(t.service + " " + t.path + " " + t.summary))
        exts = {e for fam, es in FILE_FAMILIES.items() if fam in text_words for e in es}
        anchors = {w for w in raw_anchors[t.service] if distinctive(w, t.service)}
        rules.append(ToolRule(t.tool_id, t.service, t.type, action, objects, anchors, ents, exts,
                              sum(1 for p in t.parameters or [] if p.get("required")), alt - {action}))
        obj_lists.append(set(objects))
    n = len(rules)
    df = Counter(w for s in obj_lists for w in s)
    idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
    return Registry(rules, idf, dict(anchor_df), {s.name: s.type for s in bench.services.values()})
