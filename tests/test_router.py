"""Tests for the proposed router (registry mining, Tier 1, orchestrator, gate, calibration).

Uses a small invented enterprise-style tool set (Billing / ServiceDesk / FSM / CPQ), so it needs no network
and also shows the registry is mined for enterprise tools without hand-written rules.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hirag.calibrate import cross_fit, mcnemar_exact, risk_control_tau   # noqa: E402
from hirag.data import MCPBench, Service, Tool, make_tool_id             # noqa: E402
from hirag.registry import build_registry, damerau1, split_words         # noqa: E402
from hirag.router import HybridRouter, split_request, tier1_rank         # noqa: E402

TOOLS = [  # (service, type, path, summary, params)
    ("billing", "billing", "/get_invoice", "Get an invoice by its invoice ID", ["invoice_id"]),
    ("billing", "billing", "/hold_invoice", "Put an invoice on hold", ["invoice_id"]),
    ("billing", "billing", "/create_credit_note", "Create a credit note for an invoice", ["invoice_id", "amount"]),
    ("servicedesk", "servicedesk", "/create_ticket", "Create a new support ticket", ["subject", "body"]),
    ("servicedesk", "servicedesk", "/close_ticket", "Close a support ticket", ["ticket_id"]),
    ("servicedesk", "servicedesk", "/get_ticket", "Get details of a support ticket", ["ticket_id"]),
    ("fsm", "field_service", "/create_work_order", "Create a field work order for a technician", ["site"]),
    ("fsm", "field_service", "/list_technicians", "List available field technicians", []),
    ("cpq", "cpq", "/create_quote", "Create a price quote for a customer", ["customer_id", "products"]),
    ("cpq", "cpq", "/get_quote", "Get a quote by its quote ID", ["quote_id"]),
]


def toy_bench() -> MCPBench:
    tools, services = [], {}
    for svc, typ, path, summ, params in TOOLS:
        tid = make_tool_id(svc, path)
        tools.append(Tool(tid, svc, typ, summ, path, "POST", [{"name": p, "required": True} for p in params]))
        services.setdefault(svc, Service(svc, f"{svc} service", typ)).tools.append(tid)
    return MCPBench(tools, services, [], [])


def test_registry_mining():
    reg = build_registry(toy_bench())
    r = reg.by_id["servicedesk_close_ticket"]
    assert r.action == "delete" or "close" in r.objects or r.action is not None
    assert "ticket" in r.objects
    assert reg.by_id["billing_get_invoice"].action == "get"
    assert reg.by_id["cpq_create_quote"].action == "create"


def test_tier1_routes_routine_requests():
    reg = build_registry(toy_bench())
    assert tier1_rank("show invoice INV-1023", reg)[0].tool_id == "billing_get_invoice"
    assert tier1_rank("create a new quote for customer C-77", reg)[0].tool_id == "cpq_create_quote"
    assert tier1_rank("list technicians available tomorrow", reg)[0].tool_id == "fsm_list_technicians"


def test_orchestrator_splits_multi_domain():
    parts = split_request("Create a support ticket for the outage and then put invoice INV-9 on hold")
    assert len(parts) == 2
    reg = build_registry(toy_bench())
    router = HybridRouter(reg, tau=0.0)
    d = router.route("Create a support ticket for the outage and then put invoice INV-9 on hold")
    assert d.tier == "deterministic"
    assert d.tools == ["servicedesk_create_ticket", "billing_hold_invoice"]
    assert d.llm_calls == 0


def test_gate_escalates_to_fallback():
    reg = build_registry(toy_bench())
    calls = []

    def fallback(q):
        calls.append(q)
        return ["billing_get_invoice"], {"llm_calls": 1, "prompt_tokens": 900}

    router = HybridRouter(reg, tau=10.0, fallback=fallback)      # impossible tau -> always escalate
    d = router.route("show invoice INV-1023")
    assert d.tier == "fallback" and d.llm_calls == 1 and calls


def test_consensus_gate_requires_agreement():
    reg = build_registry(toy_bench())
    router = HybridRouter(reg, tau=0.0, retriever=lambda text: "cpq_get_quote")   # retriever disagrees
    d = router.route("show invoice INV-1023")
    assert d.tier == "fallback"
    router = HybridRouter(reg, tau=0.0, retriever=lambda text: "billing_get_invoice")
    assert router.route("show invoice INV-1023").tier == "deterministic"


def test_risk_control_tau():
    conf = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    ok = [True, True, True, True, False, False]
    assert risk_control_tau(conf, ok, eps=0.2) == 0.6         # (0+1)/(4+1) = 0.2
    assert risk_control_tau(conf, ok, eps=0.1) == math.inf    # needs >= 9 correct in a row
    items = [{"conf": c, "correct": o, "kind": "single", "shape": 1} for c, o in zip(conf * 5, ok * 5)]
    taus = cross_fit(items, eps=0.2, k=5)
    assert len(taus) == len(items)


def test_mcnemar():
    a = [True] * 10 + [False] * 0
    b = [True] * 5 + [False] * 5
    n01, n10, p = mcnemar_exact(a, b)
    assert (n01, n10) == (5, 0) and abs(p - 0.0625) < 1e-9


def test_helpers():
    assert damerau1("github", "gihtub") and damerau1("bilibili", "bilibli") and not damerau1("redis", "radius")
    assert not damerau1("arvix", "arxiv")        # non-adjacent swap = distance 2
    assert split_words("getParkDetails") == ["get", "park", "details"]


def test_index_matches_full_scan():
    """The inverted index used for large registries (ToolBench) gives the same s1 and margin as a full scan."""
    from hirag import router as R
    reg = build_registry(toy_bench())
    qs = ["show invoice INV-1023", "create a quote", "list technicians", "close ticket 4521 please",
          "something unrelated entirely", "get the quote and the invoice"]
    old = R.INDEX_MIN_RULES
    try:
        for q in qs:
            R.INDEX_MIN_RULES = 0
            a = R.tier1_rank(q, reg)
            R.INDEX_MIN_RULES = 10 ** 9
            f = R.tier1_rank(q, reg)
            assert abs(a[0].score - f[0].score) < 1e-9 and abs(R.confidence(a) - R.confidence(f)) < 1e-9
    finally:
        R.INDEX_MIN_RULES = old


def test_self_distillation_learns_from_logs():
    """A request the static rules cannot route is learned from logged (query, tool) pairs."""
    from hirag.distill import Distiller, with_learned
    reg = build_registry(toy_bench())
    q = "the outage at the Chennai site needs somebody on the ground"
    assert tier1_rank(q, reg)[0].tool_id != "fsm_create_work_order"
    logs = [("outage at the Pune site needs somebody on the ground", "fsm_create_work_order"),
            ("we need somebody on the ground for the outage", "fsm_create_work_order"),
            ("send somebody on the ground to fix the outage", "fsm_create_work_order"),
            ("show invoice INV-1 for site 9", "billing_get_invoice"),
            ("display the invoice for the site", "billing_get_invoice")]
    learned = Distiller(min_count=2, min_purity=0.6).fit(logs)
    assert "w:outag" in learned["fsm_create_work_order"]
    assert not any(f[2:].isdigit() for f in learned["fsm_create_work_order"])     # no IDs memorised
    assert tier1_rank(q, with_learned(reg, learned))[0].tool_id == "fsm_create_work_order"
