"""Live demo of the proposed router (no GPU, no API key): shows the tier, the chosen tool and the rule
evidence for each request. Use --tau to set the gate (from run_router.py's calibration).

  python scripts/demo_router.py
  python scripts/demo_router.py "Get the current time in Tokyo, then search Bing for today's news"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hirag.data import load_mcpbench          # noqa: E402
from hirag.registry import build_registry     # noqa: E402
from hirag.router import HybridRouter         # noqa: E402
from hirag.sparse import BM25Retriever        # noqa: E402

EXAMPLES = [
    "Download the arXiv paper with ID 2401.12345",                                  # routine -> no LLM
    "Read the contents of /home/user/report.docx",                                  # routine -> no LLM
    "Get the current time in Asia/Tokyo, then search Bing for today's top news",    # multi-step -> no LLM
    "Search YouTube for videos about transformers",                                 # retriever disagrees -> LLM
    "I want something fun to do this weekend",                                      # no rule fires -> LLM
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*")
    ap.add_argument("--tau", type=float, default=0.10, help="gate threshold (single-step); multi-step uses --tau2")
    ap.add_argument("--tau2", type=float, default=0.20)
    a = ap.parse_args()
    bench = load_mcpbench()
    reg = build_registry(bench)
    ids = [t.tool_id for t in bench.tools]
    bm = BM25Retriever([t.summary for t in bench.tools], "clean")
    router = HybridRouter(reg, tau={1: a.tau, 2: a.tau2}, retriever=lambda q: ids[bm.rank(q, 1)[0]],
                          fallback=lambda q: (["<Hi-RAG + LLM>"], {"llm_calls": 1}))
    for q in a.queries or EXAMPLES:
        d = router.route(q)
        print(f"\nQ: {q}")
        print(f"   tier: {d.tier.upper()}  (confidence {d.confidence:.3f}, LLM calls {d.llm_calls})")
        for s in d.subs:
            ev = " ".join(f"{k}={v:.2f}" for k, v in s.parts.items())
            print(f"   - '{s.text[:60]}' -> {s.tool}  [{ev}] retriever={s.retriever_tool} "
                  f"{'agree' if s.agrees else 'DISAGREE'}")


if __name__ == "__main__":
    main()
