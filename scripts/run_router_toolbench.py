"""Router on ToolBench (StableToolBench G1/G2/G3 instruction, 10k+ tools): scalability and safety of the gate.

Same protocol as the reproduction's ToolLLM test (paper Table 6): rank tools, binary NDCG@1/3/5, no LLM.
  baseline ranking : the retriever alone (BM25 here; Hi-RAG Stage 1 with --retriever stage1 on a GPU)
  router ranking   : if the gate passes, the rule-selected tool(s) first, then the retriever's ranking;
                     otherwise the retriever's ranking unchanged (escalation).
So the router can only change a query's result when it routes deterministically: the question is whether
it does so rarely-but-correctly (safe) or often-and-wrongly (unsafe) on free-text, poorly described tools.

tau comes from (a) 5-fold cross-fitting on ToolBench, and (b) TRANSFER: tau calibrated on all 261 MCPBench
queries with the same retriever type, applied unchanged to ToolBench (zero-shot gate).

  python scripts/run_router_toolbench.py                      # BM25 consensus, CPU, ~3-5 min
  python scripts/run_router_toolbench.py --retriever stage1   # Colab GPU (bge-large-en)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hirag.calibrate import cross_fit, mcnemar_exact, tau_by_shape     # noqa: E402
from hirag.data import load_mcpbench                                  # noqa: E402
from hirag.metrics import ndcg_at_k                                   # noqa: E402
from hirag.registry import build_registry                             # noqa: E402
from hirag.router import HybridRouter                                 # noqa: E402
from hirag.toolbench import load_toolbench                            # noqa: E402

EPS_GRID = [0.05, 0.10, 0.20]


def make_ranker(bench, kind, embed_model, backend):
    """text -> ranked tool ids (top 20)."""
    ids = [t.tool_id for t in bench.tools]
    if kind == "bm25":
        from hirag.sparse import BM25Retriever
        bm = BM25Retriever([t.summary for t in bench.tools], "clean")
        return lambda text: [ids[i] for i in bm.rank(text, 20)]
    from hirag.dense import Embedder
    from hirag.stage1 import Stage1, Stage1Config
    s1 = Stage1(bench, Embedder(embed_model, backend), Stage1Config(tokenizer="clean"))
    return lambda text: s1.run(text).tools[:20]


def tier1_items(bench, router, rank_cache, det_ok):
    out = []
    for q in bench.single + bench.multi:
        subs, conf = router.tier1(q.text)
        out.append({"q": q, "kind": q.kind, "conf": conf, "shape": router.shape(subs),
                    "tools": [s.tool for s in subs], "correct": det_ok(subs, q), "subs": subs,
                    "ranking": rank_cache(q.text)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "stage1"])
    ap.add_argument("--embed-model", default="BAAI/bge-large-en")
    ap.add_argument("--embed-backend", default="st", choices=["st", "onnx"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    # ---------- ToolBench ----------
    t0 = time.time()
    tb = load_toolbench()
    reg = build_registry(tb)
    ranker = make_ranker(tb, a.retriever, a.embed_model, a.embed_backend)
    cache = {}

    def rank(text):
        if text not in cache:
            cache[text] = ranker(text)
        return cache[text]

    router = HybridRouter(reg, tau=float("inf"), retriever=lambda text: rank(text)[0])
    det_ok = lambda subs, q: bool(subs) and subs[0].tool in q.gold_tools      # top-1 relevant (NDCG@1 = 1)
    items = tier1_items(tb, router, rank, det_ok)
    print(f"ToolBench: {len(tb.tools)} tools, {len(tb.types)} categories, {len(items)} queries "
          f"(G1 {sum(i['kind']=='G1' for i in items)}, G2 {sum(i['kind']=='G2' for i in items)}, "
          f"G3 {sum(i['kind']=='G3' for i in items)}) | registry {len(reg.rules)} rules | {time.time()-t0:.0f}s")
    t1 = sum(1 for i in items if i["subs"][0].tool in i["q"].gold_tools) / len(items)
    print(f"Tier 1 alone (no gate) top-1 relevant: {100*t1:.1f}% | retriever ({a.retriever}) alone top-1: "
          f"{100*sum(i['ranking'][0] in i['q'].gold_tools for i in items)/len(items):.1f}%")

    # ---------- transfer tau from MCPBench (same retriever type) ----------
    mb = load_mcpbench()
    mreg = build_registry(mb)
    mrank = make_ranker(mb, a.retriever, a.embed_model, a.embed_backend)
    mrouter = HybridRouter(mreg, tau=float("inf"), retriever=lambda text: mrank(text)[0])
    mitems = []
    for q in mb.single + mb.multi:
        subs, conf = mrouter.tier1(q.text)
        tools = [s.tool for s in subs]
        ok = (bool(tools) and tools[0] == q.gold_tools[0]) if q.kind == "single" else all(g in tools for g in q.gold_tools)
        mitems.append({"conf": conf, "shape": mrouter.shape(subs), "correct": ok})

    results = {"settings": vars(a), "n_tools": len(tb.tools), "rows": []}
    print(f"\n{'calibration':<22}{'eps':>5} | {'set':<4} {'0-rule %':>8} {'rule prec %':>11} | "
          f"{'NDCG@1 base':>11} {'router':>7} | {'@3 base':>7} {'router':>7} | {'@5 base':>7} {'router':>7} | McNemar@1")
    for calib in ("cross-fit ToolBench", "transfer MCPBench"):
        for eps in EPS_GRID:
            if calib.startswith("cross"):
                taus = cross_fit(items, eps, a.folds)
            else:
                t = tau_by_shape(mitems, eps)
                taus = [t[it["shape"]] for it in items]
            for g in ("G1", "G2", "G3", "all"):
                idx = [i for i, it in enumerate(items) if g == "all" or it["kind"] == g]
                det = [i for i in idx if items[i]["conf"] >= taus[i] and all(s.agrees for s in items[i]["subs"])]
                base_n, rout_n = {1: [], 3: [], 5: []}, {1: [], 3: [], 5: []}
                for i in idx:
                    it = items[i]
                    base = it["ranking"]
                    rout = (list(dict.fromkeys(it["tools"])) + [t for t in base if t not in it["tools"]]) if i in det else base
                    for k in (1, 3, 5):
                        base_n[k].append(ndcg_at_k(base, it["q"].gold_tools, k))
                        rout_n[k].append(ndcg_at_k(rout, it["q"].gold_tools, k))
                n01, n10, p = mcnemar_exact([x > 0 for x in rout_n[1]], [x > 0 for x in base_n[1]])
                mean = lambda xs: 100 * sum(xs) / len(xs)
                row = {"calibration": calib, "eps": eps, "set": g, "n": len(idx),
                       "rule_rate": 100 * len(det) / len(idx),
                       "rule_precision": 100 * sum(items[i]["correct"] for i in det) / len(det) if det else None,
                       **{f"ndcg{k}_base": mean(base_n[k]) for k in (1, 3, 5)},
                       **{f"ndcg{k}_router": mean(rout_n[k]) for k in (1, 3, 5)},
                       "mcnemar": [n01, n10, p]}
                results["rows"].append(row)
                prec = f"{row['rule_precision']:.1f}" if det else "-"
                print(f"{calib:<22}{eps:>5.2f} | {g:<4} {row['rule_rate']:>8.1f} {prec:>11} | "
                      f"{row['ndcg1_base']:>11.1f} {row['ndcg1_router']:>7.1f} | {row['ndcg3_base']:>7.1f} "
                      f"{row['ndcg3_router']:>7.1f} | {row['ndcg5_base']:>7.1f} {row['ndcg5_router']:>7.1f} | "
                      f"+{n01}/-{n10} p={p:.2f}")
            print()

    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"router_toolbench_{a.retriever}.json")
    json.dump(results, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
