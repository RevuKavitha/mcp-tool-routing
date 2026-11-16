"""Evaluate the proposed Deterministic-LLM hybrid router on MCPBench (no new LLM calls needed).

The router's fallback (Tier 3) for a query is exactly the baseline pipeline it escalates to
(Hi-RAG + Qwen3-32B, temperature 0, cached), so the cascade is evaluated by combining
  - Tier 1 / Tier 2 decisions computed here, with
  - the per-query results the reproduction already saved (results/e2e_<method>_<split>_*.json).

tau is never chosen on the query it is applied to:
  --calib-file train_queries.json   calibrate on the LLM-generated Stage 2 queries (no test leakage), or
  (default) 5-fold cross-fitting on MCPBench: each fold's tau comes from the other 4 folds.

Examples
  python scripts/run_router.py                                         # Tier 1 + gate only
  python scripts/run_router.py --fallback-dir /content/drive/MyDrive/hirag_results_or \
         --calib-file /content/drive/MyDrive/hirag_stage2/train_queries.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hirag.calibrate import bootstrap_ci, cross_fit, mcnemar_exact, tau_by_shape   # noqa: E402
from hirag.data import load_mcpbench                                                   # noqa: E402
from hirag.registry import build_registry                                              # noqa: E402
from hirag.router import HybridRouter                                                  # noqa: E402

EPS_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def load_fallback(dirpath: str, method: str, split: str, model: str = ""):
    """Per-query rows of a saved e2e run; `model` filters by the model tag in the file name."""
    k = 1 if split == "single" else 3
    files = sorted(glob.glob(os.path.join(dirpath, f"e2e_{method}_{split}_k{k}_*.json")))
    files = [f for f in files if model in os.path.basename(f)]
    files = [f for f in files if "PARTIAL" not in f] or files
    if not files:
        return None, None
    d = json.load(open(files[0], encoding="utf-8"))
    return {r["qid"]: r for r in d["rows"]}, files[0]


def det_correct(tools, q) -> bool:
    if q.kind == "single":
        return bool(tools) and tools[0] == q.gold_tools[0]
    return all(g in tools for g in q.gold_tools)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fallback-dir", default=None, help="folder with e2e_<method>_<split>_k*_*.json results")
    ap.add_argument("--fallback-methods", default="hirag,full,colbert,flat,stage1")
    ap.add_argument("--fallback-model", default="qwen3-32b",
                    help="only use result files whose name contains this model tag (all methods must share one LLM)")
    ap.add_argument("--calib-file", default=None, help="train_queries.json (LLM-generated Stage 2 queries)")
    ap.add_argument("--eps", type=float, default=0.05, help="target error of the deterministic path (main row)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-split", action="store_true", help="disable the multi-request orchestrator (ablation)")
    ap.add_argument("--retriever", default="bm25", choices=["none", "bm25", "stage1"],
                    help="consensus gate: none = rules only; bm25 = clean BM25 (CPU); stage1 = Hi-RAG Stage 1 (GPU)")
    ap.add_argument("--embed-model", default="BAAI/bge-large-en")
    ap.add_argument("--embed-backend", default="st", choices=["st", "onnx"])
    ap.add_argument("--stage1-tokenizer", default="jieba", help="Stage 1 BM25 tokenizer (authors: jieba)")
    ap.add_argument("--assume-fallback", default="single=85.1,multi=53.3",
                    help="Hi-RAG accuracy used for the ESTIMATE when no --fallback-dir is given")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    bench = load_mcpbench()
    reg = build_registry(bench)
    tool_ids = [t.tool_id for t in bench.tools]
    retriever, ret_desc = None, "none (rules-only gate)"
    if a.retriever == "bm25":
        from hirag.sparse import BM25Retriever
        bm = BM25Retriever([t.summary for t in bench.tools], "clean")
        retriever, ret_desc = (lambda text: tool_ids[bm.rank(text, 1)[0]]), "BM25 (clean tokenizer), top-1 tool"
    elif a.retriever == "stage1":
        from hirag.dense import Embedder
        from hirag.stage1 import Stage1, Stage1Config
        s1 = Stage1(bench, Embedder(a.embed_model, a.embed_backend), Stage1Config(tokenizer=a.stage1_tokenizer))
        retriever, ret_desc = (lambda text: s1.run(text).tools[0]), f"Hi-RAG Stage 1 ({a.embed_model}), top-1 tool"
    router = HybridRouter(reg, tau=math.inf, split=not a.no_split, retriever=retriever)
    queries = bench.single + bench.multi

    # ---- Tier 1 on every test query (deterministic, ~ms) ----
    t0 = time.perf_counter()
    items = []
    for q in queries:
        subs, conf = router.tier1(q.text)
        tools = [s.tool for s in subs]
        items.append({"q": q, "kind": q.kind, "conf": conf, "tools": tools, "shape": router.shape(subs),
                      "correct": det_correct(tools, q), "subs": subs})
    ms = 1000 * (time.perf_counter() - t0) / len(queries)
    print(bench.summary())
    print(f"Registry: {len(reg.rules)} rules mined from tool metadata | Tier 1 time {ms:.2f} ms/query (CPU, no LLM)")
    for kind in ("single", "multi"):
        sub = [it for it in items if it["kind"] == kind]
        print(f"  Tier 1 alone (no gate), {kind}: {100*sum(i['correct'] for i in sub)/len(sub):.1f}%")

    # ---- calibration ----
    calib_desc = f"{a.folds}-fold cross-fitting on MCPBench"
    calib_items = None
    if a.calib_file:
        d = json.load(open(a.calib_file, encoding="utf-8"))
        gen = (d.get("train") or []) + (d.get("val") or [])
        calib_items = []
        for g in gen:
            subs, conf = router.tier1(g["query"])
            calib_items.append({"conf": conf, "shape": router.shape(subs),
                                "correct": bool(subs) and subs[0].tool == g["tool_id"]})
        calib_desc = f"{len(calib_items)} LLM-generated queries ({os.path.basename(a.calib_file)})"
    print(f"Calibration: {calib_desc}\nGate: consensus with {ret_desc}")

    def taus_for(eps):
        if calib_items is not None:
            t = tau_by_shape(calib_items, eps)
            return [t[it["shape"]] for it in items], t
        ts = cross_fit(items, eps, a.folds)
        per = {sh: statistics.median([t for t, it in zip(ts, items) if it["shape"] == sh] or [math.inf]) for sh in (1, 2)}
        return ts, per

    # ---- fallbacks ----
    fallbacks = {}
    if a.fallback_dir:
        for m in a.fallback_methods.split(","):
            s_rows, s_file = load_fallback(a.fallback_dir, m, "single", a.fallback_model)
            m_rows, m_file = load_fallback(a.fallback_dir, m, "multi", a.fallback_model)
            if s_rows or m_rows:
                fallbacks[m] = {"single": s_rows or {}, "multi": m_rows or {}}
                print(f"  fallback '{m}': {s_file} | {m_file}")
    assumed = dict(kv.split("=") for kv in a.assume_fallback.split(","))
    assumed = {k: float(v) / 100 for k, v in assumed.items()}

    # ---- sweep eps ----
    curve, main = [], None
    for eps in sorted(set(EPS_GRID + [a.eps])):
        taus, tau_med = taus_for(eps)
        row = {"eps": eps, "tau1": tau_med[1], "tau2": tau_med[2]}
        for kind in ("single", "multi"):
            idx = [i for i, it in enumerate(items) if it["kind"] == kind]
            det = [i for i in idx if items[i]["conf"] >= taus[i]]
            row[f"{kind}_det_rate"] = 100 * len(det) / len(idx)
            row[f"{kind}_det_prec"] = 100 * sum(items[i]["correct"] for i in det) / len(det) if det else float("nan")
            row[f"{kind}_det_correct"] = sum(items[i]["correct"] for i in det)
            row[f"{kind}_n"] = len(idx)
            esc = len(idx) - len(det)
            row[f"{kind}_est_acc"] = 100 * (row[f"{kind}_det_correct"] + esc * assumed.get(kind, 0)) / len(idx)
            for m, fb in fallbacks.items():
                rows = fb[kind]
                if not rows:
                    continue
                ok_router, ok_base, ptok_r, ptok_b, calls = [], [], [], [], 0
                for i in idx:
                    q = items[i]["q"]
                    r = rows.get(q.qid)
                    if r is None:
                        continue
                    base_ok = bool(r["correct"])
                    if items[i]["conf"] >= taus[i]:
                        ok_router.append(items[i]["correct"]); ptok_r.append(0)
                    else:
                        ok_router.append(base_ok); ptok_r.append(r.get("prompt_tokens", 0)); calls += 1
                    ok_base.append(base_ok); ptok_b.append(r.get("prompt_tokens", 0))
                n = len(ok_router)
                b01, b10, p = mcnemar_exact(ok_router, ok_base)
                row[f"{kind}_{m}_router_acc"] = 100 * sum(ok_router) / n
                row[f"{kind}_{m}_base_acc"] = 100 * sum(ok_base) / n
                row[f"{kind}_{m}_llm_rate"] = 100 * calls / n
                row[f"{kind}_{m}_router_ptok"] = sum(ptok_r) / n
                row[f"{kind}_{m}_base_ptok"] = sum(ptok_b) / n
                row[f"{kind}_{m}_mcnemar"] = (b01, b10, p)
                row[f"{kind}_{m}_n"] = n
                row[f"{kind}_{m}_ci"] = bootstrap_ci([float(x) for x in ok_router])
        curve.append(row)
        if eps == a.eps:
            main = (row, taus)

    # ---- report ----
    print("\nAccuracy-cost curve (each eps = target error of the deterministic path; tau from calibration)")
    print(f"{'eps':>5} {'tau(1)':>6} {'tau(2+)':>7} | {'single: 0-LLM %':>15} {'det prec %':>10} | {'multi: 0-LLM %':>14} {'det prec %':>10}")
    for r in curve:
        print(f"{r['eps']:>5.2f} {r['tau1']:>6.3f} {r['tau2']:>7.3f} | {r['single_det_rate']:>15.1f} {r['single_det_prec']:>10.1f} | "
              f"{r['multi_det_rate']:>14.1f} {r['multi_det_prec']:>10.1f}")

    row, taus = main
    print(f"\nMain setting: eps = {a.eps} ({calib_desc})")
    if fallbacks:
        for m in fallbacks:
            print(f"\n  Router + {m} fallback vs {m} alone (Qwen3-32B results reused, no new calls):")
            for kind in ("single", "multi"):
                if f"{kind}_{m}_router_acc" not in row:
                    continue
                b01, b10, p = row[f"{kind}_{m}_mcnemar"]
                lo, hi = row[f"{kind}_{m}_ci"]
                print(f"   {kind:6} acc {row[f'{kind}_{m}_router_acc']:.1f}% [95% CI {100*lo:.1f}-{100*hi:.1f}] vs "
                      f"{row[f'{kind}_{m}_base_acc']:.1f}% | LLM calls {row[f'{kind}_{m}_llm_rate']:.1f}% vs 100% | "
                      f"prompt tok/query {row[f'{kind}_{m}_router_ptok']:.0f} vs {row[f'{kind}_{m}_base_ptok']:.0f} | "
                      f"McNemar router-only-right {b01}, base-only-right {b10}, p={p:.3f}")
    else:
        print("  (no --fallback-dir: ESTIMATE assuming escalated queries are solved at the overall Hi-RAG rate "
              f"{a.assume_fallback}; this is optimistic because escalated queries are the harder ones)")
        for kind in ("single", "multi"):
            print(f"   {kind:6} deterministic {row[f'{kind}_det_rate']:.1f}% of queries at {row[f'{kind}_det_prec']:.1f}% "
                  f"precision -> LLM-call rate {100-row[f'{kind}_det_rate']:.1f}% | est. accuracy {row[f'{kind}_est_acc']:.1f}%")

    # ---- save audit log + curve ----
    os.makedirs(a.out_dir, exist_ok=True)
    log = []
    for it, t in zip(items, taus):
        log.append({"qid": it["q"].qid, "kind": it["kind"], "query": it["q"].text, "gold": it["q"].gold_tools,
                    "tier": "deterministic" if it["conf"] >= t else "fallback", "tau": t, "confidence": it["conf"],
                    "tier1_tools": it["tools"], "tier1_correct": it["correct"],
                    "subs": [{"text": s.text, "tool": s.tool, "confidence": s.confidence, "runner_up": s.runner_up,
                              "retriever_tool": s.retriever_tool, "evidence": s.parts} for s in it["subs"]]})
    tag = f"router_{a.retriever}" + ("_nosplit" if a.no_split else "") + ("_gencalib" if a.calib_file else "_crossfit")
    json.dump({"settings": vars(a), "calibration": calib_desc, "gate": ret_desc, "tier1_ms_per_query": ms,
               "curve": [{k: v for k, v in r.items() if not isinstance(v, tuple)} for r in curve], "decisions": log},
              open(os.path.join(a.out_dir, tag + ".json"), "w"), indent=1, ensure_ascii=False, default=str)
    with open(os.path.join(a.out_dir, tag + "_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in curve[0] if not isinstance(curve[0][k], tuple)], extrasaction="ignore")
        w.writeheader()
        w.writerows(curve)
    print(f"\nsaved {os.path.join(a.out_dir, tag + '.json')} (per-query audit log) and {tag}_curve.csv")


if __name__ == "__main__":
    main()
