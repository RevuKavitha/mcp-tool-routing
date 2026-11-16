"""Self-distillation experiment: how many requests skip the LLM as the router learns from fallback logs.

Logs = the LLM-generated Stage 2 queries (train_queries.json, near-copies of test queries already removed):
  'train' items simulate the (query, tool chosen by the LLM) log; 'val' items calibrate tau (risk control).
Test = all 261 MCPBench queries (never used for learning or calibration).
--noise r replaces a fraction r of log labels by a plausible wrong tool (BM25 runner-up) to mimic LLM mistakes
(Hi-RAG + Qwen3-32B is ~85% correct on single-service queries, i.e. ~15% noisy logs).

  python scripts/run_distill.py --calib-file train_queries.json
  python scripts/run_distill.py --calib-file train_queries.json --fallback-dir hirag_results_or --noise 0.15
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hirag.calibrate import tau_by_shape                    # noqa: E402
from hirag.data import load_mcpbench                        # noqa: E402
from hirag.distill import Distiller, explain_learned, with_learned   # noqa: E402
from hirag.registry import build_registry                   # noqa: E402
from hirag.router import HybridRouter                       # noqa: E402
from hirag.sparse import BM25Retriever                      # noqa: E402


def load_rows(dirpath, method, split, model):
    if not dirpath:
        return {}
    k = 1 if split == "single" else 3
    files = sorted(f for f in glob.glob(os.path.join(dirpath, f"e2e_{method}_{split}_k{k}_*.json"))
                   if model in os.path.basename(f) and "PARTIAL" not in f)
    return {r["qid"]: r for r in json.load(open(files[0]))["rows"]} if files else {}


def det_correct(tools, q):
    if q.kind == "single":
        return bool(tools) and tools[0] == q.gold_tools[0]
    return all(g in tools for g in q.gold_tools)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-file", required=True, help="train_queries.json from notebook 03")
    ap.add_argument("--fallback-dir", default=None)
    ap.add_argument("--fallback-method", default="hirag")
    ap.add_argument("--fallback-model", default="qwen3-32b")
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "none"], help="consensus gate retriever")
    ap.add_argument("--sizes", default="0,100,200,400,800,all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--eps", default="0.05,0.10")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--min-purity", type=float, default=0.6)
    ap.add_argument("--compose-multi", type=int, default=0,
                    help="add N two-step calibration requests made by joining two held-out queries ('X, then Y')")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    bench = load_mcpbench()
    reg0 = build_registry(bench)
    ids = [t.tool_id for t in bench.tools]
    bm = BM25Retriever([t.summary for t in bench.tools], "clean")
    bm_cache = {}

    def bm_rank(text, n=2):
        if text not in bm_cache:
            bm_cache[text] = [ids[i] for i in bm.rank(text, 5)]
        return bm_cache[text][:n]

    retr = (lambda text: bm_rank(text, 1)[0]) if a.retriever == "bm25" else None
    d = json.load(open(a.calib_file, encoding="utf-8"))
    train = [(x["query"], x["tool_id"]) for x in d.get("train", [])]
    val = [(x["query"], x["tool_id"]) for x in d.get("val", [])]
    if not val:                                           # no val split saved: hold out 20% of train
        rnd = random.Random(123)
        rnd.shuffle(train)
        cut = len(train) // 5
        val, train = train[:cut], train[cut:]
    composed = []                                         # (text, [tool1, tool2]) for calibrating multi-step tau
    if a.compose_multi:
        rnd = random.Random(7)
        svc = {t.tool_id: t.service for t in bench.tools}
        joins = [", then ", " and then ", ". After that, ", ", and after that "]
        while len(composed) < a.compose_multi:
            (q1, t1), (q2, t2) = rnd.sample(val, 2)
            if svc[t1] == svc[t2]:
                continue
            j = rnd.choice(joins)
            q2 = q2[0].lower() + q2[1:] if not j.endswith(". After that, ") else q2
            composed.append((q1.rstrip(" .?!") + j + q2, [t1, t2]))
    queries = bench.single + bench.multi
    fb = {s: load_rows(a.fallback_dir, a.fallback_method, s, a.fallback_model) for s in ("single", "multi")}
    eps_list = [float(e) for e in a.eps.split(",")]
    sizes = [len(train) if s == "all" else min(int(s), len(train)) for s in a.sizes.split(",")]
    print(f"composed two-step calibration requests: {len(composed)}")
    print(f"logs: {len(train)} train / {len(val)} val generated queries | test: {len(queries)} MCPBench queries | "
          f"gate: {'consensus with BM25' if retr else 'rules only'} | label noise {a.noise:.0%}")
    if fb["single"]:
        print(f"fallback: per-query {a.fallback_method} results ({a.fallback_model})")

    results = []
    example_learned = None
    for size in sizes:
        per_seed = []
        for seed in range(a.seeds if size not in (0, len(train)) or a.noise > 0 else 1):
            rnd = random.Random(seed)
            logs = train[:]
            rnd.shuffle(logs)
            logs = logs[:size]
            if a.noise > 0:                                # simulate LLM mistakes in the log
                logs = [(q, next((t for t in bm_rank(q, 5) if t != tool), tool)) if rnd.random() < a.noise else (q, tool)
                        for q, tool in logs]
            learned = Distiller(a.min_count, a.min_purity).fit(logs) if size else {}
            reg = with_learned(reg0, learned) if learned else reg0
            if size == len(train) and seed == 0:
                example_learned = learned
            router = HybridRouter(reg, tau=math.inf, retriever=retr)
            cal = []
            for q, tool in val:
                subs, conf = router.tier1(q)
                cal.append({"conf": conf, "shape": router.shape(subs), "correct": bool(subs) and subs[0].tool == tool})
            n_split = 0
            for q, tools in composed:
                subs, conf = router.tier1(q)
                if router.shape(subs) == 2:            # composed requests calibrate only the multi-step tau;
                    n_split += 1                       # single-step tau is calibrated on single requests only
                    cal.append({"conf": conf, "shape": 2, "correct": all(t in [x.tool for x in subs] for t in tools)})
            if composed and seed == 0:
                print(f"  [{size} logs] orchestrator split {n_split}/{len(composed)} composed two-step requests "
                      f"({100*n_split/len(composed):.0f}%)")
            test = []
            for q in queries:
                subs, conf = router.tier1(q.text)
                test.append((q, conf, router.shape(subs), det_correct([s.tool for s in subs], q)))
            for eps in eps_list:
                taus = tau_by_shape(cal, eps)
                row = {"size": size, "seed": seed, "eps": eps, "n_rules_learned": sum(len(v) for v in learned.values())}
                for kind in ("single", "multi"):
                    sub = [t for t in test if t[0].kind == kind]
                    det = [t for t in sub if t[1] >= taus[t[2]]]
                    row[f"{kind}_det_rate"] = 100 * len(det) / len(sub)
                    row[f"{kind}_det_prec"] = 100 * sum(t[3] for t in det) / len(det) if det else float("nan")
                    rows = fb[kind]
                    if rows:
                        ok = [(t[3] if t[1] >= taus[t[2]] else bool(rows[t[0].qid]["correct"])) for t in sub if t[0].qid in rows]
                        base = [bool(rows[t[0].qid]["correct"]) for t in sub if t[0].qid in rows]
                        tok = [(0 if t[1] >= taus[t[2]] else rows[t[0].qid].get("prompt_tokens", 0)) for t in sub if t[0].qid in rows]
                        row[f"{kind}_acc"] = 100 * sum(ok) / len(ok)
                        row[f"{kind}_base_acc"] = 100 * sum(base) / len(base)
                        row[f"{kind}_ptok"] = sum(tok) / len(tok)
                per_seed.append(row)
        for eps in eps_list:
            rs = [r for r in per_seed if r["eps"] == eps]
            avg = {"size": size, "eps": eps, "seeds": len(rs)}
            for k in rs[0]:
                if k in ("size", "seed", "eps"):
                    continue
                vals = [r[k] for r in rs if not (isinstance(r[k], float) and math.isnan(r[k]))]
                avg[k] = statistics.mean(vals) if vals else float("nan")
                avg[k + "_sd"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            results.append(avg)

    for eps in eps_list:
        print(f"\neps = {eps}  (tau calibrated on {len(val)} held-out generated queries; mean over seeds)")
        head = f"{'logs':>5} {'rules':>6} | {'single 0-LLM %':>14} {'prec %':>7}"
        head += f" {'acc %':>6} {'(base)':>8} {'tok/q':>6}" if fb["single"] else ""
        head += f" | {'multi 0-LLM %':>13} {'prec %':>7}"
        print(head)
        for r in [r for r in results if r["eps"] == eps]:
            line = (f"{r['size']:>5} {r['n_rules_learned']:>6.0f} | {r['single_det_rate']:>9.1f}±{r['single_det_rate_sd']:<4.1f}"
                    f" {r['single_det_prec']:>7.1f}")
            if fb["single"]:
                line += f" {r['single_acc']:>6.1f} {r['single_base_acc']:>8.1f} {r['single_ptok']:>6.0f}"
            line += f" | {r['multi_det_rate']:>13.1f} {r['multi_det_prec']:>7.1f}"
            print(line)
    if example_learned:
        print("\nExamples of learned rules (word or word-pair -> tool, purity):")
        for t in sorted(example_learned, key=lambda t: -len(example_learned[t]))[:5]:
            print("  " + explain_learned(example_learned, t))
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"distill_{a.retriever}_noise{int(100*a.noise)}" + (f"_multi{a.compose_multi}" if a.compose_multi else "")
    json.dump({"settings": vars(a), "results": results, "learned_rules_all_logs": example_learned},
              open(os.path.join(a.out_dir, tag + ".json"), "w"), indent=1)
    print("saved", os.path.join(a.out_dir, tag + ".json"))


if __name__ == "__main__":
    main()
