"""Reproduce Stage 1 retrieval results (paper Table 3 and the paper-vs-code settings).

Examples
--------
# On Colab/Kaggle GPU with the paper's model:
python scripts/run_stage1.py --model BAAI/bge-large-en --backend st

# On CPU with a local ONNX export of bge-small (testing only):
python scripts/run_stage1.py --model /path/to/bge-small-onnx --backend onnx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hirag.data import load_mcpbench                       # noqa: E402
from hirag.dense import BGE_QUERY_INSTRUCTION, Embedder    # noqa: E402
from hirag.metrics import format_table, hit_at_k, mean, recall_at_k, unique_in_order  # noqa: E402
from hirag.stage1 import Stage1, Stage1Config              # noqa: E402

# Settings to compare. name -> overrides of Stage1Config
SETTINGS = {
    "BM25 only (jieba, authors)":          dict(fusion="bm25", tokenizer="jieba"),
    "BM25 only (clean tokenizer)":         dict(fusion="bm25", tokenizer="clean"),
    "Dense only":                          dict(fusion="dense"),
    "Hybrid: code RRF w=0.1 (authors)":    dict(fusion="code", weight=0.1, tokenizer="jieba"),
    "Hybrid: paper W-RRF a=0.1 (jieba)":   dict(fusion="paper", weight=0.1, tokenizer="jieba"),
    "Hybrid: paper W-RRF a=0.7 (jieba)":   dict(fusion="paper", weight=0.7, tokenizer="jieba"),
    "Hybrid: code RRF w=0.1 (clean)":      dict(fusion="code", weight=0.1, tokenizer="clean"),
    "Hybrid: paper W-RRF a=0.5 (clean)":   dict(fusion="paper", weight=0.5, tokenizer="clean"),
}


def evaluate(bench, stage1: Stage1):
    so = stage1.service_of
    s1, t1, t10, cand_s, ncand = [], [], [], [], []
    for q in bench.single:
        r = stage1.run(q.text)
        svc_rank = unique_in_order([so[t] for t in r.tools])
        s1.append(hit_at_k(svc_rank, q.gold_services, 1))
        t1.append(hit_at_k(r.tools, q.gold_tools, 1))
        t10.append(hit_at_k(r.tools, q.gold_tools, 10))
        cand_s.append(recall_at_k(list(r.candidates), q.gold_services, len(r.candidates)))
        ncand.append(len(r.candidates))
    m_tool, m_cand = [], []
    for q in bench.multi:
        r = stage1.run(q.text)
        m_tool.append(recall_at_k(r.tools, q.gold_tools, 10))
        m_cand.append(recall_at_k(list(r.candidates), q.gold_services, len(r.candidates)))
        ncand.append(len(r.candidates))
    return {
        "Service@1 (single)": mean(s1),
        "Tool@1 (single)": mean(t1),
        "Tool@10 (single)": mean(t10),
        "Tool R@10 (multi)": mean(m_tool),
        "S_cand recall (single)": mean(cand_s),
        "S_cand recall (multi)": mean(m_cand),
        "Avg |S_cand|": sum(ncand) / len(ncand),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-large-en", help="HF model id (st) or ONNX folder (onnx)")
    ap.add_argument("--backend", default="st", choices=["st", "onnx"])
    ap.add_argument("--query-instruction", action="store_true", help="prepend BGE's query instruction")
    ap.add_argument("--only", nargs="*", help="run only settings whose name contains these strings")
    ap.add_argument("--out", default="results/stage1.json")
    args = ap.parse_args()

    bench = load_mcpbench()
    print(bench.summary())
    emb = Embedder(args.model, args.backend, BGE_QUERY_INSTRUCTION if args.query_instruction else "")

    rows, stages = [], {}
    all_queries = [q.text for q in bench.single + bench.multi]
    for name, over in SETTINGS.items():
        if args.only and not any(s in name for s in args.only):
            continue
        cfg = replace(Stage1Config(), **over)
        key = (cfg.tokenizer,)  # BM25 index depends only on the tokenizer
        if key not in stages:
            stages[key] = Stage1(bench, emb, cfg)
            stages[key].precompute_dense(all_queries)
        st = stages[key]
        st.cfg = cfg
        res = evaluate(bench, st)
        rows.append({"Setting": name, **res})
        print(f"done: {name}")

    cols = list(rows[0].keys())
    print("\nEmbedding model:", args.model)
    print(format_table(rows, cols))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"model": args.model, "rows": rows, "settings": SETTINGS}, open(args.out, "w"), indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
